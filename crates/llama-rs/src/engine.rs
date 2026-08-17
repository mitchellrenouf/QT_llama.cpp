use crate::sys::*;
use anyhow::{anyhow, Result};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

struct EngineContextHandle {
    ptr: *mut ggml_engine_context_t,
}

unsafe impl Send for EngineContextHandle {}
unsafe impl Sync for EngineContextHandle {}

impl Drop for EngineContextHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ggml_engine_free(self.ptr) };
        }
    }
}

#[derive(Clone)]
pub struct LlamaEngine {
    handle: Arc<Mutex<EngineContextHandle>>,
    n_vocab: usize,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    pub fn new(model_path: &Path, n_gpu_layers: i32, n_ctx: u32, backend: Option<&str>) -> Result<Self> {
        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path UTF-8"))?;
        let c_path = CString::new(path_str)?;

        let backend_choice = backend
            .map(|s| s.to_string())
            .or_else(|| std::env::var("QT_LLAMA_BACKEND").ok())
            .or_else(|| std::env::var("LLAMA_BACKEND").ok())
            .unwrap_or_else(|| "auto".to_string());
        let c_backend = CString::new(backend_choice.as_str())?;

        let ctx_size = if n_ctx > 0 && n_ctx <= 32768 { n_ctx } else { 8192 };

        let ptr = unsafe {
            ggml_engine_init(
                c_path.as_ptr(),
                n_gpu_layers,
                ctx_size,
                c_backend.as_ptr(),
            )
        };

        if ptr.is_null() {
            return Err(anyhow!("Failed to initialize GGML model engine from {}", model_path.display()));
        }

        let n_vocab = unsafe { ggml_engine_get_n_vocab(ptr) } as usize;

        Ok(Self {
            handle: Arc::new(Mutex::new(EngineContextHandle { ptr })),
            n_vocab: if n_vocab > 0 { n_vocab } else { 256000 },
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<llama_token>> {
        let guard = self.handle.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let c_text = CString::new(text)?;
        let mut tokens = vec![0i32; text.len() + 64];

        let count = unsafe {
            ggml_engine_tokenize(
                guard.ptr,
                c_text.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };

        if count < 0 {
            return Err(anyhow!("Tokenization error code: {}", count));
        }

        tokens.truncate(count as usize);
        Ok(tokens)
    }

    pub fn token_to_piece(&self, token: llama_token) -> Result<String> {
        let guard = self.handle.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut buf = vec![0u8; 256];

        let n = unsafe {
            ggml_engine_token_to_piece(
                guard.ptr,
                token,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as i32,
            )
        };

        if n <= 0 {
            return Ok(String::new());
        }

        buf.truncate(n as usize);
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    pub fn is_eog(&self, token: llama_token) -> bool {
        if let Ok(guard) = self.handle.lock() {
            unsafe { ggml_engine_is_eog(guard.ptr, token) }
        } else {
            true
        }
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(64);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancelled.clone();

        let engine = self.clone();
        let prompt_string = prompt.to_string();

        std::thread::spawn(move || {
            let prompt_tokens = match engine.tokenize(&prompt_string, true, true) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.blocking_send(Err(anyhow!("Failed to tokenize prompt: {}", e)));
                    return;
                }
            };

            let guard = match engine.handle.lock() {
                Ok(g) => g,
                Err(_) => {
                    let _ = tx.blocking_send(Err(anyhow!("Poisoned GGML context mutex")));
                    return;
                }
            };
            let ctx = guard.ptr;

            // Clear KV cache before evaluating new sequence
            unsafe { ggml_engine_kv_cache_clear(ctx) };

            let mut logits = vec![0.0f32; engine.n_vocab];

            // 1. Evaluate prompt in chunks
            let n_batch = 2048;
            let mut i = 0;
            while i < prompt_tokens.len() {
                if cancel_flag.load(Ordering::Relaxed) {
                    return;
                }

                let cur_batch_size = (prompt_tokens.len() - i).min(n_batch);
                let chunk = &prompt_tokens[i..i + cur_batch_size];

                let is_last = (i + cur_batch_size) == prompt_tokens.len();
                let out_ptr = if is_last { logits.as_mut_ptr() } else { std::ptr::null_mut() };

                let ret = unsafe {
                    ggml_engine_eval(
                        ctx,
                        chunk.as_ptr(),
                        cur_batch_size as i32,
                        i as i32,
                        out_ptr,
                    )
                };

                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("GGML forward evaluation failed on prompt")));
                    return;
                }

                i += cur_batch_size;
            }

            // 2. Autoregressive token generation loop
            let mut n_past = prompt_tokens.len();
            let mut generated = 0;

            while generated < max_tokens {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }

                // Sample token from logits
                let next_token = unsafe {
                    ggml_engine_sample(ctx, logits.as_ptr(), temperature, 0.95)
                };

                if next_token == LLAMA_TOKEN_NULL || unsafe { ggml_engine_is_eog(ctx, next_token) } {
                    break;
                }

                // Detokenize piece
                let piece = match engine.token_to_piece(next_token) {
                    Ok(p) => p,
                    Err(_) => String::new(),
                };

                if tx.blocking_send(Ok(piece)).is_err() {
                    break;
                }

                generated += 1;

                // Forward pass for next token
                let token_arr = [next_token];
                let ret = unsafe {
                    ggml_engine_eval(
                        ctx,
                        token_arr.as_ptr(),
                        1,
                        n_past as i32,
                        logits.as_mut_ptr(),
                    )
                };

                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("GGML forward evaluation failed on token generation")));
                    break;
                }

                n_past += 1;
            }
        });

        (rx, cancelled)
    }
}
