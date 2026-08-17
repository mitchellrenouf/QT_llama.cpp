use crate::sys::*;
use anyhow::{anyhow, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub struct LlamaModelHandle {
    ptr: *mut llama_model,
}

unsafe impl Send for LlamaModelHandle {}
unsafe impl Sync for LlamaModelHandle {}

impl Drop for LlamaModelHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_model_free(self.ptr) };
        }
    }
}

pub struct LlamaContextHandle {
    ptr: *mut llama_context,
}

unsafe impl Send for LlamaContextHandle {}
unsafe impl Sync for LlamaContextHandle {}

impl Drop for LlamaContextHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { llama_free(self.ptr) };
        }
    }
}

#[derive(Clone)]
pub struct LlamaEngine {
    #[allow(dead_code)]
    model: Arc<LlamaModelHandle>,
    context: Arc<Mutex<LlamaContextHandle>>,
    vocab: *const llama_vocab,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

static BACKEND_INIT: std::sync::Once = std::sync::Once::new();

unsafe extern "C" fn quiet_llama_log(level: c_int, text: *const c_char, _user_data: *mut c_void) {
    // Only pass through actual errors (level >= 3), filter out verbose CUDA Graph reuse messages
    if level >= 3 && !text.is_null() {
        let s = std::ffi::CStr::from_ptr(text).to_string_lossy();
        if !s.contains("CUDA Graph") {
            eprint!("{}", s);
        }
    }
}

impl LlamaEngine {
    pub fn new(model_path: &Path, n_gpu_layers: i32, n_ctx: u32, backend: Option<&str>) -> Result<Self> {
        BACKEND_INIT.call_once(|| unsafe {
            llama_backend_init();
            llama_log_set(Some(quiet_llama_log), std::ptr::null_mut());
            ggml_log_set(Some(quiet_llama_log), std::ptr::null_mut());
        });

        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path UTF-8"))?;
        let c_path = CString::new(path_str)?;

        let backend_choice = backend
            .map(|s| s.to_string())
            .or_else(|| std::env::var("QT_LLAMA_BACKEND").ok())
            .or_else(|| std::env::var("LLAMA_BACKEND").ok())
            .unwrap_or_else(|| "auto".to_string());
        let _c_backend = CString::new(backend_choice.as_str())?;

        let env_override = std::env::var("LLAMA_GPU_LAYERS")
            .ok()
            .or_else(|| std::env::var("QT_LLAMA_GPU_LAYERS").ok())
            .and_then(|s| s.parse::<i32>().ok());

        let num_threads = std::thread::available_parallelism()
            .map(|p| p.get() as i32)
            .unwrap_or(8)
            .min(16);

        let mut m_params = unsafe { llama_model_default_params() };
        let mut c_params = unsafe { llama_context_default_params() };
        let target_ctx = if n_ctx > 0 && n_ctx <= 32768 {
            n_ctx
        } else {
            8192
        };

        c_params.n_ctx = target_ctx;
        c_params.n_batch = 2048;
        c_params.n_ubatch = 512;
        c_params.flash_attn_type = 1; // LLAMA_FLASH_ATTN_TYPE_ENABLED
        c_params.offload_kqv = true;  // Keep KV cache fully in GPU VRAM
        c_params.op_offload = true;   // Offload all tensor operations
        c_params.swa_full = false;    // Use sliding window size, do NOT allocate full 256k buffer
        c_params.n_threads = num_threads;
        c_params.n_threads_batch = num_threads;
        c_params.no_perf = true;
        m_params.load_mode = -1;      // LLAMA_LOAD_MODE_AUTO (mmap for zero-copy memory mapping)

        if let Some(layers) = env_override {
            m_params.n_gpu_layers = layers;
        } else if n_gpu_layers >= 0 {
            m_params.n_gpu_layers = n_gpu_layers;
        } else {
            // Offload all layers to GPU by default for maximum token generation speed (matching llama-server -ngl 99)
            if backend_choice != "cpu" {
                m_params.n_gpu_layers = 999;
            } else {
                m_params.n_gpu_layers = 0;
            }
        }

        eprintln!(
            "[llama.cpp] Loading model '{}' (n_gpu_layers = {}, n_ctx = {})...",
            model_path.display(),
            m_params.n_gpu_layers,
            c_params.n_ctx
        );

        let model_ptr = unsafe { llama_model_load_from_file(c_path.as_ptr(), m_params) };
        if model_ptr.is_null() {
            return Err(anyhow!("Failed to load GGUF model from {}", model_path.display()));
        }

        let total_layers = unsafe { llama_model_n_layer(model_ptr) };
        let vocab_ptr = unsafe { llama_model_get_vocab(model_ptr) };

        // Try initializing context in GPU VRAM with target_ctx
        let mut ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };

        // If tight on VRAM, scale down context size while keeping KV cache 100% in GPU VRAM
        if ctx_ptr.is_null() && c_params.n_ctx > 4096 {
            eprintln!("[llama.cpp] Context size {} exceeded VRAM, retrying with 4096 tokens (GPU KV cache)...", c_params.n_ctx);
            c_params.n_ctx = 4096;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        if ctx_ptr.is_null() && c_params.n_ctx > 2048 {
            eprintln!("[llama.cpp] Retrying with 2048 tokens (GPU KV cache)...");
            c_params.n_ctx = 2048;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        // Only fall back to CPU KV cache if GPU VRAM is completely full
        if ctx_ptr.is_null() {
            eprintln!("[llama.cpp] GPU VRAM exhausted for KV cache, falling back to CPU KV cache...");
            c_params.n_ctx = target_ctx;
            c_params.offload_kqv = false;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        if ctx_ptr.is_null() {
            unsafe { llama_model_free(model_ptr) };
            return Err(anyhow!("Failed to initialize llama_context from model"));
        }

        eprintln!(
            "[llama.cpp] Successfully initialized model & context (total layers: {})",
            total_layers
        );

        Ok(Self {
            model: Arc::new(LlamaModelHandle { ptr: model_ptr }),
            context: Arc::new(Mutex::new(LlamaContextHandle { ptr: ctx_ptr })),
            vocab: vocab_ptr,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<llama_token>> {
        let c_text = CString::new(text)?;
        let mut tokens: Vec<llama_token> = vec![0; text.len() + 32];

        let n = unsafe {
            llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };

        if n < 0 {
            tokens.resize((-n) as usize, 0);
            let n_retry = unsafe {
                llama_tokenize(
                    self.vocab,
                    c_text.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    parse_special,
                )
            };
            if n_retry < 0 {
                return Err(anyhow!("Tokenization failed with error code {}", n_retry));
            }
            tokens.truncate(n_retry as usize);
        } else {
            tokens.truncate(n as usize);
        }

        Ok(tokens)
    }

    pub fn token_to_piece(&self, token: llama_token) -> Result<String> {
        let mut buf = vec![0u8; 128];
        let n = unsafe {
            llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as i32,
                0,
                true,
            )
        };

        if n < 0 {
            buf.resize((-n) as usize, 0);
            let n2 = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n2 < 0 {
                return Ok(String::new());
            }
            buf.truncate(n2 as usize);
        } else {
            buf.truncate(n as usize);
        }

        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    pub fn is_eog(&self, token: llama_token) -> bool {
        unsafe { llama_vocab_is_eog(self.vocab, token) }
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

            let ctx_guard = match engine.context.lock() {
                Ok(g) => g,
                Err(_) => {
                    let _ = tx.blocking_send(Err(anyhow!("Poisoned llama context mutex")));
                    return;
                }
            };
            let ctx = ctx_guard.ptr;

            // Reset KV cache / memory module for clean prompt evaluation
            unsafe {
                let mem = llama_get_memory(ctx);
                if !mem.is_null() {
                    llama_memory_clear(mem, true);
                }
            }

            // Initialize Sampler
            let smpl = unsafe {
                let chain_params = llama_sampler_chain_default_params();
                let chain = llama_sampler_chain_init(chain_params);
                if temperature <= 0.0 {
                    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
                } else {
                    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
                    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
                    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                }
                chain
            };

            // Evaluate prompt tokens in batches
            let n_batch = 2048;
            let mut batch = unsafe { llama_batch_init(n_batch, 0, 1) };

            let mut i = 0;
            while i < prompt_tokens.len() {
                if cancel_flag.load(Ordering::Relaxed) {
                    unsafe {
                        llama_batch_free(batch);
                        llama_sampler_free(smpl);
                    }
                    return;
                }

                let cur_batch_size = (prompt_tokens.len() - i).min(n_batch as usize);
                batch.n_tokens = 0;

                for j in 0..cur_batch_size {
                    let token = prompt_tokens[i + j];
                    let is_last = (i + j) == (prompt_tokens.len() - 1);
                    unsafe {
                        *batch.token.add(j) = token;
                        *batch.pos.add(j) = (i + j) as llama_pos;
                        *batch.n_seq_id.add(j) = 1;
                        **batch.seq_id.add(j) = 0;
                        *batch.logits.add(j) = if is_last { 1 } else { 0 };
                    }
                    batch.n_tokens += 1;
                }

                let ret = unsafe { llama_decode(ctx, batch) };
                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("llama_decode failed on prompt evaluation")));
                    unsafe {
                        llama_batch_free(batch);
                        llama_sampler_free(smpl);
                    }
                    return;
                }

                i += cur_batch_size;
            }

            // Generation loop
            let mut n_cur = prompt_tokens.len();
            let mut generated = 0;

            while generated < max_tokens {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }

                let token = unsafe { llama_sampler_sample(smpl, ctx, -1) };
                if token == LLAMA_TOKEN_NULL || engine.is_eog(token) {
                    break;
                }

                let piece = match engine.token_to_piece(token) {
                    Ok(p) => p,
                    Err(_) => String::new(),
                };

                if tx.blocking_send(Ok(piece)).is_err() {
                    break;
                }

                generated += 1;

                // Prepare next token decode
                batch.n_tokens = 0;
                unsafe {
                    *batch.token = token;
                    *batch.pos = n_cur as llama_pos;
                    *batch.n_seq_id = 1;
                    **batch.seq_id = 0;
                    *batch.logits = 1;
                }
                batch.n_tokens = 1;

                let ret = unsafe { llama_decode(ctx, batch) };
                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("llama_decode failed on generation token")));
                    break;
                }

                n_cur += 1;
            }

            unsafe {
                llama_batch_free(batch);
                llama_sampler_free(smpl);
            }
        });

        (rx, cancelled)
    }
}
