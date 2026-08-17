use crate::sys::*;
use anyhow::{anyhow, Result};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

struct LlamaModelHandle {
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

struct LlamaContextHandle {
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
    _model: Arc<LlamaModelHandle>,
    context: Arc<Mutex<LlamaContextHandle>>,
    vocab: *const llama_vocab,
}

unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

unsafe extern "C" fn quiet_log_callback(
    level: i32,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    // Only display critical error messages, suppress CUDA graph / verbose trace
    if level <= 1 {
        if let Ok(s) = std::ffi::CStr::from_ptr(text).to_str() {
            if !s.contains("CUDA graph") && !s.contains("warmup") && !s.contains("reused") {
                eprint!("{}", s);
            }
        }
    }
}

impl LlamaEngine {
    pub fn new(
        model_path: &Path,
        n_gpu_layers: i32,
        n_ctx: u32,
        backend: Option<&str>,
    ) -> Result<Self> {
        unsafe {
            llama_backend_init();
            llama_log_set(Some(quiet_log_callback), std::ptr::null_mut());
        };

        let path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path UTF-8"))?;
        let c_path = CString::new(path_str)?;

        // Inspect GGUF structure using pure Rust qtensor engine
        if let Ok(gguf) = qtensor::gguf::GgufFile::open(model_path) {
            eprintln!(
                "[llama-rs + qtensor] Model Arch: '{}', Tensors: {}, Key-Values: {}",
                gguf.architecture(),
                gguf.tensors.len(),
                gguf.metadata.len()
            );
        }

        let backend_choice = backend
            .map(|s| s.to_string())
            .or_else(|| std::env::var("QT_LLAMA_BACKEND").ok())
            .or_else(|| std::env::var("LLAMA_BACKEND").ok())
            .unwrap_or_else(|| "auto".to_string());

        let _num_threads = std::thread::available_parallelism()
            .map(|p| p.get() as i32)
            .unwrap_or(4)
            .min(8);

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
        c_params.flash_attn_type = 1;
        c_params.type_k = 8; // GGML_TYPE_Q8_0
        c_params.type_v = 8; // GGML_TYPE_Q8_0
        c_params.offload_kqv = true;
        c_params.op_offload = true;
        c_params.swa_full = false;
        c_params.n_threads = 2;
        c_params.n_threads_batch = 2;
        c_params.no_perf = true;
        m_params.load_mode = -1; // mmap

        if n_gpu_layers >= 0 {
            m_params.n_gpu_layers = n_gpu_layers;
        } else if backend_choice != "cpu" {
            m_params.n_gpu_layers = 999;
        } else {
            m_params.n_gpu_layers = 0;
        }

        eprintln!(
            "[llama-rs] Loading GGUF model '{}' (n_gpu_layers = {}, n_ctx = {})...",
            model_path.display(),
            m_params.n_gpu_layers,
            c_params.n_ctx
        );

        let model_ptr = unsafe { llama_model_load_from_file(c_path.as_ptr(), m_params) };
        if model_ptr.is_null() {
            return Err(anyhow!("Failed to load GGUF model from {}", model_path.display()));
        }

        let vocab_ptr = unsafe { llama_model_get_vocab(model_ptr) };

        // Attempt GPU KV cache allocation
        let mut ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };

        if ctx_ptr.is_null() && c_params.n_ctx > 4096 {
            eprintln!("[llama-rs] Scaling context to 4096 tokens in VRAM...");
            c_params.n_ctx = 4096;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        if ctx_ptr.is_null() && c_params.n_ctx > 2048 {
            eprintln!("[llama-rs] Scaling context to 2048 tokens in VRAM...");
            c_params.n_ctx = 2048;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        if ctx_ptr.is_null() {
            eprintln!("[llama-rs] VRAM exhausted for KV cache, falling back to CPU KV cache...");
            c_params.n_ctx = target_ctx;
            c_params.offload_kqv = false;
            ctx_ptr = unsafe { llama_init_from_model(model_ptr, c_params) };
        }

        if ctx_ptr.is_null() {
            unsafe { llama_model_free(model_ptr) };
            return Err(anyhow!("Failed to initialize llama_context from model"));
        }

        eprintln!("[llama-rs] In-process engine initialized successfully with GPU acceleration!");

        Ok(Self {
            _model: Arc::new(LlamaModelHandle { ptr: model_ptr }),
            context: Arc::new(Mutex::new(LlamaContextHandle { ptr: ctx_ptr })),
            vocab: vocab_ptr,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<llama_token>> {
        let c_text = CString::new(text)?;
        let mut tokens: Vec<llama_token> = vec![0; text.len() + 64];

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
            let needed = -n as usize;
            tokens.resize(needed, 0);
            let n2 = unsafe {
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
            if n2 < 0 {
                return Err(anyhow!("Failed to tokenize text"));
            }
            tokens.truncate(n2 as usize);
        } else {
            tokens.truncate(n as usize);
        }

        Ok(tokens)
    }

    pub fn token_to_piece(&self, token: llama_token) -> Result<String> {
        let mut buf = [0u8; 128];
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

        if n > 0 {
            let slice = &buf[..n as usize];
            return Ok(String::from_utf8_lossy(slice).to_string());
        }

        if n < 0 {
            let needed = -n as usize;
            let mut dynamic_buf = vec![0u8; needed];
            let n2 = unsafe {
                llama_token_to_piece(
                    self.vocab,
                    token,
                    dynamic_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    dynamic_buf.len() as i32,
                    0,
                    true,
                )
            };
            if n2 > 0 {
                dynamic_buf.truncate(n2 as usize);
                return Ok(String::from_utf8_lossy(&dynamic_buf).to_string());
            }
        }

        Ok(String::new())
    }

    pub fn is_eog(&self, token: llama_token) -> bool {
        if token == LLAMA_TOKEN_NULL {
            return true;
        }
        let eos = unsafe { llama_vocab_eos(self.vocab) };
        let eot = unsafe { llama_vocab_eot(self.vocab) };
        if token == eos || token == eot {
            return true;
        }
        if unsafe { llama_vocab_is_eog(self.vocab, token) } {
            return true;
        }
        false
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(4096);
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

            let guard = match engine.context.lock() {
                Ok(g) => g,
                Err(_) => {
                    let _ = tx.blocking_send(Err(anyhow!("Context mutex poisoned")));
                    return;
                }
            };
            let ctx = guard.ptr;

            // Clear KV memory before evaluating new request
            unsafe {
                let mem = llama_get_memory(ctx);
                if !mem.is_null() {
                    llama_memory_clear(mem, true);
                }
            }

            // Create sampler chain
            let smpl_chain = unsafe {
                let chain = llama_sampler_chain_init(llama_sampler_chain_params::default());
                if temperature > 0.0 {
                    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
                    llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
                    llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                } else {
                    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
                }
                chain
            };

            // 1. Evaluate prompt batch
            let n_batch = 2048;
            let mut i = 0;
            while i < prompt_tokens.len() {
                if cancel_flag.load(Ordering::Relaxed) {
                    unsafe { llama_sampler_free(smpl_chain) };
                    return;
                }

                let cur_batch_size = (prompt_tokens.len() - i).min(n_batch);
                let mut batch = unsafe { llama_batch_init(cur_batch_size as i32, 0, 1) };
                batch.n_tokens = cur_batch_size as i32;

                for j in 0..cur_batch_size {
                    let token_idx = i + j;
                    unsafe {
                        *batch.token.add(j) = prompt_tokens[token_idx];
                        *batch.pos.add(j) = token_idx as i32;
                        *batch.n_seq_id.add(j) = 1;
                        **batch.seq_id.add(j) = 0;
                        *batch.logits.add(j) = if token_idx == prompt_tokens.len() - 1 { 1 } else { 0 };
                    }
                }

                let ret = unsafe { llama_decode(ctx, batch) };
                unsafe { llama_batch_free(batch) };

                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("llama_decode failed on prompt evaluation (code {})", ret)));
                    unsafe { llama_sampler_free(smpl_chain) };
                    return;
                }

                i += cur_batch_size;
            }

            // 2. Autoregressive token generation loop
            let mut n_past = prompt_tokens.len() as i32;
            let mut generated = 0;

            let mut batch_single = unsafe { llama_batch_init(1, 0, 1) };
            batch_single.n_tokens = 1;
            unsafe {
                *batch_single.n_seq_id = 1;
                **batch_single.seq_id = 0;
                *batch_single.logits = 1;
            }

            while generated < max_tokens {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }

                let token = unsafe { llama_sampler_sample(smpl_chain, ctx, -1) };

                if token == LLAMA_TOKEN_NULL || engine.is_eog(token) {
                    break;
                }

                let piece = match engine.token_to_piece(token) {
                    Ok(p) => p,
                    Err(_) => String::new(),
                };

                let trimmed = piece.trim();
                if trimmed == "<end_of_turn>"
                    || trimmed == "<|turn_end|>"
                    || trimmed == "<|im_end|>"
                    || trimmed == "</s>"
                    || trimmed == "<start_of_turn>"
                {
                    break;
                }

                generated += 1;

                // 1. Launch GPU forward pass asynchronously on CUDA stream for next token
                unsafe {
                    *batch_single.token = token;
                    *batch_single.pos = n_past;
                }

                let ret = unsafe { llama_decode(ctx, batch_single) };

                if ret != 0 {
                    let _ = tx.blocking_send(Err(anyhow!("llama_decode failed during token generation (code {})", ret)));
                    break;
                }

                n_past += 1;

                // 2. Dispatch current token piece to Tokio stream while GPU is computing
                if tx.blocking_send(Ok(piece)).is_err() {
                    break;
                }
            }

            unsafe {
                llama_batch_free(batch_single);
                llama_sampler_free(smpl_chain);
            };
        });

        (rx, cancelled)
    }
}
