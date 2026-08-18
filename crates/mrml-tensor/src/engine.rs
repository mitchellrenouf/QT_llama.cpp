use crate::model::MrmlModel;
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct MrmlEngine {
    pub model: Arc<Mutex<MrmlModel>>,
}

impl MrmlEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, max_context: usize) -> Result<Self> {
        let model = MrmlModel::load_from_gguf(model_path, max_context)?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }

    pub fn tokenize(&self, text: &str, _add_special: bool) -> Result<Vec<i32>> {
        let guard = self.model.lock().unwrap();
        Ok(guard.tokenize(text))
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        let guard = self.model.lock().unwrap();
        Ok(guard.token_to_piece(token))
    }

    pub fn is_eog(&self, token: i32) -> bool {
        let guard = self.model.lock().unwrap();
        guard.is_eog_token(token)
    }

    pub fn new_with_cache<P: AsRef<Path>>(
        model_path: P,
        max_context: usize,
        cache_type_k: &str,
        cache_type_v: &str,
    ) -> Result<Self> {
        let cache_type_k = crate::kv_cache::KvCacheFormat::parse(cache_type_k)?;
        let cache_type_v = crate::kv_cache::KvCacheFormat::parse(cache_type_v)?;
        let model = MrmlModel::load_from_gguf_with_cache(
            model_path, max_context, cache_type_k, cache_type_v,
        )?;
        Ok(Self { model: Arc::new(Mutex::new(model)) })
    }

    pub fn chat_template(&self) -> Option<String> {
        let guard = self.model.lock().unwrap();
        guard.chat_template.clone()
    }

    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        mut emit: F,
    ) -> Arc<AtomicBool>
    where
        F: FnMut(Result<String>) -> bool + Send + 'static,
    {
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancelled.clone();

        let model_arc = self.model.clone();
        let prompt_string = prompt.to_string();

        std::thread::spawn(move || {
            let prompt_tokens = {
                let guard = model_arc.lock().unwrap();
                guard.tokenize(&prompt_string)
            };

            if prompt_tokens.is_empty() {
                return;
            }

            let mut state = {
                let guard = model_arc.lock().unwrap();
                guard.init_generation_state(&prompt_tokens)
            };

            let mut generated = 0;
            let start_time = Instant::now();

            while generated < max_tokens {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }

                let next_token = {
                    let guard = model_arc.lock().unwrap();
                    guard.step_generation(&mut state, temperature)
                };

                let (piece, is_eog) = {
                    let guard = model_arc.lock().unwrap();
                    let eog = guard.is_eog_token(next_token);
                    let piece_str = guard.token_to_piece(next_token);
                    (piece_str, eog)
                };

                let trimmed = piece.trim();
                if is_eog
                    || next_token == 106
                    || next_token == 1
                    || trimmed == "<end_of_turn>"
                    || trimmed == "<turn|>"
                    || trimmed == "<|turn_end|>"
                    || trimmed == "<|im_end|>"
                    || trimmed == "</s>"
                    || trimmed == "<eos>"
                {
                    break;
                }

                generated += 1;

                if !emit(Ok(piece)) {
                    break;
                }
            }

            let elapsed = start_time.elapsed().as_secs_f64();
            let tps = if elapsed > 0.0001 {
                generated as f64 / elapsed
            } else {
                0.0
            };
            eprintln!("[mrml] Generated {} tokens in {:.2}s ({:.1} tk/s)", generated, elapsed, tps);
        });

        cancelled
    }
}
