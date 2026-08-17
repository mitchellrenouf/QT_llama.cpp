use crate::model::QTensorModel;
use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

pub struct QTensorEngine {
    pub model: Arc<Mutex<QTensorModel>>,
}

impl QTensorEngine {
    pub fn new<P: AsRef<Path>>(model_path: P, max_context: usize) -> Result<Self> {
        let model = QTensorModel::load_from_gguf(model_path, max_context)?;
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

    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(4096);
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

                if is_eog {
                    break;
                }

                let trimmed = piece.trim();
                if trimmed == "<end_of_turn>"
                    || trimmed == "<turn|>"
                    || trimmed == "<|turn_end|>"
                    || trimmed == "<|im_end|>"
                    || trimmed == "</s>"
                    || trimmed == "<eos>"
                {
                    break;
                }

                generated += 1;

                if tx.blocking_send(Ok(piece)).is_err() {
                    break;
                }
            }
        });

        (rx, cancelled)
    }
}
