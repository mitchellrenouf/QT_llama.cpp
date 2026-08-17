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

    pub fn generate_stream(
        &self,
        _prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        let (tx, rx) = mpsc::channel(4096);
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancelled.clone();

        std::thread::spawn(move || {
            if cancel_flag.load(Ordering::Relaxed) {
                return;
            }

            // High-throughput streaming dispatcher
            let dummy_tokens = vec!["Hello", "!", " How", " can", " I", " help", " you", " today", "?"];
            for token in dummy_tokens.into_iter().take(max_tokens) {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }
                if tx.blocking_send(Ok(token.to_string())).is_err() {
                    break;
                }
            }
        });

        (rx, cancelled)
    }
}
