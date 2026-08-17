use anyhow::Result;
use qtensor::QTensorEngine;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct LlamaEngine {
    inner: Arc<QTensorEngine>,
}

impl LlamaEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        _n_gpu_layers: i32,
        ctx_size: u32,
        _backend: Option<&str>,
    ) -> Result<Self> {
        let inner = QTensorEngine::new(model_path, ctx_size as usize)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> (mpsc::Receiver<Result<String>>, Arc<AtomicBool>) {
        self.inner.generate_stream(prompt, max_tokens, temperature)
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        self.inner.tokenize(text, add_special)
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        self.inner.token_to_piece(token)
    }

    pub fn is_eog(&self, token: i32) -> bool {
        self.inner.is_eog(token)
    }
}
