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

pub struct GenerationChunk {
    pub text: String,
    pub token_count: usize,
}

impl LlamaEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        _n_gpu_layers: i32,
        ctx_size: u32,
        _cache_type_k: &str,
        _cache_type_v: &str,
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
    ) -> (mpsc::Receiver<Result<GenerationChunk>>, Arc<AtomicBool>) {
        let (mut source, cancel) = self.inner.generate_stream(prompt, max_tokens, temperature);
        let (tx, rx) = mpsc::channel(4096);
        tokio::spawn(async move {
            while let Some(piece) = source.recv().await {
                let piece = piece.map(|text| GenerationChunk {
                    text,
                    token_count: 1,
                });
                if tx.send(piece).await.is_err() {
                    break;
                }
            }
        });
        (rx, cancel)
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
