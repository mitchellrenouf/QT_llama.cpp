use crate::error::{Error, Result};
use mrml_tensor::MrmlEngine;
use std::path::Path;
use std::sync::mpsc;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Clone)]
pub struct ModelEngine {
    inner: Arc<MrmlEngine>,
}

pub struct GenerationChunk {
    pub text: String,
    pub token_count: usize,
}

impl ModelEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        _n_gpu_layers: i32,
        ctx_size: u32,
        cache_type_k: &str,
        cache_type_v: &str,
        _backend: Option<&str>,
    ) -> Result<Self> {
        let automatic = if ctx_size >= 131_072 { "q4_0" } else { "f16" };
        let cache_type_k = if cache_type_k == "auto" {
            automatic
        } else {
            cache_type_k
        };
        let cache_type_v = if cache_type_v == "auto" {
            automatic
        } else {
            cache_type_v
        };
        let inner =
            MrmlEngine::new_with_cache(model_path, ctx_size as usize, cache_type_k, cache_type_v)?;
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
        let (tx, rx) = mpsc::sync_channel(4096);
        let cancel = self
            .inner
            .generate_stream(prompt, max_tokens, temperature, move |piece| {
                let chunk = piece.map_err(Error::from).map(|text| GenerationChunk {
                    text,
                    token_count: 1,
                });
                tx.send(chunk).is_ok()
            });
        (rx, cancel)
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        Ok(self.inner.tokenize(text, add_special)?)
    }

    pub fn token_to_piece(&self, token: i32) -> Result<String> {
        Ok(self.inner.token_to_piece(token)?)
    }

    pub fn is_eog(&self, token: i32) -> bool {
        self.inner.is_eog(token)
    }

    pub fn chat_template(&self) -> Option<String> {
        self.inner.chat_template()
    }


    pub fn gpu_layer_residency(&self) -> Option<(usize, usize)> {
        self.inner.gpu_layer_residency()
    }
}
