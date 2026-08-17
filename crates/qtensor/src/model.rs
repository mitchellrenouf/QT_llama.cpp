use crate::device::{DeviceManager, DeviceType};
use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheManager;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub rms_norm_eps: f32,
    pub max_context: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 2304,
            n_layers: 30,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            vocab_size: 256000,
            sliding_window: 4096,
            rope_freq_base: 500000.0,
            rope_freq_scale: 1.0,
            rms_norm_eps: 1e-6,
            max_context: 256000,
        }
    }
}

pub struct QTensorModel {
    pub config: ModelConfig,
    pub device_manager: DeviceManager,
    pub kv_cache: KvCacheManager,
    pub layer_devices: Vec<DeviceType>,
    pub vocab: Vec<String>,
    pub vocab_to_id: HashMap<String, i32>,
}

impl QTensorModel {
    pub fn load_from_gguf<P: AsRef<Path>>(path: P, max_context: usize) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        
        let dim = gguf.get_meta("gemma2.embedding_length")
            .or_else(|| gguf.get_meta("gemma4.embedding_length"))
            .or_else(|| gguf.get_meta("general.embedding_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(2304) as usize;

        let n_layers = gguf.get_meta("gemma2.block_count")
            .or_else(|| gguf.get_meta("gemma4.block_count"))
            .or_else(|| gguf.get_meta("general.block_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(30) as usize;

        let n_heads = gguf.get_meta("gemma2.attention.head_count")
            .or_else(|| gguf.get_meta("gemma4.attention.head_count"))
            .or_else(|| gguf.get_meta("general.attention.head_count"))
            .and_then(|v| v.as_u32())
            .unwrap_or(16) as usize;

        let n_kv_heads = gguf.get_meta("gemma2.attention.head_count_kv")
            .or_else(|| gguf.get_meta("gemma4.attention.head_count_kv"))
            .or_else(|| gguf.get_meta("general.attention.head_count_kv"))
            .and_then(|v| v.as_u32())
            .unwrap_or(8) as usize;

        let head_dim = gguf.get_meta("gemma2.attention.key_length")
            .or_else(|| gguf.get_meta("gemma4.attention.key_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(256) as usize;

        let sliding_window = gguf.get_meta("gemma2.attention.sliding_window")
            .or_else(|| gguf.get_meta("gemma4.attention.sliding_window"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        // Load Vocabulary from GGUF metadata
        let mut vocab = Vec::new();
        let mut vocab_to_id = HashMap::new();
        if let Some(tokens_meta) = gguf.get_meta("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            for (id, val) in tokens_meta.iter().enumerate() {
                if let Some(s) = val.as_str() {
                    vocab.push(s.to_string());
                    vocab_to_id.insert(s.to_string(), id as i32);
                }
            }
        }

        let vocab_size = if !vocab.is_empty() { vocab.len() } else { 256000 };

        let config = ModelConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            sliding_window,
            rope_freq_base: 500000.0,
            rope_freq_scale: 1.0,
            rms_norm_eps: 1e-6,
            max_context: max_context.max(8192),
        };

        let device_manager = DeviceManager::new();
        let layer_bytes = 450_000_000;
        let layer_devices = device_manager.plan_layers(n_layers, layer_bytes);

        // Every alternate layer in Gemma 4 is sliding window
        let sw_layers: Vec<usize> = (0..n_layers).filter(|i| i % 2 == 0).collect();

        let kv_cache = KvCacheManager::new(
            n_layers,
            n_kv_heads,
            head_dim,
            config.max_context,
            &sw_layers,
            sliding_window,
            &layer_devices,
        )?;

        eprintln!(
            "[qtensor] Initialized model: {} layers, {} dim, {} max_context, {} vocab items ({} GPU layers, {} CPU layers)",
            n_layers,
            dim,
            config.max_context,
            vocab.len(),
            layer_devices.iter().filter(|d| matches!(d, DeviceType::Cuda(_))).count(),
            layer_devices.iter().filter(|d| matches!(d, DeviceType::Cpu)).count(),
        );

        Ok(Self {
            config,
            device_manager,
            kv_cache,
            layer_devices,
            vocab,
            vocab_to_id,
        })
    }

    /// Tokenize text using longest-matching BPE over the loaded GGUF vocabulary
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        if self.vocab.is_empty() {
            return text.as_bytes().iter().map(|&b| b as i32 + 100).collect();
        }

        let mut tokens = Vec::new();
        // Standard Gemma special token prefix
        if let Some(&bos) = self.vocab_to_id.get("<bos>").or_else(|| self.vocab_to_id.get("<s>")) {
            tokens.push(bos);
        }

        let formatted = text.replace(' ', " ");
        let mut char_indices: Vec<usize> = formatted.char_indices().map(|(i, _)| i).collect();
        char_indices.push(formatted.len());

        let mut i = 0;
        while i < char_indices.len() - 1 {
            let mut matched = false;
            let max_lookahead = (i + 32).min(char_indices.len() - 1);

            for j in (i + 1..=max_lookahead).rev() {
                let sub = &formatted[char_indices[i]..char_indices[j]];
                if let Some(&tid) = self.vocab_to_id.get(sub) {
                    tokens.push(tid);
                    i = j;
                    matched = true;
                    break;
                }
            }

            if !matched {
                let single_char = &formatted[char_indices[i]..char_indices[i + 1]];
                if let Some(&tid) = self.vocab_to_id.get(single_char) {
                    tokens.push(tid);
                } else {
                    for &b in single_char.as_bytes() {
                        let hex_repr = format!("<0x{:02X}>", b);
                        if let Some(&tid) = self.vocab_to_id.get(&hex_repr) {
                            tokens.push(tid);
                        }
                    }
                }
                i += 1;
            }
        }

        tokens
    }

    /// Detokenize token ID to text string
    pub fn token_to_piece(&self, token_id: i32) -> String {
        if token_id < 0 {
            return String::new();
        }

        if let Some(piece) = self.vocab.get(token_id as usize) {
            if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                    return String::from_utf8_lossy(&[byte_val]).to_string();
                }
            }
            return piece.replace(' ', " ");
        }

        if token_id >= 100 && token_id <= 355 {
            let byte = (token_id - 100) as u8;
            return String::from_utf8_lossy(&[byte]).to_string();
        }

        format!("_{}", token_id)
    }
}
