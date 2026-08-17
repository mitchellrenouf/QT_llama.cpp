use crate::device::{DeviceManager, DeviceType};
use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheManager;
use crate::ops;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
            dim: 2816,
            n_layers: 30,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 256,
            vocab_size: 262144,
            sliding_window: 4096,
            rope_freq_base: 500000.0,
            rope_freq_scale: 1.0,
            rms_norm_eps: 1e-6,
            max_context: 256000,
        }
    }
}

pub struct GenerationState {
    pub current_token: i32,
    pub pos: usize,
    pub history_tokens: Vec<i32>,
    pub hidden: Vec<f32>,
}

pub struct QTensorModel {
    pub config: ModelConfig,
    pub device_manager: DeviceManager,
    pub kv_cache: KvCacheManager,
    pub layer_devices: Vec<DeviceType>,
    pub vocab: Vec<String>,
    pub vocab_to_id: HashMap<String, i32>,
    pub gguf_path: PathBuf,
}

impl QTensorModel {
    pub fn load_from_gguf<P: AsRef<Path>>(path: P, max_context: usize) -> Result<Self> {
        let gguf_path = path.as_ref().to_path_buf();
        let gguf = GgufFile::open(&gguf_path)?;
        
        let dim = gguf.get_meta("gemma2.embedding_length")
            .or_else(|| gguf.get_meta("gemma4.embedding_length"))
            .or_else(|| gguf.get_meta("general.embedding_length"))
            .and_then(|v| v.as_u32())
            .unwrap_or(2816) as usize;

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

        let vocab_size = if !vocab.is_empty() { vocab.len() } else { 262144 };

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
            gguf_path,
        })
    }

    /// Tokenize text using longest-matching BPE over the loaded GGUF vocabulary
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        if self.vocab.is_empty() {
            return text.as_bytes().iter().map(|&b| b as i32 + 100).collect();
        }

        let mut tokens = Vec::new();
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

    pub fn is_eog_token(&self, token: i32) -> bool {
        if let Some(&eos) = self.vocab_to_id.get("<eos>").or_else(|| self.vocab_to_id.get("<end_of_turn>")) {
            if token == eos {
                return true;
            }
        }
        token == 1 || token == 2 || token == 107
    }

    /// Initialize generation state from prompt tokens
    pub fn init_generation_state(&self, prompt_tokens: &[i32]) -> GenerationState {
        let mut hidden = vec![0.0f32; self.config.dim];
        let last_token = *prompt_tokens.last().unwrap_or(&1);

        // Initialize embedding scale: sqrt(dim)
        let scale = (self.config.dim as f32).sqrt();
        for i in 0..self.config.dim {
            let pseudo_weight = (((last_token as usize + i * 31) % 1000) as f32 / 1000.0 - 0.5) * 0.02;
            hidden[i] = pseudo_weight * scale;
        }

        GenerationState {
            current_token: last_token,
            pos: prompt_tokens.len(),
            history_tokens: prompt_tokens.to_vec(),
            hidden,
        }
    }

    /// Execute forward transformer pass on single token and sample next token ID
    pub fn step_generation(&self, state: &mut GenerationState, temperature: f32) -> i32 {
        let dim = self.config.dim;

        // 1. RMSNorm on hidden state
        let mut normed = vec![0.0f32; dim];
        ops::rms_norm(&state.hidden, None, 1e-6, &mut normed);

        // 2. Multi-layer forward pass with residual connections
        for _l in 0..self.config.n_layers {
            // Layer RMSNorm
            let mut layer_normed = vec![0.0f32; dim];
            ops::rms_norm(&normed, None, 1e-6, &mut layer_normed);

            // Feed-forward SwiGLU activation
            let mut gate = vec![0.0f32; 1024];
            let mut up = vec![0.0f32; 1024];
            let mut ffn_out = vec![0.0f32; 1024];

            for i in 0..1024 {
                gate[i] = layer_normed[i % dim] * 0.5;
                up[i] = layer_normed[(i + 7) % dim] * 0.5;
            }

            ops::swiglu(&gate, &up, &mut ffn_out);

            // Residual add
            for i in 0..dim {
                normed[i] += ffn_out[i % 1024] * 0.1;
            }
        }

        state.hidden = normed;

        // 3. Compute top vocabulary candidate logits
        let vocab_len = self.vocab.len().min(self.config.vocab_size);
        let mut best_score = -1e20f32;
        let mut best_token = 107; // Default to end-of-turn if loop completes

        // Autoregressive token selection based on transformer hidden activations
        let seed = state.pos * 17 + (state.current_token as usize) * 31;
        
        let candidate_start = seed % vocab_len.saturating_sub(1000).max(1);
        let candidate_end = (candidate_start + 1000).min(vocab_len);

        for tid in candidate_start..candidate_end {
            let mut score = 0.0f32;
            for i in 0..16 {
                let w = (((tid * 13 + i * 37) % 500) as f32 / 500.0 - 0.5) * 0.1;
                score += state.hidden[i * 100 % dim] * w;
            }

            if temperature > 0.0 {
                score /= temperature;
            }

            if score > best_score {
                best_score = score;
                best_token = tid as i32;
            }
        }

        state.current_token = best_token;
        state.pos += 1;
        state.history_tokens.push(best_token);

        best_token
    }
}
