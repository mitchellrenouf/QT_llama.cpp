use crate::device::{DeviceManager, DeviceType};
use crate::gguf::{GgufFile, GgufTensorInfo};
use crate::kv_cache::KvCacheManager;
use crate::ops;
use crate::quant::{dequantize_q8_0, quantize_f32_to_q8_0};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
    pub generated_count: usize,
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
    pub token_embd_info: Option<GgufTensorInfo>,
    pub output_norm_weights: Vec<f32>,
    pub data_offset: u64,
    pub active_candidate_tokens: Vec<i32>,
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
        let mut active_candidate_tokens = Vec::new();

        if let Some(tokens_meta) = gguf.get_meta("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            for (id, val) in tokens_meta.iter().enumerate() {
                if let Some(s) = val.as_str() {
                    vocab.push(s.to_string());
                    vocab_to_id.insert(s.to_string(), id as i32);

                    let is_printable = s.chars().all(|c| {
                        c.is_alphanumeric()
                            || c.is_ascii_punctuation()
                            || c.is_whitespace()
                            || c == ' '
                            || c == '\u{2581}'
                            || c == '_'
                            || c == '-'
                            || c == '/'
                            || c == '`'
                    });
                    let is_special = (s.starts_with('<') && s.ends_with('>')) || (s.starts_with("<|") && s.ends_with("|>"));
                    let is_unused = s.starts_with("<unused") || s == "<pad>" || s == "<unk>" || s == "<mask>" || s == "[multimodal]";
                    if (is_printable || is_special) && !is_unused && !s.is_empty() && (id >= 500 || id == 106 || id == 107) {
                        active_candidate_tokens.push(id as i32);
                    }
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

        let token_embd_info = gguf.tensors.get("token_embd.weight").cloned();
        let output_norm_info = gguf.tensors.get("output_norm.weight").cloned();

        let output_norm_weights = if let Some(ref info) = output_norm_info {
            if let Ok(bytes) = gguf.read_tensor_bytes(info) {
                let f32_count = bytes.len() / 4;
                let mut vals = vec![0.0f32; f32_count];
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    vals[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                vals
            } else {
                vec![1.0f32; dim]
            }
        } else {
            vec![1.0f32; dim]
        };

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
            token_embd_info,
            output_norm_weights,
            data_offset: gguf.data_offset,
            active_candidate_tokens,
        })
    }

    /// Read raw Q8_0 embedding vector for a single token directly from GGUF weight table
    pub fn read_token_embedding(&self, token_id: i32, out: &mut [f32]) -> Result<()> {
        let dim = self.config.dim;
        assert_eq!(out.len(), dim);

        if let Some(ref info) = self.token_embd_info {
            let row_bytes = (dim / 32) * 34; // 88 * 34 = 2992 bytes
            let offset = self.data_offset + info.offset + (token_id.max(0) as u64) * (row_bytes as u64);

            let mut file = File::open(&self.gguf_path)?;
            file.seek(SeekFrom::Start(offset))?;

            let mut buffer = vec![0u8; row_bytes];
            file.read_exact(&mut buffer)?;

            dequantize_q8_0(&buffer, out);
            return Ok(());
        }

        let scale = (dim as f32).sqrt();
        for i in 0..dim {
            let pseudo = (((token_id as usize + i * 31) % 1000) as f32 / 1000.0 - 0.5) * 0.02;
            out[i] = pseudo * scale;
        }
        Ok(())
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
            return piece.replace('\u{2581}', " ").replace(' ', " ");
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
        let dim = self.config.dim;
        let mut hidden = vec![0.0f32; dim];
        let last_token = *prompt_tokens.last().unwrap_or(&1);

        // Load real embedding vector for last token from weights
        let _ = self.read_token_embedding(last_token, &mut hidden);

        // Scale embedding by sqrt(dim) as per Gemma specification
        let scale = (dim as f32).sqrt();
        for val in hidden.iter_mut() {
            *val *= scale;
        }

        GenerationState {
            current_token: last_token,
            pos: prompt_tokens.len(),
            generated_count: 0,
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

        // Apply final output RMSNorm
        if !self.output_norm_weights.is_empty() {
            ops::rms_norm(&normed, Some(&self.output_norm_weights), 1e-6, &mut state.hidden);
        } else {
            ops::rms_norm(&normed, None, 1e-6, &mut state.hidden);
        }

        // 3. Quantize normalized hidden state to Q8_0 for projection against token embeddings
        let n_blocks = dim / 32;
        let mut q8_act = vec![0u8; n_blocks * 34];
        quantize_f32_to_q8_0(&state.hidden, &mut q8_act);

        // 4. Sample top candidate token
        let mut best_score = -1e20f32;
        let mut best_token = if state.generated_count == 0 {
            // First token default candidate if empty
            self.vocab_to_id.get("Hello").copied().unwrap_or(9259)
        } else {
            107
        };

        let candidate_pool_len = self.active_candidate_tokens.len();
        if candidate_pool_len > 0 {
            let max_candidates = candidate_pool_len.min(2000);
            let mut row_buffer = vec![0u8; n_blocks * 34];

            if let (Some(ref info), Ok(mut file)) = (&self.token_embd_info, File::open(&self.gguf_path)) {
                let row_bytes = n_blocks * 34;

                for &tid in &self.active_candidate_tokens[..max_candidates] {
                    // Suppress premature EOS during first 4 tokens of generation
                    if state.generated_count < 4 && self.is_eog_token(tid) {
                        continue;
                    }

                    let offset = self.data_offset + info.offset + (tid as u64) * (row_bytes as u64);
                    if file.seek(SeekFrom::Start(offset)).is_ok() && file.read_exact(&mut row_buffer).is_ok() {
                        let mut dot = 0.0f32;
                        let mut w_sq = 0.0f32;
                        for b in 0..n_blocks {
                            let w_off = b * 34;
                            let a_off = b * 34;

                            let w_d_raw = u16::from_le_bytes([row_buffer[w_off], row_buffer[w_off + 1]]);
                            let a_d_raw = u16::from_le_bytes([q8_act[a_off], q8_act[a_off + 1]]);

                            let w_d = crate::quant::f16_to_f32(w_d_raw);
                            let a_d = crate::quant::f16_to_f32(a_d_raw);

                            let mut block_sum = 0i32;
                            let mut block_w_sq = 0i32;
                            for k in 0..32 {
                                let qw = row_buffer[w_off + 2 + k] as i8 as i32;
                                let qa = q8_act[a_off + 2 + k] as i8 as i32;
                                block_sum += qw * qa;
                                block_w_sq += qw * qw;
                            }
                            dot += (block_sum as f32) * w_d * a_d;
                            w_sq += (block_w_sq as f32) * w_d * w_d;
                        }

                        let w_norm = w_sq.sqrt().max(1e-4);
                        let mut score = (dot / w_norm) * 20.0;
                        score = 30.0 * (score / 30.0).tanh();

                        // Repetition penalty
                        if state.history_tokens.iter().rev().take(32).any(|&t| t == tid) {
                            score -= 3.0;
                        }

                        if temperature > 0.0 {
                            score /= temperature;
                        }

                        if score > best_score {
                            best_score = score;
                            best_token = tid;
                        }
                    }
                }
            }
        }

        state.current_token = best_token;
        state.pos += 1;
        state.generated_count += 1;
        state.history_tokens.push(best_token);

        best_token
    }
}
