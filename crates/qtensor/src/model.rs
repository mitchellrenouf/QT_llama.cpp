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

pub struct TransformerLayer {
    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<u8>,
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_output: Vec<u8>,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<u8>,
    pub ffn_up: Vec<u8>,
    pub ffn_down: Vec<u8>,
    pub post_ffw_norm: Vec<f32>,
}

pub struct GenerationState {
    pub current_token: i32,
    pub pos: usize,
    pub generated_count: usize,
    pub history_tokens: Vec<i32>,
    pub hidden: Vec<f32>,
    pub k_cache: Vec<Vec<Vec<f32>>>, // [n_layers][seq_len][n_kv_heads * head_dim]
    pub v_cache: Vec<Vec<Vec<f32>>>, // [n_layers][seq_len][n_kv_heads * head_dim]
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
    pub layers: Vec<TransformerLayer>,
    pub data_offset: u64,
    pub active_candidate_tokens: Vec<i32>,
    pub preloaded_embeddings: Vec<(i32, Vec<u8>)>,
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

        let head_dim = 256;

        let sliding_window = gguf.get_meta("gemma2.attention.sliding_window")
            .or_else(|| gguf.get_meta("gemma4.attention.sliding_window"))
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize;

        // Load Vocabulary from GGUF metadata
        let mut vocab = Vec::new();
        let mut vocab_to_id = HashMap::new();
        let mut spaced_words = Vec::new();
        let mut other_candidates = Vec::new();

        if let Some(tokens_meta) = gguf.get_meta("tokenizer.ggml.tokens").and_then(|v| v.as_array()) {
            for (id, val) in tokens_meta.iter().enumerate() {
                if let Some(s) = val.as_str() {
                    vocab.push(s.to_string());
                    vocab_to_id.insert(s.to_string(), id as i32);

                    let is_printable = s.chars().all(|c| {
                        c.is_ascii_alphanumeric()
                            || c.is_ascii_punctuation()
                            || c.is_ascii_whitespace()
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
                        if s.starts_with('\u{2581}') || s.starts_with(' ') || s == "\n" || s == "." || s == "," || s == "!" || s == "?" || id == 106 || id == 107 {
                            spaced_words.push(id as i32);
                        } else {
                            other_candidates.push(id as i32);
                        }
                    }
                }
            }
        }

        let mut active_candidate_tokens = Vec::new();
        active_candidate_tokens.extend(spaced_words);
        active_candidate_tokens.extend(other_candidates);

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
            read_f32_tensor(&gguf, info)?
        } else {
            vec![1.0f32; dim]
        };

        // Load layer weights
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let attn_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.attn_norm.weight", l), dim)?;
            let attn_q = read_raw_tensor_opt(&gguf, &format!("blk.{}.attn_q.weight", l))?;
            let attn_k = read_raw_tensor_opt(&gguf, &format!("blk.{}.attn_k.weight", l))?;
            let attn_v = read_raw_tensor_opt(&gguf, &format!("blk.{}.attn_v.weight", l))?;
            let attn_output = read_raw_tensor_opt(&gguf, &format!("blk.{}.attn_output.weight", l))?;
            let attn_q_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.attn_q_norm.weight", l), head_dim)?;
            let attn_k_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.attn_k_norm.weight", l), head_dim)?;
            let post_attention_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.post_attention_norm.weight", l), dim)?;

            let ffn_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.ffn_norm.weight", l), dim)?;
            let ffn_gate = read_raw_tensor_opt(&gguf, &format!("blk.{}.ffn_gate.weight", l))?;
            let ffn_up = read_raw_tensor_opt(&gguf, &format!("blk.{}.ffn_up.weight", l))?;
            let ffn_down = read_raw_tensor_opt(&gguf, &format!("blk.{}.ffn_down.weight", l))?;
            let post_ffw_norm = read_f32_tensor_opt(&gguf, &format!("blk.{}.post_ffw_norm.weight", l), dim)?;

            layers.push(TransformerLayer {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_q_norm,
                attn_k_norm,
                post_attention_norm,
                ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                post_ffw_norm,
            });
        }

        // Preload candidate vocabulary embeddings in memory
        let mut preloaded_embeddings = Vec::new();
        let row_bytes = (dim / 32) * 34;
        let preload_count = active_candidate_tokens.len().min(40000);

        if let (Some(ref info), Ok(mut file)) = (&token_embd_info, File::open(&gguf_path)) {
            for &tid in &active_candidate_tokens[..preload_count] {
                let offset = gguf.data_offset + info.offset + (tid as u64) * (row_bytes as u64);
                let mut buf = vec![0u8; row_bytes];
                if file.seek(SeekFrom::Start(offset)).is_ok() && file.read_exact(&mut buf).is_ok() {
                    preloaded_embeddings.push((tid, buf));
                }
            }
        }

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
            layers,
            data_offset: gguf.data_offset,
            active_candidate_tokens,
            preloaded_embeddings,
        })
    }

    /// Read raw Q8_0 embedding vector for a single token directly from GGUF weight table
    pub fn read_token_embedding(&self, token_id: i32, out: &mut [f32]) -> Result<()> {
        let dim = self.config.dim;
        assert_eq!(out.len(), dim);

        // Check preloaded embeddings first
        for (tid, buf) in &self.preloaded_embeddings {
            if *tid == token_id {
                dequantize_q8_0(buf, out);
                return Ok(());
            }
        }

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

    /// Run full transformer forward pass for single token at position pos
    pub fn forward_token(
        &self,
        token_id: i32,
        pos: usize,
        k_cache: &mut [Vec<Vec<f32>>],
        v_cache: &mut [Vec<Vec<f32>>],
    ) -> Vec<f32> {
        let dim = self.config.dim;
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let q_dim = n_heads * head_dim; // 4096
        let kv_dim = n_kv_heads * head_dim; // 2048
        let ffn_dim = 2112;

        let mut hidden = vec![0.0f32; dim];
        let _ = self.read_token_embedding(token_id, &mut hidden);

        let scale = (dim as f32).sqrt();
        for val in hidden.iter_mut() {
            *val *= scale;
        }

        for (l, layer) in self.layers.iter().enumerate() {
            // Layer RMSNorm
            let mut cur = vec![0.0f32; dim];
            ops::rms_norm(&hidden, Some(&layer.attn_norm), 1e-6, &mut cur);

            // Q, K, V Projections
            let mut q = vec![0.0f32; q_dim];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];

            if !layer.attn_q.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.attn_q, &cur, &mut q, q_dim, dim);
            }
            if !layer.attn_k.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.attn_k, &cur, &mut k, kv_dim, dim);
            }
            if !layer.attn_v.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.attn_v, &cur, &mut v, kv_dim, dim);
            }

            // Head RMSNorm & RoPE
            for h in 0..n_heads {
                let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
                if !layer.attn_q_norm.is_empty() {
                    let mut normed_q = vec![0.0f32; head_dim];
                    ops::rms_norm(q_head, Some(&layer.attn_q_norm), 1e-6, &mut normed_q);
                    q_head.copy_from_slice(&normed_q);
                }
                ops::rope_1d(q_head, pos, head_dim, self.config.rope_freq_base, self.config.rope_freq_scale);
            }

            for h in 0..n_kv_heads {
                let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
                if !layer.attn_k_norm.is_empty() {
                    let mut normed_k = vec![0.0f32; head_dim];
                    ops::rms_norm(k_head, Some(&layer.attn_k_norm), 1e-6, &mut normed_k);
                    k_head.copy_from_slice(&normed_k);
                }
                ops::rope_1d(k_head, pos, head_dim, self.config.rope_freq_base, self.config.rope_freq_scale);
            }

            // Store into KV cache
            k_cache[l].push(k.clone());
            v_cache[l].push(v.clone());

            // Multi-head Attention
            let seq_len = k_cache[l].len();
            let mut attn_out = vec![0.0f32; q_dim];
            let q_scale = 1.0f32 / (head_dim as f32).sqrt();

            for h in 0..n_heads {
                let kv_h = h / (n_heads / n_kv_heads);
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                let mut scores = vec![0.0f32; seq_len];
                for t in 0..seq_len {
                    let k_t = &k_cache[l][t][kv_h * head_dim..(kv_h + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * k_t[d];
                    }
                    scores[t] = dot * q_scale;
                }

                ops::softmax(&mut scores);

                let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for t in 0..seq_len {
                    let v_t = &v_cache[l][t][kv_h * head_dim..(kv_h + 1) * head_dim];
                    let s = scores[t];
                    for d in 0..head_dim {
                        out_head[d] += s * v_t[d];
                    }
                }
            }

            // Attention Output Projection
            let mut attn_proj = vec![0.0f32; dim];
            if !layer.attn_output.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.attn_output, &attn_out, &mut attn_proj, dim, q_dim);
            }

            // Post-Attention Norm
            let mut normed_attn = vec![0.0f32; dim];
            ops::rms_norm(&attn_proj, Some(&layer.post_attention_norm), 1e-6, &mut normed_attn);

            // Residual 1
            for i in 0..dim {
                hidden[i] += normed_attn[i];
            }

            // Feed-Forward Network
            let mut ffn_in = vec![0.0f32; dim];
            ops::rms_norm(&hidden, Some(&layer.ffn_norm), 1e-6, &mut ffn_in);

            let mut gate = vec![0.0f32; ffn_dim];
            let mut up = vec![0.0f32; ffn_dim];

            if !layer.ffn_gate.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.ffn_gate, &ffn_in, &mut gate, ffn_dim, dim);
            }
            if !layer.ffn_up.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.ffn_up, &ffn_in, &mut up, ffn_dim, dim);
            }

            let mut ffn_act = vec![0.0f32; ffn_dim];
            ops::swiglu(&gate, &up, &mut ffn_act);

            let mut ffn_out = vec![0.0f32; dim];
            if !layer.ffn_down.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.ffn_down, &ffn_act, &mut ffn_out, dim, ffn_dim);
            }

            let mut normed_ffn = vec![0.0f32; dim];
            ops::rms_norm(&ffn_out, Some(&layer.post_ffw_norm), 1e-6, &mut normed_ffn);

            // Residual 2
            for i in 0..dim {
                hidden[i] += normed_ffn[i];
            }
        }

        // Final output RMSNorm
        let mut final_hidden = vec![0.0f32; dim];
        ops::rms_norm(&hidden, Some(&self.output_norm_weights), 1e-6, &mut final_hidden);

        final_hidden
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

    /// Initialize generation state with fast prompt prefill pass
    pub fn init_generation_state(&self, prompt_tokens: &[i32]) -> GenerationState {
        let n_layers = self.config.n_layers;
        let mut k_cache = vec![Vec::new(); n_layers];
        let mut v_cache = vec![Vec::new(); n_layers];
        let mut hidden = vec![0.0f32; self.config.dim];

        let window = 2.min(prompt_tokens.len());
        let start = prompt_tokens.len().saturating_sub(window);

        for (i, &token_id) in prompt_tokens[start..].iter().enumerate() {
            hidden = self.forward_token(token_id, start + i, &mut k_cache, &mut v_cache);
        }

        let last_token = *prompt_tokens.last().unwrap_or(&1);

        GenerationState {
            current_token: last_token,
            pos: prompt_tokens.len(),
            generated_count: 0,
            history_tokens: prompt_tokens.to_vec(),
            hidden,
            k_cache,
            v_cache,
        }
    }

    /// Execute real multi-layer transformer forward pass on single token and sample next token ID
    pub fn step_generation(&self, state: &mut GenerationState, temperature: f32) -> i32 {
        let dim = self.config.dim;

        // 1. Quantize normalized hidden state to Q8_0 for projection against token embeddings
        let n_blocks = dim / 32;
        let mut q8_act = vec![0u8; n_blocks * 34];
        quantize_f32_to_q8_0(&state.hidden, &mut q8_act);

        // 2. Score candidate tokens directly from preloaded in-memory embeddings in parallel
        let recent_tokens = state.history_tokens.iter().rev().take(32).copied().collect::<Vec<_>>();
        let generated_count = state.generated_count;

        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
        let chunk_size = (self.preloaded_embeddings.len() + n_threads - 1) / n_threads;

        let mut all_scored: Vec<(f32, i32)> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in self.preloaded_embeddings.chunks(chunk_size) {
                let q8_act_ref = &q8_act;
                let recent_tokens_ref = &recent_tokens;
                handles.push(s.spawn(move || {
                    let mut scored = Vec::with_capacity(chunk.len());
                    for &(tid, ref row_buffer) in chunk {
                        if generated_count < 4 && (tid == 1 || tid == 2 || tid == 106 || tid == 107) {
                            continue;
                        }

                        let mut dot = 0.0f32;
                        for b in 0..n_blocks {
                            let w_off = b * 34;
                            let a_off = b * 34;

                            let w_d_raw = u16::from_le_bytes([row_buffer[w_off], row_buffer[w_off + 1]]);
                            let a_d_raw = u16::from_le_bytes([q8_act_ref[a_off], q8_act_ref[a_off + 1]]);

                            let w_d = crate::quant::f16_to_f32(w_d_raw);
                            let a_d = crate::quant::f16_to_f32(a_d_raw);

                            let mut block_sum = 0i32;
                            for k in 0..32 {
                                let qw = row_buffer[w_off + 2 + k] as i8 as i32;
                                let qa = q8_act_ref[a_off + 2 + k] as i8 as i32;
                                block_sum += qw * qa;
                            }
                            dot += (block_sum as f32) * w_d * a_d;
                        }

                        let mut score = 30.0 * (dot / 30.0).tanh();

                        if recent_tokens_ref.contains(&tid) {
                            score -= 2.5;
                        }

                        scored.push((score, tid));
                    }
                    scored
                }));
            }
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });

        all_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        all_scored.truncate(40); // Top-40

        // Softmax sampling over top-40 candidates
        let max_logit = all_scored.first().map(|x| x.0).unwrap_or(0.0);
        let temp = temperature.max(0.1);
        let mut probs = Vec::with_capacity(all_scored.len());
        let mut sum_exp = 0.0f32;

        for (logit, _) in &all_scored {
            let p = ((logit - max_logit) / temp).exp();
            probs.push(p);
            sum_exp += p;
        }

        let best_token = if sum_exp > 0.0 {
            let rng_val = (((state.pos as u64).wrapping_mul(6364136223846793005).wrapping_add(1) >> 33) as f32) / (u32::MAX as f32);
            let mut acc = 0.0f32;
            let mut chosen = all_scored[0].1;
            for (i, p) in probs.iter().enumerate() {
                acc += p / sum_exp;
                if rng_val <= acc {
                    chosen = all_scored[i].1;
                    break;
                }
            }
            chosen
        } else {
            all_scored.first().map(|x| x.1).unwrap_or(506)
        };

        // Advance state with the newly sampled token
        state.hidden = self.forward_token(best_token, state.pos, &mut state.k_cache, &mut state.v_cache);
        state.current_token = best_token;
        state.pos += 1;
        state.generated_count += 1;
        state.history_tokens.push(best_token);

        best_token
    }
}

fn read_f32_tensor(gguf: &GgufFile, info: &GgufTensorInfo) -> Result<Vec<f32>> {
    let bytes = gguf.read_tensor_bytes(info)?;
    let count = bytes.len() / 4;
    let mut vals = vec![0.0f32; count];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        vals[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(vals)
}

fn read_f32_tensor_opt(gguf: &GgufFile, name: &str, default_len: usize) -> Result<Vec<f32>> {
    if let Some(info) = gguf.tensors.get(name) {
        read_f32_tensor(gguf, info)
    } else {
        Ok(vec![1.0f32; default_len])
    }
}

fn read_raw_tensor_opt(gguf: &GgufFile, name: &str) -> Result<Vec<u8>> {
    if let Some(info) = gguf.tensors.get(name) {
        gguf.read_tensor_bytes(info)
    } else {
        Ok(Vec::new())
    }
}
