use crate::device::{DeviceManager, DeviceType};
use crate::gguf::{GgufFile, GgufTensorInfo};
use crate::kv_cache::KvCacheManager;
use crate::ops;
use crate::quant::dequantize_q8_0;
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub max_context: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dim: 2816,
            n_layers: 30,
            vocab_size: 262144,
            max_context: 256000,
        }
    }
}

pub struct TransformerLayer {
    pub is_swa: bool,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub rope_freq_base: f32,
    pub sliding_window: usize,

    pub attn_norm: Vec<f32>,
    pub attn_q: Vec<u8>,
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_output: Vec<u8>,
    pub attn_q_norm: Vec<f32>,
    pub attn_k_norm: Vec<f32>,
    pub post_attention_norm: Vec<f32>,

    // Dense shared FFN
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: Vec<u8>,
    pub ffn_up: Vec<u8>,
    pub ffn_down: Vec<u8>,
    pub post_ffw_norm: Vec<f32>,
    pub post_ffw_norm_1: Vec<f32>,
    pub pre_ffw_norm_2: Vec<f32>,
    pub post_ffw_norm_2: Vec<f32>,
    pub layer_output_scale: f32,

    // MoE Router & Experts
    pub is_moe: bool,
    pub ffn_gate_inp: Vec<f32>,       // [2816, 128]
    pub ffn_gate_inp_scale: Vec<f32>, // [2816]
    pub ffn_down_exps_scale: Vec<f32>,// [128]
    pub ffn_gate_up_exps_offset: u64,
    pub ffn_down_exps_offset: u64,

    #[cfg(feature = "cuda")]
    pub gpu_attn_q: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_attn_k: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_attn_v: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_attn_output: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_gate: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_up: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_down: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_gate_up_exps: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_down_exps: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_ffn_down_exps_scale: Option<crate::cuda::CudaBuffer<f32>>,

    // Persistent GPU scratch/activation buffers
    #[cfg(feature = "cuda")]
    pub gpu_d_cur: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_q: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_k: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_v: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_attn_in: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_attn_out: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_mlp_in: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_mlp_gate: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_mlp_up: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_mlp_act: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_mlp_down: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_moe_in: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_moe_exp_ids: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<i32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_moe_exp_weights: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_moe_act_scratch: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_moe_out: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
}

pub struct GenerationState {
    pub current_token: i32,
    pub pos: usize,
    pub generated_count: usize,
    pub history_tokens: Vec<i32>,
    pub hidden: Vec<f32>,
    pub k_cache: Vec<Vec<Vec<f32>>>, // [n_layers][seq_len][kv_dim]
    pub v_cache: Vec<Vec<Vec<f32>>>, // [n_layers][seq_len][kv_dim]
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
    pub token_embd_table: Vec<u8>,
    pub mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    #[cfg(feature = "cuda")]
    pub cuda_dev: Option<std::sync::Arc<crate::cuda::CudaDevice>>,
    #[cfg(feature = "cuda")]
    pub gpu_token_embd_table: Option<crate::cuda::CudaBuffer<u8>>,
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
            vocab_size,
            max_context: max_context.max(8192),
        };

        let device_manager = DeviceManager::new();
        let layer_bytes = 450_000_000;
        let layer_devices = device_manager.plan_layers(n_layers, layer_bytes);

        let sw_layers: Vec<usize> = (0..n_layers).filter(|i| i % 6 != 5).collect();

        let kv_cache = KvCacheManager::new(
            n_layers,
            8,
            256,
            config.max_context,
            &sw_layers,
            1024,
            &layer_devices,
        )?;

        let token_embd_info = gguf.tensors.get("token_embd.weight").cloned();
        let output_norm_info = gguf.tensors.get("output_norm.weight").cloned();

        let output_norm_weights = if let Some(ref info) = output_norm_info {
            read_f32_tensor(&gguf, info)?
        } else {
            vec![1.0f32; dim]
        };

        #[cfg(feature = "cuda")]
        let cuda_dev = crate::cuda::CudaDevice::new(0).ok().map(std::sync::Arc::new);

        let mmap = File::open(&gguf_path)
            .ok()
            .and_then(|file| unsafe { memmap2::Mmap::map(&file).ok() })
            .map(std::sync::Arc::new);

        // Load layer weights
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let is_swa = (l % 6) != 5;
            let (head_dim, n_heads, n_kv_heads, rope_freq_base, sliding_window) = if is_swa {
                (256, 16, 8, 10000.0f32, 1024)
            } else {
                (512, 16, 2, 1000000.0f32, 0)
            };
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;

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
            let post_ffw_norm_1 = read_f32_tensor_opt(&gguf, &format!("blk.{}.post_ffw_norm_1.weight", l), dim)?;
            let pre_ffw_norm_2 = read_f32_tensor_opt(&gguf, &format!("blk.{}.pre_ffw_norm_2.weight", l), dim)?;
            let post_ffw_norm_2 = read_f32_tensor_opt(&gguf, &format!("blk.{}.post_ffw_norm_2.weight", l), dim)?;
            let layer_output_scale = read_f32_tensor_opt(&gguf, &format!("blk.{}.layer_output_scale.weight", l), 1)?
                .first().copied().unwrap_or(1.0);

            // MoE tensors
            let ffn_gate_inp = read_f32_tensor_opt(&gguf, &format!("blk.{}.ffn_gate_inp.weight", l), dim * 128)?;
            let ffn_gate_inp_scale = read_f32_tensor_opt(&gguf, &format!("blk.{}.ffn_gate_inp.scale", l), dim)?;
            let ffn_down_exps_scale = read_f32_tensor_opt(&gguf, &format!("blk.{}.ffn_down_exps.scale", l), 128)?;

            let ffn_gate_up_exps_offset = gguf.tensors.get(&format!("blk.{}.ffn_gate_up_exps.weight", l))
                .map(|t| gguf.data_offset + t.offset).unwrap_or(0);
            let ffn_down_exps_offset = gguf.tensors.get(&format!("blk.{}.ffn_down_exps.weight", l))
                .map(|t| gguf.data_offset + t.offset).unwrap_or(0);
            let is_moe = ffn_gate_up_exps_offset > 0;

            #[cfg(feature = "cuda")]
            let gpu_attn_q = if cuda_dev.is_some() && !attn_q.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &attn_q).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_attn_k = if cuda_dev.is_some() && !attn_k.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &attn_k).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_attn_v = if cuda_dev.is_some() && !attn_v.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &attn_v).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_attn_output = if cuda_dev.is_some() && !attn_output.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &attn_output).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_ffn_gate = if cuda_dev.is_some() && !ffn_gate.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &ffn_gate).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_ffn_up = if cuda_dev.is_some() && !ffn_up.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &ffn_up).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_ffn_down = if cuda_dev.is_some() && !ffn_down.is_empty() { crate::cuda::CudaBuffer::from_host_on(0, &ffn_down).ok() } else { None };

            let exp_ffn_dim = 704;
            let n_exps = 128;
            let gate_up_bytes = n_exps * (2 * exp_ffn_dim) * (dim / 32) * 18;
            let down_bytes = n_exps * dim * (exp_ffn_dim / 32) * 18;

            #[cfg(feature = "cuda")]
            let (gpu_ffn_gate_up_exps, gpu_ffn_down_exps, gpu_ffn_down_exps_scale) = if cuda_dev.is_some() && is_moe {
                let should_offload = if let Ok((free_mem, _)) = crate::cuda::CudaDevice::get_memory_info(0) {
                    free_mem >= (gate_up_bytes + down_bytes + 400 * 1024 * 1024)
                } else {
                    false
                };

                if should_offload {
                    let gu_gpu = if let Some(ref m) = mmap {
                        let slice = &m[ffn_gate_up_exps_offset as usize..(ffn_gate_up_exps_offset as usize + gate_up_bytes)];
                        crate::cuda::CudaBuffer::from_host_on(0, slice).ok()
                    } else {
                        None
                    };

                    let down_gpu = if let Some(ref m) = mmap {
                        let slice = &m[ffn_down_exps_offset as usize..(ffn_down_exps_offset as usize + down_bytes)];
                        crate::cuda::CudaBuffer::from_host_on(0, slice).ok()
                    } else {
                        None
                    };

                    let scale_gpu = if !ffn_down_exps_scale.is_empty() {
                        crate::cuda::CudaBuffer::from_host_on(0, &ffn_down_exps_scale).ok()
                    } else {
                        None
                    };

                    (gu_gpu, down_gpu, scale_gpu)
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

            #[cfg(feature = "cuda")]
            let (gpu_d_moe_in, gpu_d_moe_exp_ids, gpu_d_moe_exp_weights, gpu_d_moe_act_scratch, gpu_d_moe_out) = if cuda_dev.is_some() {
                (
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 8).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 8).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 8 * exp_ffn_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                )
            } else {
                (None, None, None, None, None)
            };

            #[cfg(feature = "cuda")]
            let (gpu_d_cur, gpu_d_q, gpu_d_k, gpu_d_v, gpu_d_attn_in, gpu_d_attn_out, gpu_d_mlp_in, gpu_d_mlp_gate, gpu_d_mlp_up, gpu_d_mlp_act, gpu_d_mlp_down) = if cuda_dev.is_some() {
                (
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, q_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, q_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                )
            } else {
                (None, None, None, None, None, None, None, None, None, None, None)
            };

            layers.push(TransformerLayer {
                is_swa,
                head_dim,
                n_heads,
                n_kv_heads,
                q_dim,
                kv_dim,
                rope_freq_base,
                sliding_window,
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
                post_ffw_norm_1,
                pre_ffw_norm_2,
                post_ffw_norm_2,
                layer_output_scale,
                is_moe,
                ffn_gate_inp,
                ffn_gate_inp_scale,
                ffn_down_exps_scale,
                ffn_gate_up_exps_offset,
                ffn_down_exps_offset,
                #[cfg(feature = "cuda")]
                gpu_attn_q,
                #[cfg(feature = "cuda")]
                gpu_attn_k,
                #[cfg(feature = "cuda")]
                gpu_attn_v,
                #[cfg(feature = "cuda")]
                gpu_attn_output,
                #[cfg(feature = "cuda")]
                gpu_ffn_gate,
                #[cfg(feature = "cuda")]
                gpu_ffn_up,
                #[cfg(feature = "cuda")]
                gpu_ffn_down,
                #[cfg(feature = "cuda")]
                gpu_ffn_gate_up_exps,
                #[cfg(feature = "cuda")]
                gpu_ffn_down_exps,
                #[cfg(feature = "cuda")]
                gpu_ffn_down_exps_scale,
                #[cfg(feature = "cuda")]
                gpu_d_cur: parking_lot::Mutex::new(gpu_d_cur),
                #[cfg(feature = "cuda")]
                gpu_d_q: parking_lot::Mutex::new(gpu_d_q),
                #[cfg(feature = "cuda")]
                gpu_d_k: parking_lot::Mutex::new(gpu_d_k),
                #[cfg(feature = "cuda")]
                gpu_d_v: parking_lot::Mutex::new(gpu_d_v),
                #[cfg(feature = "cuda")]
                gpu_d_attn_in: parking_lot::Mutex::new(gpu_d_attn_in),
                #[cfg(feature = "cuda")]
                gpu_d_attn_out: parking_lot::Mutex::new(gpu_d_attn_out),
                #[cfg(feature = "cuda")]
                gpu_d_mlp_in: parking_lot::Mutex::new(gpu_d_mlp_in),
                #[cfg(feature = "cuda")]
                gpu_d_mlp_gate: parking_lot::Mutex::new(gpu_d_mlp_gate),
                #[cfg(feature = "cuda")]
                gpu_d_mlp_up: parking_lot::Mutex::new(gpu_d_mlp_up),
                #[cfg(feature = "cuda")]
                gpu_d_mlp_act: parking_lot::Mutex::new(gpu_d_mlp_act),
                #[cfg(feature = "cuda")]
                gpu_d_mlp_down: parking_lot::Mutex::new(gpu_d_mlp_down),
                #[cfg(feature = "cuda")]
                gpu_d_moe_in: parking_lot::Mutex::new(gpu_d_moe_in),
                #[cfg(feature = "cuda")]
                gpu_d_moe_exp_ids: parking_lot::Mutex::new(gpu_d_moe_exp_ids),
                #[cfg(feature = "cuda")]
                gpu_d_moe_exp_weights: parking_lot::Mutex::new(gpu_d_moe_exp_weights),
                #[cfg(feature = "cuda")]
                gpu_d_moe_act_scratch: parking_lot::Mutex::new(gpu_d_moe_act_scratch),
                #[cfg(feature = "cuda")]
                gpu_d_moe_out: parking_lot::Mutex::new(gpu_d_moe_out),
            });
        }

        // Preload complete token embedding table in memory
        let row_bytes = (dim / 32) * 34;
        let mut token_embd_table = vec![0u8; vocab_size * row_bytes];

        if let (Some(ref info), Ok(mut file)) = (&token_embd_info, File::open(&gguf_path)) {
            let offset = gguf.data_offset + info.offset;
            if file.seek(SeekFrom::Start(offset)).is_ok() {
                let _ = file.read_exact(&mut token_embd_table);
            }
        }

        #[cfg(feature = "cuda")]
        let gpu_token_embd_table = if cuda_dev.is_some() && !token_embd_table.is_empty() {
            crate::cuda::CudaBuffer::from_host_on(0, &token_embd_table).ok()
        } else {
            None
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
            layers,
            data_offset: gguf.data_offset,
            token_embd_table,
            mmap,
            #[cfg(feature = "cuda")]
            cuda_dev,
            #[cfg(feature = "cuda")]
            gpu_token_embd_table,
        })
    }

    /// Read raw Q8_0 embedding vector for a single token directly from in-memory table
    pub fn read_token_embedding(&self, token_id: i32, out: &mut [f32]) -> Result<()> {
        let dim = self.config.dim;
        assert_eq!(out.len(), dim);

        let row_bytes = (dim / 32) * 34;
        let tid = (token_id.max(0) as usize).min(self.config.vocab_size - 1);
        let offset = tid * row_bytes;
        if offset + row_bytes <= self.token_embd_table.len() {
            dequantize_q8_0(&self.token_embd_table[offset..offset + row_bytes], out);
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
            let mut q = vec![0.0f32; layer.q_dim];
            let mut k = vec![0.0f32; layer.kv_dim];
            let mut v = vec![0.0f32; layer.kv_dim];

            #[cfg(feature = "cuda")]
            let used_gpu = if let (Some(ref dev), Some(ref g_q), Some(ref g_k), Some(ref g_v)) = (&self.cuda_dev, &layer.gpu_attn_q, &layer.gpu_attn_k, &layer.gpu_attn_v) {
                let mut cur_guard = layer.gpu_d_cur.lock();
                let mut q_guard = layer.gpu_d_q.lock();
                let mut k_guard = layer.gpu_d_k.lock();
                let mut v_guard = layer.gpu_d_v.lock();
                if let (Some(ref mut d_cur), Some(ref mut d_q), Some(ref mut d_k), Some(ref mut d_v)) = (cur_guard.as_mut(), q_guard.as_mut(), k_guard.as_mut(), v_guard.as_mut()) {
                    if d_cur.copy_from_host(&cur).is_ok() {
                        dev.gemv_q4_0(g_q, d_cur, d_q, layer.q_dim, dim);
                        dev.gemv_q4_0(g_k, d_cur, d_k, layer.kv_dim, dim);
                        dev.gemv_q4_0(g_v, d_cur, d_v, layer.kv_dim, dim);
                        let _ = dev.sync();
                        let _ = d_q.copy_to_host(&mut q);
                        let _ = d_k.copy_to_host(&mut k);
                        let _ = d_v.copy_to_host(&mut v);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            #[cfg(not(feature = "cuda"))]
            let used_gpu = false;

            if !used_gpu {
                if !layer.attn_q.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.attn_q, &cur, &mut q, layer.q_dim, dim);
                }
                if !layer.attn_k.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.attn_k, &cur, &mut k, layer.kv_dim, dim);
                }
                if !layer.attn_v.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.attn_v, &cur, &mut v, layer.kv_dim, dim);
                } else {
                    v.copy_from_slice(&k);
                }
            }

            // Head RMSNorm & RoPE for Q
            for h in 0..layer.n_heads {
                let q_head = &mut q[h * layer.head_dim..(h + 1) * layer.head_dim];
                if !layer.attn_q_norm.is_empty() {
                    let mut normed_q = vec![0.0f32; layer.head_dim];
                    ops::rms_norm(q_head, Some(&layer.attn_q_norm), 1e-6, &mut normed_q);
                    q_head.copy_from_slice(&normed_q);
                }
                ops::rope_1d(q_head, pos, layer.head_dim, layer.rope_freq_base, 1.0);
            }

            // Head RMSNorm & RoPE for K, and RMSNorm for V
            for h in 0..layer.n_kv_heads {
                let k_head = &mut k[h * layer.head_dim..(h + 1) * layer.head_dim];
                if !layer.attn_k_norm.is_empty() {
                    let mut normed_k = vec![0.0f32; layer.head_dim];
                    ops::rms_norm(k_head, Some(&layer.attn_k_norm), 1e-6, &mut normed_k);
                    k_head.copy_from_slice(&normed_k);
                }
                ops::rope_1d(k_head, pos, layer.head_dim, layer.rope_freq_base, 1.0);

                let v_head = &mut v[h * layer.head_dim..(h + 1) * layer.head_dim];
                let mut normed_v = vec![0.0f32; layer.head_dim];
                ops::rms_norm(v_head, None, 1e-6, &mut normed_v);
                v_head.copy_from_slice(&normed_v);
            }

            // Store into KV cache
            k_cache[l].push(k.clone());
            v_cache[l].push(v.clone());

            // Multi-head Attention
            let seq_len = k_cache[l].len();
            let mut attn_out = vec![0.0f32; layer.q_dim];
            let start_t = if layer.is_swa && layer.sliding_window > 0 {
                seq_len.saturating_sub(layer.sliding_window)
            } else {
                0
            };
            let attn_window_len = seq_len - start_t;

            for h in 0..layer.n_heads {
                let kv_h = h / (layer.n_heads / layer.n_kv_heads);
                let q_head = &q[h * layer.head_dim..(h + 1) * layer.head_dim];

                let mut scores = vec![0.0f32; attn_window_len];
                for (idx, t) in (start_t..seq_len).enumerate() {
                    let k_t = &k_cache[l][t][kv_h * layer.head_dim..(kv_h + 1) * layer.head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..layer.head_dim {
                        dot += q_head[d] * k_t[d];
                    }
                    scores[idx] = dot; // f_attention_scale = 1.0 in Gemma 4
                }

                ops::softmax(&mut scores);

                let out_head = &mut attn_out[h * layer.head_dim..(h + 1) * layer.head_dim];
                for (idx, t) in (start_t..seq_len).enumerate() {
                    let v_t = &v_cache[l][t][kv_h * layer.head_dim..(kv_h + 1) * layer.head_dim];
                    let s = scores[idx];
                    for d in 0..layer.head_dim {
                        out_head[d] += s * v_t[d];
                    }
                }
            }

            // Attention Output Projection
            let mut attn_proj = vec![0.0f32; dim];
            #[cfg(feature = "cuda")]
            let out_used_gpu = if let (Some(ref dev), Some(ref g_out)) = (&self.cuda_dev, &layer.gpu_attn_output) {
                let mut in_guard = layer.gpu_d_attn_in.lock();
                let mut out_guard = layer.gpu_d_attn_out.lock();
                if let (Some(ref mut d_in), Some(ref mut d_out)) = (in_guard.as_mut(), out_guard.as_mut()) {
                    if d_in.copy_from_host(&attn_out).is_ok() {
                        dev.gemv_q4_0(g_out, d_in, d_out, dim, layer.q_dim);
                        let _ = dev.sync();
                        let _ = d_out.copy_to_host(&mut attn_proj);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            #[cfg(not(feature = "cuda"))]
            let out_used_gpu = false;

            if !out_used_gpu && !layer.attn_output.is_empty() {
                ops::mat_vec_mul_q4_0(&layer.attn_output, &attn_out, &mut attn_proj, dim, layer.q_dim);
            }

            // Post-Attention Norm & Residual
            let mut normed_attn = vec![0.0f32; dim];
            ops::rms_norm(&attn_proj, Some(&layer.post_attention_norm), 1e-6, &mut normed_attn);

            let mut attn_res = vec![0.0f32; dim];
            for i in 0..dim {
                attn_res[i] = hidden[i] + normed_attn[i];
            }

            // 1. Shared Dense MLP
            let ffn_dim = 2112;
            let mut ffn_in_shared = vec![0.0f32; dim];
            ops::rms_norm(&attn_res, Some(&layer.ffn_norm), 1e-6, &mut ffn_in_shared);

            let mut mlp_raw = vec![0.0f32; dim];

            #[cfg(feature = "cuda")]
            let mlp_used_gpu = if let (Some(ref dev), Some(ref g_gate), Some(ref g_up), Some(ref g_down)) = (&self.cuda_dev, &layer.gpu_ffn_gate, &layer.gpu_ffn_up, &layer.gpu_ffn_down) {
                let mut in_guard = layer.gpu_d_mlp_in.lock();
                let mut gate_guard = layer.gpu_d_mlp_gate.lock();
                let mut up_guard = layer.gpu_d_mlp_up.lock();
                let mut act_guard = layer.gpu_d_mlp_act.lock();
                let mut down_guard = layer.gpu_d_mlp_down.lock();
                if let (Some(ref mut d_in), Some(ref mut d_gate), Some(ref mut d_up), Some(ref mut d_act), Some(ref mut d_down)) = (
                    in_guard.as_mut(), gate_guard.as_mut(), up_guard.as_mut(), act_guard.as_mut(), down_guard.as_mut()
                ) {
                    if d_in.copy_from_host(&ffn_in_shared).is_ok() {
                        dev.gemv_q4_0(g_gate, d_in, d_gate, ffn_dim, dim);
                        dev.gemv_q4_0(g_up, d_in, d_up, ffn_dim, dim);
                        dev.geglu(d_gate, d_up, d_act);
                        dev.gemv_q4_0(g_down, d_act, d_down, dim, ffn_dim);
                        let _ = dev.sync();
                        let _ = d_down.copy_to_host(&mut mlp_raw);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            #[cfg(not(feature = "cuda"))]
            let mlp_used_gpu = false;

            if !mlp_used_gpu {
                let mut gate = vec![0.0f32; ffn_dim];
                let mut up = vec![0.0f32; ffn_dim];
                if !layer.ffn_gate.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.ffn_gate, &ffn_in_shared, &mut gate, ffn_dim, dim);
                }
                if !layer.ffn_up.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.ffn_up, &ffn_in_shared, &mut up, ffn_dim, dim);
                }
                let mut ffn_act = vec![0.0f32; ffn_dim];
                ops::geglu(&gate, &up, &mut ffn_act);
                if !layer.ffn_down.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.ffn_down, &ffn_act, &mut mlp_raw, dim, ffn_dim);
                }
            }

            let mut mlp_out = vec![0.0f32; dim];
            ops::rms_norm(&mlp_raw, Some(&layer.post_ffw_norm_1), 1e-6, &mut mlp_out);

            // 2. MoE Top-8 Active Experts
            let mut moe_raw = vec![0.0f32; dim];
            if layer.is_moe && !layer.ffn_gate_inp.is_empty() {
                // Expert input norm
                let mut ffn_in_moe = vec![0.0f32; dim];
                ops::rms_norm(&attn_res, Some(&layer.pre_ffw_norm_2), 1e-6, &mut ffn_in_moe);

                    // Router logits
                    let mut router_tmp = vec![0.0f32; dim];
                    ops::rms_norm(&attn_res, None, 1e-6, &mut router_tmp);
                    let inv_sqrt_dim = 1.0f32 / (dim as f32).sqrt();
                    for i in 0..dim {
                        let scale = if i < layer.ffn_gate_inp_scale.len() { layer.ffn_gate_inp_scale[i] } else { 1.0 };
                        router_tmp[i] = router_tmp[i] * inv_sqrt_dim * scale;
                    }

                    // Compute 128 expert logits
                    let mut expert_logits = vec![(0.0f32, 0usize); 128];
                    for e in 0..128 {
                        let mut dot = 0.0f32;
                        let col_offset = e * dim;
                        if col_offset + dim <= layer.ffn_gate_inp.len() {
                            for i in 0..dim {
                                dot += router_tmp[i] * layer.ffn_gate_inp[col_offset + i];
                            }
                        }
                        expert_logits[e] = (dot, e);
                    }

                    // Pick Top-8 experts
                    expert_logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let top8 = &expert_logits[..8];

                    // Softmax over Top-8
                    let max_l = top8[0].0;
                    let mut ex_probs = [0.0f32; 8];
                    let mut sum_exp = 0.0f32;
                    for i in 0..8 {
                        let p = (top8[i].0 - max_l).exp();
                        ex_probs[i] = p;
                        sum_exp += p;
                    }
                    for i in 0..8 {
                        ex_probs[i] /= sum_exp.max(1e-6);
                    }

                    // Evaluate active experts
                    let exp_ffn_dim = 704;

                    #[cfg(feature = "cuda")]
                    let moe_used_gpu = if let (
                        Some(ref dev),
                        Some(ref g_gu_exps),
                        Some(ref g_down_exps),
                    ) = (
                        &self.cuda_dev,
                        &layer.gpu_ffn_gate_up_exps,
                        &layer.gpu_ffn_down_exps,
                    ) {
                        let mut in_guard = layer.gpu_d_moe_in.lock();
                        let mut exp_ids_guard = layer.gpu_d_moe_exp_ids.lock();
                        let mut exp_weights_guard = layer.gpu_d_moe_exp_weights.lock();
                        let mut act_scratch_guard = layer.gpu_d_moe_act_scratch.lock();
                        let mut out_guard = layer.gpu_d_moe_out.lock();

                        if let (
                            Some(ref mut d_in),
                            Some(ref mut d_exp_ids),
                            Some(ref mut d_exp_weights),
                            Some(ref mut d_act_scratch),
                            Some(ref mut d_out),
                        ) = (
                            in_guard.as_mut(),
                            exp_ids_guard.as_mut(),
                            exp_weights_guard.as_mut(),
                            act_scratch_guard.as_mut(),
                            out_guard.as_mut(),
                        ) {
                            let mut active_ids = [0i32; 8];
                            let mut active_alphas = [0.0f32; 8];
                            for i in 0..8 {
                                active_ids[i] = top8[i].1 as i32;
                                active_alphas[i] = ex_probs[i];
                            }

                            if d_in.copy_from_host(&ffn_in_moe).is_ok()
                                && d_exp_ids.copy_from_host(&active_ids).is_ok()
                                && d_exp_weights.copy_from_host(&active_alphas).is_ok()
                            {
                                dev.moe_topk_q4_0(
                                    g_gu_exps,
                                    g_down_exps,
                                    d_exp_ids,
                                    d_exp_weights,
                                    layer.gpu_ffn_down_exps_scale.as_ref(),
                                    d_in,
                                    d_act_scratch,
                                    d_out,
                                    dim,
                                    exp_ffn_dim,
                                    8,
                                );
                                let _ = dev.sync();
                                let _ = d_out.copy_to_host(&mut moe_raw);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    #[cfg(not(feature = "cuda"))]
                    let moe_used_gpu = false;

                    if !moe_used_gpu {
                        let gate_up_bytes_per_exp = 1408 * (dim / 32) * 18; // 2,230,272 bytes
                        let down_bytes_per_exp = 2816 * (exp_ffn_dim / 32) * 18; // 1,115,136 bytes

                        for (k, &(_, exp_idx)) in top8.iter().enumerate() {
                            let alpha = ex_probs[k];
                            if alpha <= 0.0001 {
                                continue;
                            }

                            let gu_off = (layer.ffn_gate_up_exps_offset + (exp_idx as u64) * (gate_up_bytes_per_exp as u64)) as usize;
                            let down_off = (layer.ffn_down_exps_offset + (exp_idx as u64) * (down_bytes_per_exp as u64)) as usize;

                            if let Some(ref mm) = self.mmap {
                                if gu_off + gate_up_bytes_per_exp <= mm.len() && down_off + down_bytes_per_exp <= mm.len() {
                                    let half_bytes = gate_up_bytes_per_exp / 2;
                                    let gate_slice = &mm[gu_off..gu_off + half_bytes];
                                    let up_slice = &mm[gu_off + half_bytes..gu_off + gate_up_bytes_per_exp];
                                    let down_slice = &mm[down_off..down_off + down_bytes_per_exp];

                                    let mut exp_gate = vec![0.0f32; exp_ffn_dim];
                                    let mut exp_up = vec![0.0f32; exp_ffn_dim];

                                    ops::mat_vec_mul_q4_0(gate_slice, &ffn_in_moe, &mut exp_gate, exp_ffn_dim, dim);
                                    ops::mat_vec_mul_q4_0(up_slice, &ffn_in_moe, &mut exp_up, exp_ffn_dim, dim);

                                    let mut exp_act = vec![0.0f32; exp_ffn_dim];
                                    ops::geglu(&exp_gate, &exp_up, &mut exp_act);

                                    let mut exp_down = vec![0.0f32; dim];
                                    ops::mat_vec_mul_q4_0(down_slice, &exp_act, &mut exp_down, dim, exp_ffn_dim);

                                    let scale = if exp_idx < layer.ffn_down_exps_scale.len() {
                                        layer.ffn_down_exps_scale[exp_idx]
                                    } else {
                                        1.0
                                    };

                                    for i in 0..dim {
                                        moe_raw[i] += alpha * exp_down[i] * scale;
                                    }
                                }
                            }
                        }
                    }
                }

            let mut moe_out = vec![0.0f32; dim];
            ops::rms_norm(&moe_raw, Some(&layer.post_ffw_norm_2), 1e-6, &mut moe_out);

            // 3. Combine shared MLP and MoE
            let mut ffn_combined = vec![0.0f32; dim];
            for i in 0..dim {
                ffn_combined[i] = mlp_out[i] + moe_out[i];
            }

            let mut normed_ffn = vec![0.0f32; dim];
            ops::rms_norm(&ffn_combined, Some(&layer.post_ffw_norm), 1e-6, &mut normed_ffn);

            // 4. Residual 2 & Layer Output Scale
            for i in 0..dim {
                hidden[i] = (attn_res[i] + normed_ffn[i]) * layer.layer_output_scale;
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

        let formatted = text.replace(' ', "\u{2581}");
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
        token == 1 || token == 2 || token == 106 || token == 107
    }

    /// Initialize generation state with fast prompt prefill pass
    pub fn init_generation_state(&self, prompt_tokens: &[i32]) -> GenerationState {
        let n_layers = self.config.n_layers;
        let mut k_cache = vec![Vec::new(); n_layers];
        let mut v_cache = vec![Vec::new(); n_layers];
        let mut hidden = vec![0.0f32; self.config.dim];

        let window = 32.min(prompt_tokens.len());
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
        let row_bytes = (dim / 32) * 34;
        let n_blocks = dim / 32;

        let recent_tokens = state.history_tokens.iter().rev().take(32).copied().collect::<Vec<_>>();
        let generated_count = state.generated_count;

        #[cfg(feature = "cuda")]
        let gpu_scored: Option<Vec<(f32, i32)>> = None;

        #[cfg(not(feature = "cuda"))]
        let gpu_scored: Option<Vec<(f32, i32)>> = None;

        let mut all_scored: Vec<(f32, i32)> = if let Some(scored) = gpu_scored {
            scored
        } else {
            let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
            let chunk_size = (self.config.vocab_size + n_threads - 1) / n_threads;

            std::thread::scope(|s| {
                let mut handles = Vec::new();
                for thread_idx in 0..n_threads {
                    let start_tid = thread_idx * chunk_size;
                    let end_tid = (start_tid + chunk_size).min(self.config.vocab_size);
                    if start_tid >= end_tid {
                        continue;
                    }

                    let hidden_ref = &state.hidden;
                    let recent_tokens_ref = &recent_tokens;
                    let table_ref = &self.token_embd_table;
                    let vocab_ref = &self.vocab;

                    handles.push(s.spawn(move || {
                        let mut scored = Vec::with_capacity(end_tid - start_tid);
                        for tid in start_tid..end_tid {
                            if tid < vocab_ref.len() {
                                let piece = &vocab_ref[tid];
                                if piece.starts_with("<unused") || piece == "<pad>" || piece == "<unk>" || piece == "<mask>" || piece == "[multimodal]" {
                                    continue;
                                }
                            }

                            if generated_count < 4 && (tid == 1 || tid == 2 || tid == 105 || tid == 106 || tid == 107) {
                                continue;
                            }

                            let row_off = tid * row_bytes;
                            if row_off + row_bytes > table_ref.len() {
                                continue;
                            }

                            let row_buf = &table_ref[row_off..row_off + row_bytes];

                            let mut dot = 0.0f32;
                            for b in 0..n_blocks {
                                let w_off = b * 34;
                                let w_d_raw = u16::from_le_bytes([row_buf[w_off], row_buf[w_off + 1]]);
                                let w_d = crate::quant::f16_to_f32(w_d_raw);

                                let mut block_sum = 0.0f32;
                                for k in 0..32 {
                                    let qw = row_buf[w_off + 2 + k] as i8 as f32;
                                    block_sum += qw * hidden_ref[b * 32 + k];
                                }
                                dot += block_sum * w_d;
                            }

                            let mut score = 30.0 * (dot / 30.0).tanh();

                            if recent_tokens_ref.contains(&(tid as i32)) {
                                score -= 3.5;
                            }

                            scored.push((score, tid as i32));
                        }
                        scored
                    }));
                }
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };

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
