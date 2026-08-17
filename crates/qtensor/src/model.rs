use crate::device::{DeviceManager, DeviceType};
use crate::gguf::{GgufFile, GgufTensorInfo};
use crate::kv_cache::KvCacheManager;
use crate::ops;
use crate::quant::dequantize_q8_0;
use anyhow::Result;
use rayon::prelude::*;
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
    pub gpu_attn_q_norm: Option<crate::cuda::CudaBuffer<f32>>,
    #[cfg(feature = "cuda")]
    pub gpu_attn_k_norm: Option<crate::cuda::CudaBuffer<f32>>,
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
    pub gpu_d_qkv: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_k_cache: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_v_cache: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_kv_capacity: usize,
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
    pub valid_vocab_token: Vec<bool>,
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
    #[cfg(feature = "cuda")]
    pub gpu_d_final_hidden: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_vocab_logits: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_valid_vocab: Option<crate::cuda::CudaBuffer<u8>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_recent_tokens: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<i32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_topk_scores: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<f32>>>,
    #[cfg(feature = "cuda")]
    pub gpu_d_topk_ids: parking_lot::Mutex<Option<crate::cuda::CudaBuffer<i32>>>,
}

impl QTensorModel {
    pub fn load_from_gguf<P: AsRef<Path>>(path: P, max_context: usize) -> Result<Self> {
        let gguf_path = path.as_ref().to_path_buf();
        let gguf = GgufFile::open(&gguf_path)?;

        if gguf
            .get_meta("general.architecture")
            .and_then(|v| v.as_str())
            == Some("gemma4-assistant")
        {
            anyhow::bail!(
                "Gemma 4 assistant GGUFs are MTP draft heads and require a target model; \
                 they cannot be loaded as a standalone generation model"
            );
        }
        
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
        let valid_vocab_token: Vec<bool> = (0..vocab_size)
            .map(|tid| {
                vocab.get(tid).map_or(true, |piece| {
                    !piece.starts_with("<unused")
                        && piece != "<pad>"
                        && piece != "<unk>"
                        && piece != "<mask>"
                        && piece != "[multimodal]"
                })
            })
            .collect();

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
            let gpu_attn_q_norm = if cuda_dev.is_some() { crate::cuda::CudaBuffer::from_host_on(0, &attn_q_norm).ok() } else { None };
            #[cfg(feature = "cuda")]
            let gpu_attn_k_norm = if cuda_dev.is_some() { crate::cuda::CudaBuffer::from_host_on(0, &attn_k_norm).ok() } else { None };
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
            let (gpu_d_cur, gpu_d_q, gpu_d_k, gpu_d_v, gpu_d_qkv, gpu_d_attn_in, gpu_d_attn_out, gpu_d_mlp_in, gpu_d_mlp_gate, gpu_d_mlp_up, gpu_d_mlp_act, gpu_d_mlp_down) = if cuda_dev.is_some() {
                (
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, q_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, q_dim + 2 * kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, q_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, 2112).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                )
            } else {
                (None, None, None, None, None, None, None, None, None, None, None, None)
            };

            #[cfg(feature = "cuda")]
            let gpu_kv_capacity = max_context.min(1024);
            #[cfg(feature = "cuda")]
            let (gpu_d_k_cache, gpu_d_v_cache) = if cuda_dev.is_some() {
                (
                    crate::cuda::CudaBuffer::alloc_on(0, gpu_kv_capacity * kv_dim).ok(),
                    crate::cuda::CudaBuffer::alloc_on(0, gpu_kv_capacity * kv_dim).ok(),
                )
            } else {
                (None, None)
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
                gpu_attn_q_norm,
                #[cfg(feature = "cuda")]
                gpu_attn_k_norm,
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
                gpu_d_qkv: parking_lot::Mutex::new(gpu_d_qkv),
                #[cfg(feature = "cuda")]
                gpu_d_k_cache: parking_lot::Mutex::new(gpu_d_k_cache),
                #[cfg(feature = "cuda")]
                gpu_d_v_cache: parking_lot::Mutex::new(gpu_d_v_cache),
                #[cfg(feature = "cuda")]
                gpu_kv_capacity,
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

        #[cfg(feature = "cuda")]
        let gpu_valid_vocab = if cuda_dev.is_some() {
            let bytes: Vec<u8> = valid_vocab_token.iter().map(|&v| u8::from(v)).collect();
            crate::cuda::CudaBuffer::from_host_on(0, &bytes).ok()
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let (gpu_d_final_hidden, gpu_d_vocab_logits) = if cuda_dev.is_some() {
            (
                crate::cuda::CudaBuffer::alloc_on(0, dim).ok(),
                crate::cuda::CudaBuffer::alloc_on(0, vocab_size).ok(),
            )
        } else {
            (None, None)
        };

        #[cfg(feature = "cuda")]
        let (gpu_d_recent_tokens, gpu_d_topk_scores, gpu_d_topk_ids) = if cuda_dev.is_some() {
            (
                crate::cuda::CudaBuffer::alloc_on(0, 32).ok(),
                crate::cuda::CudaBuffer::alloc_on(0, 128 * 40).ok(),
                crate::cuda::CudaBuffer::alloc_on(0, 128 * 40).ok(),
            )
        } else {
            (None, None, None)
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
            valid_vocab_token,
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
            #[cfg(feature = "cuda")]
            gpu_d_final_hidden: parking_lot::Mutex::new(gpu_d_final_hidden),
            #[cfg(feature = "cuda")]
            gpu_d_vocab_logits: parking_lot::Mutex::new(gpu_d_vocab_logits),
            #[cfg(feature = "cuda")]
            gpu_valid_vocab,
            #[cfg(feature = "cuda")]
            gpu_d_recent_tokens: parking_lot::Mutex::new(gpu_d_recent_tokens),
            #[cfg(feature = "cuda")]
            gpu_d_topk_scores: parking_lot::Mutex::new(gpu_d_topk_scores),
            #[cfg(feature = "cuda")]
            gpu_d_topk_ids: parking_lot::Mutex::new(gpu_d_topk_ids),
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
        let profile = std::env::var_os("QTENSOR_PROFILE").is_some();
        let mut profile_qkv = std::time::Duration::ZERO;
        let mut profile_attention = std::time::Duration::ZERO;
        let mut profile_output = std::time::Duration::ZERO;
        let mut profile_ffn = std::time::Duration::ZERO;

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
            let profile_start = std::time::Instant::now();

            #[cfg(feature = "cuda")]
            let gpu_qkv_pipeline = if k_cache[l].len() < layer.gpu_kv_capacity {
                if let (
                    Some(ref dev), Some(ref g_q), Some(ref g_k), Some(ref g_v),
                    Some(ref q_norm), Some(ref k_norm),
                ) = (
                    &self.cuda_dev, &layer.gpu_attn_q, &layer.gpu_attn_k, &layer.gpu_attn_v,
                    &layer.gpu_attn_q_norm, &layer.gpu_attn_k_norm,
                ) {
                let mut cur_guard = layer.gpu_d_cur.lock();
                let mut qkv_guard = layer.gpu_d_qkv.lock();
                let mut kc_guard = layer.gpu_d_k_cache.lock();
                let mut vc_guard = layer.gpu_d_v_cache.lock();
                if let (Some(ref mut d_cur), Some(ref mut d_qkv), Some(ref mut d_kc), Some(ref mut d_vc)) =
                    (cur_guard.as_mut(), qkv_guard.as_mut(), kc_guard.as_mut(), vc_guard.as_mut())
                {
                    if dev.copy_from_host_async(d_cur, &cur).is_ok() {
                        dev.gemv_q4_0_qkv(g_q, g_k, g_v, d_cur, d_qkv, layer.q_dim, layer.kv_dim, dim);
                        dev.qkv_postprocess(
                            d_qkv, q_norm, k_norm, d_kc, d_vc, pos, k_cache[l].len(),
                            layer.n_heads, layer.n_kv_heads, layer.head_dim, layer.rope_freq_base,
                        );
                        true
                    } else {
                        false
                    }
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
            let gpu_qkv_pipeline = false;

            if !gpu_qkv_pipeline {
                let mut cur_q8 = vec![0u8; ops::q8_0_size(dim)];
                crate::quant::quantize_f32_to_q8_0(&cur, &mut cur_q8);
                if !layer.attn_q.is_empty() {
                    ops::mat_vec_mul_q4_0_q8_0(&layer.attn_q, &cur_q8, &mut q, layer.q_dim, dim);
                }
                if !layer.attn_k.is_empty() {
                    ops::mat_vec_mul_q4_0_q8_0(&layer.attn_k, &cur_q8, &mut k, layer.kv_dim, dim);
                }
                if !layer.attn_v.is_empty() {
                    ops::mat_vec_mul_q4_0_q8_0(&layer.attn_v, &cur_q8, &mut v, layer.kv_dim, dim);
                } else {
                    v.copy_from_slice(&k);
                }
            }

            // CPU fallback head normalization and RoPE.
            if !gpu_qkv_pipeline { for h in 0..layer.n_heads {
                let q_head = &mut q[h * layer.head_dim..(h + 1) * layer.head_dim];
                if !layer.attn_q_norm.is_empty() {
                    ops::rms_norm_inplace(q_head, Some(&layer.attn_q_norm), 1e-6);
                }
            } }
            profile_qkv += profile_start.elapsed();
            if !gpu_qkv_pipeline {
                ops::rope_1d_batched(&mut q, pos, layer.n_heads, layer.head_dim, layer.rope_freq_base, 1.0);
            }

            // Head RMSNorm & RoPE for K, and RMSNorm for V
            if !gpu_qkv_pipeline { for h in 0..layer.n_kv_heads {
                let k_head = &mut k[h * layer.head_dim..(h + 1) * layer.head_dim];
                if !layer.attn_k_norm.is_empty() {
                    ops::rms_norm_inplace(k_head, Some(&layer.attn_k_norm), 1e-6);
                }

                let v_head = &mut v[h * layer.head_dim..(h + 1) * layer.head_dim];
                ops::rms_norm_inplace(v_head, None, 1e-6);
            } }
            if !gpu_qkv_pipeline {
                ops::rope_1d_batched(&mut k, pos, layer.n_kv_heads, layer.head_dim, layer.rope_freq_base, 1.0);
            }

            // Store into KV cache
            if gpu_qkv_pipeline {
                k_cache[l].push(Vec::new());
                v_cache[l].push(Vec::new());
            } else {
                k_cache[l].push(k.clone());
                v_cache[l].push(v.clone());
            }

            // Multi-head Attention
            let profile_start = std::time::Instant::now();
            let seq_len = k_cache[l].len();
            let mut attn_out = vec![0.0f32; layer.q_dim];
            let start_t = if layer.is_swa && layer.sliding_window > 0 {
                seq_len.saturating_sub(layer.sliding_window)
            } else {
                0
            };
            let attn_window_len = seq_len - start_t;
            let mut scores = vec![0.0f32; attn_window_len];

            #[cfg(feature = "cuda")]
            let attention_used_gpu = if seq_len <= layer.gpu_kv_capacity {
                if let Some(ref dev) = self.cuda_dev {
                    let mut kc_guard = layer.gpu_d_k_cache.lock();
                    let mut vc_guard = layer.gpu_d_v_cache.lock();
                    let mut out_guard = layer.gpu_d_attn_in.lock();
                    if let (Some(ref mut d_kc), Some(ref mut d_vc), Some(ref mut d_out)) =
                        (kc_guard.as_mut(), vc_guard.as_mut(), out_guard.as_mut()) {
                        if gpu_qkv_pipeline {
                            let qkv_guard = layer.gpu_d_qkv.lock();
                            if let Some(ref d_qkv) = qkv_guard.as_ref() {
                                dev.attention_causal(
                                    d_qkv, d_kc, d_vc, d_out, seq_len - 1,
                                    layer.n_heads, layer.n_kv_heads, layer.head_dim, 1.0,
                                    if layer.is_swa { Some(layer.sliding_window) } else { None },
                                );
                                true
                            } else {
                                false
                            }
                        } else {
                            let mut q_guard = layer.gpu_d_q.lock();
                            if let Some(ref mut d_q) = q_guard.as_mut() {
                                let offset = (seq_len - 1) * layer.kv_dim;
                                if dev.copy_from_host_async(d_q, &q).is_ok()
                                    && dev.copy_from_host_at_async(d_kc, offset, &k).is_ok()
                                    && dev.copy_from_host_at_async(d_vc, offset, &v).is_ok() {
                            dev.attention_causal(
                                d_q,
                                d_kc,
                                d_vc,
                                d_out,
                                seq_len - 1,
                                layer.n_heads,
                                layer.n_kv_heads,
                                layer.head_dim,
                                1.0,
                                if layer.is_swa { Some(layer.sliding_window) } else { None },
                            );
                            d_out.copy_to_host(&mut attn_out).is_ok()
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
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
            let attention_used_gpu = false;

            if !attention_used_gpu { for h in 0..layer.n_heads {
                let kv_h = h / (layer.n_heads / layer.n_kv_heads);
                let q_head = &q[h * layer.head_dim..(h + 1) * layer.head_dim];

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
            } }
            profile_attention += profile_start.elapsed();

            // Attention Output Projection
            let profile_start = std::time::Instant::now();
            let mut attn_proj = vec![0.0f32; dim];
            #[cfg(feature = "cuda")]
            let out_used_gpu = if let (Some(ref dev), Some(ref g_out)) = (&self.cuda_dev, &layer.gpu_attn_output) {
                let mut in_guard = layer.gpu_d_attn_in.lock();
                let mut out_guard = layer.gpu_d_attn_out.lock();
                if let (Some(ref mut d_in), Some(ref mut d_out)) = (in_guard.as_mut(), out_guard.as_mut()) {
                    if attention_used_gpu || dev.copy_from_host_async(d_in, &attn_out).is_ok() {
                        dev.gemv_q4_0(g_out, d_in, d_out, dim, layer.q_dim);
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
            profile_output += profile_start.elapsed();

            // 1. Prepare Shared Dense MLP & MoE inputs
            let profile_start = std::time::Instant::now();
            let ffn_dim = 2112;
            let mut ffn_in_shared = vec![0.0f32; dim];
            ops::rms_norm(&attn_res, Some(&layer.ffn_norm), 1e-6, &mut ffn_in_shared);

            let mut ffn_in_moe = vec![0.0f32; dim];
            if layer.is_moe && !layer.ffn_gate_inp.is_empty() {
                ops::rms_norm(&attn_res, Some(&layer.pre_ffw_norm_2), 1e-6, &mut ffn_in_moe);
            }

            // Router logits & Top-8 selection
            let mut top8_experts = Vec::new();
            let mut ex_probs = [0.0f32; 8];
            if layer.is_moe && !layer.ffn_gate_inp.is_empty() {
                let mut router_tmp = vec![0.0f32; dim];
                ops::rms_norm(&attn_res, None, 1e-6, &mut router_tmp);
                let inv_sqrt_dim = 1.0f32 / (dim as f32).sqrt();
                for i in 0..dim {
                    let scale = if i < layer.ffn_gate_inp_scale.len() { layer.ffn_gate_inp_scale[i] } else { 1.0 };
                    router_tmp[i] = router_tmp[i] * inv_sqrt_dim * scale;
                }

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

                expert_logits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let max_l = expert_logits[0].0;
                let mut sum_exp = 0.0f32;
                for i in 0..8 {
                    top8_experts.push(expert_logits[i].1);
                    let p = (expert_logits[i].0 - max_l).exp();
                    ex_probs[i] = p;
                    sum_exp += p;
                }
                for i in 0..8 {
                    ex_probs[i] /= sum_exp.max(1e-6);
                }
            }

            // Queue GPU kernels for both Dense MLP & MoE simultaneously
            let mut mlp_raw = vec![0.0f32; dim];
            let mut moe_raw = vec![0.0f32; dim];

            #[cfg(feature = "cuda")]
            let (mlp_queued, moe_queued) = if let Some(ref dev) = self.cuda_dev {
                let mlp_q = if let (Some(ref g_gate), Some(ref g_up), Some(ref g_down)) = (&layer.gpu_ffn_gate, &layer.gpu_ffn_up, &layer.gpu_ffn_down) {
                    let mut in_g = layer.gpu_d_mlp_in.lock();
                    let mut gate_g = layer.gpu_d_mlp_gate.lock();
                    let mut up_g = layer.gpu_d_mlp_up.lock();
                    let mut act_g = layer.gpu_d_mlp_act.lock();
                    let mut down_g = layer.gpu_d_mlp_down.lock();
                    if let (Some(ref mut d_in), Some(ref mut _d_gate), Some(ref mut _d_up), Some(ref mut d_act), Some(ref mut d_down)) = (
                        in_g.as_mut(), gate_g.as_mut(), up_g.as_mut(), act_g.as_mut(), down_g.as_mut()
                    ) {
                        if dev.copy_from_host_async(d_in, &ffn_in_shared).is_ok() {
                            dev.gemv_q4_0_geglu(g_gate, g_up, d_in, d_act, ffn_dim, dim);
                            dev.gemv_q4_0(g_down, d_act, d_down, dim, ffn_dim);
                            true
                        } else { false }
                    } else { false }
                } else { false };

                let moe_q = if layer.is_moe && !top8_experts.is_empty() {
                    if let (Some(ref g_gu_exps), Some(ref g_down_exps)) = (&layer.gpu_ffn_gate_up_exps, &layer.gpu_ffn_down_exps) {
                        let mut in_g = layer.gpu_d_moe_in.lock();
                        let mut ids_g = layer.gpu_d_moe_exp_ids.lock();
                        let mut weights_g = layer.gpu_d_moe_exp_weights.lock();
                        let mut act_scratch_g = layer.gpu_d_moe_act_scratch.lock();
                        let mut out_g = layer.gpu_d_moe_out.lock();
                        if let (Some(ref mut d_in), Some(ref mut d_ids), Some(ref mut d_weights), Some(ref mut d_act_scratch), Some(ref mut d_out)) = (
                            in_g.as_mut(), ids_g.as_mut(), weights_g.as_mut(), act_scratch_g.as_mut(), out_g.as_mut()
                        ) {
                            let mut active_ids = [0i32; 8];
                            for i in 0..8 { active_ids[i] = top8_experts[i] as i32; }
                            if dev.copy_from_host_async(d_in, &ffn_in_moe).is_ok()
                                && dev.copy_from_host_async(d_ids, &active_ids).is_ok()
                                && dev.copy_from_host_async(d_weights, &ex_probs).is_ok()
                            {
                                dev.moe_topk_q4_0(
                                    g_gu_exps,
                                    g_down_exps,
                                    d_ids,
                                    d_weights,
                                    layer.gpu_ffn_down_exps_scale.as_ref(),
                                    d_in,
                                    d_act_scratch,
                                    d_out,
                                    dim,
                                    704,
                                    8,
                                );
                                true
                            } else { false }
                        } else { false }
                    } else { false }
                } else { false };

                if mlp_q || moe_q {
                    if mlp_q {
                        let down_g = layer.gpu_d_mlp_down.lock();
                        if let Some(ref d_down) = down_g.as_ref() {
                            let _ = d_down.copy_to_host(&mut mlp_raw);
                        }
                    }
                    if moe_q {
                        let out_g = layer.gpu_d_moe_out.lock();
                        if let Some(ref d_out) = out_g.as_ref() {
                            let _ = d_out.copy_to_host(&mut moe_raw);
                        }
                    }
                }

                (mlp_q, moe_q)
            } else {
                (false, false)
            };

            #[cfg(not(feature = "cuda"))]
            let (mlp_queued, moe_queued) = (false, false);

            if !mlp_queued {
                let mut gate = vec![0.0f32; ffn_dim];
                let mut up = vec![0.0f32; ffn_dim];
                let mut ffn_q8 = vec![0u8; ops::q8_0_size(dim)];
                crate::quant::quantize_f32_to_q8_0(&ffn_in_shared, &mut ffn_q8);
                if !layer.ffn_gate.is_empty() {
                    ops::mat_vec_mul_q4_0_q8_0(&layer.ffn_gate, &ffn_q8, &mut gate, ffn_dim, dim);
                }
                if !layer.ffn_up.is_empty() {
                    ops::mat_vec_mul_q4_0_q8_0(&layer.ffn_up, &ffn_q8, &mut up, ffn_dim, dim);
                }
                let mut ffn_act = vec![0.0f32; ffn_dim];
                ops::geglu(&gate, &up, &mut ffn_act);
                if !layer.ffn_down.is_empty() {
                    ops::mat_vec_mul_q4_0(&layer.ffn_down, &ffn_act, &mut mlp_raw, dim, ffn_dim);
                }
            }

            if !moe_queued && layer.is_moe && !top8_experts.is_empty() {
                let gate_up_bytes_per_exp = 1408 * (dim / 32) * 18;
                let down_bytes_per_exp = 2816 * (704 / 32) * 18;
                let mut moe_q8 = vec![0u8; ops::q8_0_size(dim)];
                crate::quant::quantize_f32_to_q8_0(&ffn_in_moe, &mut moe_q8);

                for (k, &exp_idx) in top8_experts.iter().enumerate() {
                    let alpha = ex_probs[k];
                    if alpha <= 0.0001 { continue; }

                    let gu_off = (layer.ffn_gate_up_exps_offset + (exp_idx as u64) * (gate_up_bytes_per_exp as u64)) as usize;
                    let down_off = (layer.ffn_down_exps_offset + (exp_idx as u64) * (down_bytes_per_exp as u64)) as usize;

                    if let Some(ref mm) = self.mmap {
                        if gu_off + gate_up_bytes_per_exp <= mm.len() && down_off + down_bytes_per_exp <= mm.len() {
                            let half_bytes = gate_up_bytes_per_exp / 2;
                            let gate_slice = &mm[gu_off..gu_off + half_bytes];
                            let up_slice = &mm[gu_off + half_bytes..gu_off + gate_up_bytes_per_exp];
                            let down_slice = &mm[down_off..down_off + down_bytes_per_exp];

                            let mut exp_gate = vec![0.0f32; 704];
                            let mut exp_up = vec![0.0f32; 704];

                            ops::mat_vec_mul_q4_0_q8_0(gate_slice, &moe_q8, &mut exp_gate, 704, dim);
                            ops::mat_vec_mul_q4_0_q8_0(up_slice, &moe_q8, &mut exp_up, 704, dim);

                            let mut exp_act = vec![0.0f32; 704];
                            ops::geglu(&exp_gate, &exp_up, &mut exp_act);

                            let mut exp_down = vec![0.0f32; dim];
                            ops::mat_vec_mul_q4_0(down_slice, &exp_act, &mut exp_down, dim, 704);

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

            let mut mlp_out = vec![0.0f32; dim];
            ops::rms_norm(&mlp_raw, Some(&layer.post_ffw_norm_1), 1e-6, &mut mlp_out);

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
            profile_ffn += profile_start.elapsed();
        }

        // Final output RMSNorm
        let mut final_hidden = vec![0.0f32; dim];
        ops::rms_norm(&hidden, Some(&self.output_norm_weights), 1e-6, &mut final_hidden);

        if profile {
            eprintln!(
                "[profile pos={pos}] qkv={:.2}ms attention={:.2}ms output={:.2}ms ffn={:.2}ms",
                profile_qkv.as_secs_f64() * 1000.0,
                profile_attention.as_secs_f64() * 1000.0,
                profile_output.as_secs_f64() * 1000.0,
                profile_ffn.as_secs_f64() * 1000.0,
            );
        }

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
        let profile = std::env::var_os("QTENSOR_PROFILE").is_some();
        let sample_start = std::time::Instant::now();
        let dim = self.config.dim;
        let row_bytes = (dim / 32) * 34;

        let recent_tokens = state.history_tokens.iter().rev().take(32).copied().collect::<Vec<_>>();
        let generated_count = state.generated_count;

        #[cfg(feature = "cuda")]
        let gpu_scored: Option<Vec<(f32, i32)>> = if let (Some(ref dev), Some(ref g_table)) = (&self.cuda_dev, &self.gpu_token_embd_table) {
            let mut hid_guard = self.gpu_d_final_hidden.lock();
            let mut log_guard = self.gpu_d_vocab_logits.lock();
            let mut recent_guard = self.gpu_d_recent_tokens.lock();
            let mut scores_guard = self.gpu_d_topk_scores.lock();
            let mut ids_guard = self.gpu_d_topk_ids.lock();
            if let (Some(ref mut d_hid), Some(ref mut d_log), Some(ref valid), Some(ref mut d_recent), Some(ref mut d_scores), Some(ref mut d_ids)) = (
                hid_guard.as_mut(), log_guard.as_mut(), self.gpu_valid_vocab.as_ref(),
                recent_guard.as_mut(), scores_guard.as_mut(), ids_guard.as_mut(),
            ) {
                if dev.copy_from_host_async(d_hid, &state.hidden).is_ok() {
                    dev.gemv_q8_0(g_table, d_hid, d_log, self.config.vocab_size, dim);
                    if std::env::var_os("QTENSOR_FULL_LOGITS").is_some() {
                        let mut logits = vec![0.0f32; self.config.vocab_size];
                        if d_log.copy_to_host(&mut logits).is_ok() {
                            for &token in &recent_tokens {
                                if let Some(logit) = logits.get_mut(token.max(0) as usize) {
                                    *logit -= 1.8;
                                }
                            }
                            Some(logits.into_iter().enumerate().filter_map(|(tid, score)| {
                                if !self.valid_vocab_token[tid]
                                    || (generated_count < 4 && matches!(tid, 1 | 2 | 105 | 106 | 107))
                                {
                                    None
                                } else {
                                    Some((score, tid as i32))
                                }
                            }).collect())
                        } else {
                            None
                        }
                    } else {
                        let mut recent_upload = [0i32; 32];
                        recent_upload[..recent_tokens.len()].copy_from_slice(&recent_tokens);
                        if dev.copy_from_host_async(d_recent, &recent_upload).is_ok() {
                            const PARTITIONS: usize = 128;
                            const K: usize = 40;
                            dev.vocab_topk(
                                d_log, valid, d_recent, d_scores, d_ids,
                                self.config.vocab_size, recent_tokens.len(), generated_count,
                                K, PARTITIONS,
                            );
                            let mut scores = vec![0.0f32; PARTITIONS * K];
                            let mut ids = vec![0i32; PARTITIONS * K];
                            if d_scores.copy_to_host(&mut scores).is_ok() && d_ids.copy_to_host(&mut ids).is_ok() {
                                Some(scores.into_iter().zip(ids).filter(|(_, id)| *id >= 0).collect())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_scored: Option<Vec<(f32, i32)>> = None;

        let mut all_scored: Vec<(f32, i32)> = if let Some(scored) = gpu_scored {
            scored
        } else {
            let mut hidden_q8 = vec![0u8; row_bytes];
            crate::quant::quantize_f32_to_q8_0(&state.hidden, &mut hidden_q8);

            self.token_embd_table
                .par_chunks_exact(row_bytes)
                .enumerate()
                .take(self.config.vocab_size)
                .filter_map(|(tid, row)| {
                    if !self.valid_vocab_token[tid] {
                        return None;
                    }
                    if generated_count < 4 && (tid == 1 || tid == 2 || tid == 105 || tid == 106 || tid == 107) {
                        return None;
                    }

                    let dot = crate::quant::vec_dot_q8_0_q8_0(row, &hidden_q8, dim);
                    let mut score = 30.0 * (dot / 30.0).tanh();
                    if recent_tokens.contains(&(tid as i32)) {
                        score -= 3.5;
                    }
                    Some((score, tid as i32))
                })
                .collect()
        };

        // Partition in linear time before sorting the small candidate set. A
        // full sort of a 256k-token vocabulary was needlessly O(vocab log vocab).
        let score_order = |a: &(f32, i32), b: &(f32, i32)| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        };
        if all_scored.len() > 40 {
            all_scored.select_nth_unstable_by(40, score_order);
            all_scored.truncate(40);
        }
        all_scored.sort_unstable_by(score_order);

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

        if profile {
            eprintln!(
                "[profile pos={}] vocab+sample={:.2}ms",
                state.pos,
                sample_start.elapsed().as_secs_f64() * 1000.0,
            );
        }

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
