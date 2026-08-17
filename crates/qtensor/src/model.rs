use crate::device::{DeviceManager, DeviceType};
use crate::gguf::GgufFile;
use crate::kv_cache::KvCacheManager;
use anyhow::Result;
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

        let config = ModelConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size: 256000,
            sliding_window,
            rope_freq_base: 500000.0,
            rope_freq_scale: 1.0,
            rms_norm_eps: 1e-6,
            max_context: max_context.max(8192),
        };

        let device_manager = DeviceManager::new();
        // Estimate ~450 MB per layer in Q4_0
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
            "[qtensor] Initialized model: {} layers, {} dim, {} max_context ({} GPU layers, {} CPU layers)",
            n_layers,
            dim,
            config.max_context,
            layer_devices.iter().filter(|d| matches!(d, DeviceType::Cuda(_))).count(),
            layer_devices.iter().filter(|d| matches!(d, DeviceType::Cpu)).count(),
        );

        Ok(Self {
            config,
            device_manager,
            kv_cache,
            layer_devices,
        })
    }
}
