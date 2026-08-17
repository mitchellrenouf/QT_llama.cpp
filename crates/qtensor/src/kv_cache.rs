use anyhow::Result;

#[cfg(feature = "cuda")]
use crate::cuda::CudaBuffer;

/// Layer-specific KV Cache with Sliding Window Attention (SWA) and 256k support
pub struct LayerKvCache {
    pub is_sliding_window: bool,
    pub sliding_window: usize,
    pub max_capacity: usize,
    pub cur_seq_len: usize,

    #[cfg(feature = "cuda")]
    pub d_k: Option<CudaBuffer<f32>>,
    #[cfg(feature = "cuda")]
    pub d_v: Option<CudaBuffer<f32>>,

    pub host_k: Vec<f32>,
    pub host_v: Vec<f32>,
}

impl LayerKvCache {
    pub fn new(
        n_kv_heads: usize,
        head_dim: usize,
        max_context: usize,
        sliding_window: Option<usize>,
        on_device: bool,
        device_id: i32,
    ) -> Result<Self> {
        let is_swa = sliding_window.is_some();
        let sw_size = sliding_window.unwrap_or(max_context);
        let cap = if is_swa { sw_size.min(max_context) } else { max_context };

        // GenerationState owns the active cache. Keep this manager metadata-only
        // until its storage is wired into the forward pass.
        let _ = (n_kv_heads, head_dim, on_device, device_id);

        Ok(Self {
            is_sliding_window: is_swa,
            sliding_window: sw_size,
            max_capacity: cap,
            cur_seq_len: 0,
            #[cfg(feature = "cuda")]
            d_k: None,
            #[cfg(feature = "cuda")]
            d_v: None,
            host_k: Vec::new(),
            host_v: Vec::new(),
        })
    }

    pub fn clear(&mut self) {
        self.cur_seq_len = 0;
    }

    pub fn truncate(&mut self, len: usize) {
        if self.cur_seq_len > len {
            self.cur_seq_len = len;
        }
    }

    pub fn step_increment(&mut self) {
        self.cur_seq_len += 1;
    }
}

/// Prompt Prefix Cache for 0ms initial prefill on recurring system instructions & rules
#[derive(Default)]
pub struct PrefixCache {
    cached_tokens: Vec<i32>,
    cached_len: usize,
}

impl PrefixCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Match current prompt tokens against cached prefix to find longest common prefix
    pub fn match_prefix(&self, tokens: &[i32]) -> usize {
        let max_match = self.cached_tokens.len().min(tokens.len());
        let mut match_len = 0;
        for i in 0..max_match {
            if self.cached_tokens[i] == tokens[i] {
                match_len += 1;
            } else {
                break;
            }
        }
        match_len
    }

    /// Update the cached prefix with newly evaluated tokens
    pub fn update(&mut self, tokens: &[i32], len: usize) {
        self.cached_tokens = tokens[..len.min(tokens.len())].to_vec();
        self.cached_len = self.cached_tokens.len();
    }

    pub fn clear(&mut self) {
        self.cached_tokens.clear();
        self.cached_len = 0;
    }

    pub fn len(&self) -> usize {
        self.cached_len
    }

    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }
}

/// Global KV Cache Manager managing all transformer layers up to 256k context
pub struct KvCacheManager {
    pub layers: Vec<LayerKvCache>,
    pub max_context: usize,
    pub prefix_cache: PrefixCache,
}

impl KvCacheManager {
    pub fn new(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_context: usize,
        sliding_window_layers: &[usize],
        sliding_window_size: usize,
        layer_devices: &[crate::device::DeviceType],
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let is_swa = sliding_window_layers.contains(&l);
            let sw = if is_swa { Some(sliding_window_size) } else { None };

            let dev = layer_devices.get(l).cloned().unwrap_or(crate::device::DeviceType::Cpu);
            let on_device = matches!(dev, crate::device::DeviceType::Cuda(_));
            let dev_id = match dev {
                crate::device::DeviceType::Cuda(id) => id,
                crate::device::DeviceType::Cpu => 0,
            };

            let cache = LayerKvCache::new(n_kv_heads, head_dim, max_context, sw, on_device, dev_id)?;
            layers.push(cache);
        }

        Ok(Self {
            layers,
            max_context,
            prefix_cache: PrefixCache::new(),
        })
    }

    pub fn truncate(&mut self, len: usize) {
        for l in &mut self.layers {
            l.truncate(len);
        }
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
        self.prefix_cache.clear();
    }
}
