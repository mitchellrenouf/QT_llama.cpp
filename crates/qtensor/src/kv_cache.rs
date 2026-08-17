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

        let total_elements = cap * n_kv_heads * head_dim;

        #[cfg(feature = "cuda")]
        if on_device {
            let d_k = CudaBuffer::alloc_on(device_id, total_elements)?;
            let d_v = CudaBuffer::alloc_on(device_id, total_elements)?;
            return Ok(Self {
                is_sliding_window: is_swa,
                sliding_window: sw_size,
                max_capacity: cap,
                cur_seq_len: 0,
                d_k: Some(d_k),
                d_v: Some(d_v),
                host_k: Vec::new(),
                host_v: Vec::new(),
            });
        }

        Ok(Self {
            is_sliding_window: is_swa,
            sliding_window: sw_size,
            max_capacity: cap,
            cur_seq_len: 0,
            #[cfg(feature = "cuda")]
            d_k: None,
            #[cfg(feature = "cuda")]
            d_v: None,
            host_k: vec![0.0f32; total_elements],
            host_v: vec![0.0f32; total_elements],
        })
    }

    pub fn clear(&mut self) {
        self.cur_seq_len = 0;
    }

    pub fn step_increment(&mut self) {
        self.cur_seq_len += 1;
    }
}

/// Global KV Cache Manager managing all transformer layers up to 256k context
pub struct KvCacheManager {
    pub layers: Vec<LayerKvCache>,
    pub max_context: usize,
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
        })
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }
}
