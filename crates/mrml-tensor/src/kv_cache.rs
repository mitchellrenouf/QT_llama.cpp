use crate::anyhow::{self, Result};
use crate::quant::{f16_to_f32, quantize_f32_to_q4_0, quantize_f32_to_q8_0,
    vec_dot_q4_0_q8_0, vec_dot_q8_0_q8_0};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvCacheFormat {
    Q4,
    Q8,
    #[default]
    F32,
}

impl KvCacheFormat {
    pub fn cuda_code(self) -> i32 {
        match self { Self::F32 => 0, Self::Q8 => 1, Self::Q4 => 2 }
    }

    pub fn cuda_bytes_per_token(self, elements: usize) -> usize {
        match self {
            Self::F32 => elements * 2,
            Self::Q8 => elements / 32 * 34,
            Self::Q4 => elements / 32 * 18,
        }
    }
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "q4" | "q4_0" => Ok(Self::Q4),
            "q8" | "q8_0" => Ok(Self::Q8),
            "f32" | "f16" => Ok(Self::F32),
            value => anyhow::bail!("unsupported CPU KV cache type '{value}' (use q4_0, q8_0, or f32)"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum KvCacheRow {
    Empty,
    F32(Vec<f32>),
    Q8(Vec<u8>),
    Q4(Vec<u8>),
}

impl KvCacheRow {
    pub fn from_f32(values: &[f32], format: KvCacheFormat) -> Self {
        assert_eq!(values.len() % 32, 0);
        match format {
            KvCacheFormat::F32 => Self::F32(values.to_vec()),
            KvCacheFormat::Q8 => {
                let mut data = vec![0; values.len() / 32 * 34];
                quantize_f32_to_q8_0(values, &mut data);
                Self::Q8(data)
            }
            KvCacheFormat::Q4 => {
                let mut data = vec![0; values.len() / 32 * 18];
                quantize_f32_to_q4_0(values, &mut data);
                Self::Q4(data)
            }
        }
    }

    pub fn dot_head(&self, q: &[f32], q8: &[u8], head_offset: usize) -> f32 {
        match self {
            Self::F32(row) => q.iter().zip(&row[head_offset..head_offset + q.len()]).map(|(a,b)| a*b).sum(),
            Self::Q8(row) => {
                let bytes = q.len() / 32 * 34;
                let offset = head_offset / 32 * 34;
                vec_dot_q8_0_q8_0(&row[offset..offset + bytes], q8, q.len())
            }
            Self::Q4(row) => {
                let bytes = q.len() / 32 * 18;
                let offset = head_offset / 32 * 18;
                vec_dot_q4_0_q8_0(&row[offset..offset + bytes], q8, q.len())
            }
            Self::Empty => 0.0,
        }
    }

    pub fn add_head_scaled(&self, out: &mut [f32], head_offset: usize, scale: f32) {
        let head_len = out.len();
        match self {
            Self::F32(row) => out.iter_mut().zip(&row[head_offset..head_offset + head_len])
                .for_each(|(dst, src)| *dst += scale * src),
            Self::Q8(row) => {
                let offset = head_offset / 32 * 34;
                for (block, dst) in out.chunks_exact_mut(32).enumerate() {
                    let base = offset + block * 34;
                    let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]])) * scale;
                    for i in 0..32 { dst[i] += d * row[base + 2 + i] as i8 as f32; }
                }
            }
            Self::Q4(row) => {
                let offset = head_offset / 32 * 18;
                for (block, dst) in out.chunks_exact_mut(32).enumerate() {
                    let base = offset + block * 18;
                    let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]])) * scale;
                    for i in 0..16 {
                        let packed = row[base + 2 + i];
                        dst[i] += d * ((packed & 15) as i8 - 8) as f32;
                        dst[i + 16] += d * ((packed >> 4) as i8 - 8) as f32;
                    }
                }
            }
            Self::Empty => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_rows_preserve_attention_operations() {
        let values: Vec<f32> = (0..128).map(|i| ((i * 29 % 97) as f32 - 48.0) / 23.0).collect();
        let query: Vec<f32> = (0..64).map(|i| ((i * 11 % 53) as f32 - 26.0) / 17.0).collect();
        let mut query_q8 = vec![0; query.len() / 32 * 34];
        quantize_f32_to_q8_0(&query, &mut query_q8);
        let exact_dot: f32 = values[64..].iter().zip(&query).map(|(a, b)| a * b).sum();

        for (format, tolerance) in [(KvCacheFormat::Q8, 0.03), (KvCacheFormat::Q4, 1.0)] {
            let row = KvCacheRow::from_f32(&values, format);
            let dot = row.dot_head(&query, &query_q8, 64);
            assert!((dot - exact_dot).abs() < tolerance, "{format:?}: {dot} vs {exact_dot}");
            let mut actual = vec![0.0; 64];
            row.add_head_scaled(&mut actual, 64, 0.37);
            let max_error = actual.iter().zip(&values[64..])
                .map(|(a, b)| (a - b * 0.37).abs()).fold(0.0f32, f32::max);
            assert!(max_error < tolerance, "{format:?} max error: {max_error}");
        }
    }

    #[test]
    fn cuda_layout_sizes_match_block_formats() {
        assert_eq!(KvCacheFormat::F32.cuda_bytes_per_token(2048), 4096);
        assert_eq!(KvCacheFormat::Q8.cuda_bytes_per_token(2048), 2176);
        assert_eq!(KvCacheFormat::Q4.cuda_bytes_per_token(2048), 1152);
    }
}

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
