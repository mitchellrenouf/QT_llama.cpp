use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatmulBackend {
    Q4F32,
    Int4Int8,
    BlackwellBlockScaled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoeBackend {
    IndexedGemv,
    GroupedGemm,
    DenseActiveExperts,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBackend {
    Causal,
    MultiBlock,
    Flash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceClass {
    PortableCuda,
    TensorCoreInt8,
    Blackwell,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionPlan {
    pub device_class: DeviceClass,
    pub matmul: MatmulBackend,
    pub moe: MoeBackend,
    pub attention: AttentionBackend,
    pub decode_graph: bool,
    pub prefill_chunk: usize,
    pub kv_page_tokens: usize,
}

impl ExecutionPlan {
    #[cfg(feature = "cuda")]
    pub fn for_device(info: crate::cuda::CudaDeviceInfo) -> Self {
        let cc = info.compute_capability();
        if info.is_blackwell() {
            Self {
                device_class: DeviceClass::Blackwell,
                // These are the currently executable fallbacks. The device
                // class is retained so validated SM100 kernels can replace
                // individual operations without changing model semantics.
                matmul: MatmulBackend::Q4F32,
                moe: MoeBackend::IndexedGemv,
                attention: AttentionBackend::Causal,
                decode_graph: false,
                prefill_chunk: 128,
                kv_page_tokens: 64,
            }
        } else if cc >= 80 {
            Self {
                device_class: DeviceClass::TensorCoreInt8,
                matmul: MatmulBackend::Q4F32,
                moe: MoeBackend::IndexedGemv,
                attention: AttentionBackend::Causal,
                decode_graph: false,
                prefill_chunk: 128,
                kv_page_tokens: 32,
            }
        } else {
            Self::portable()
        }
    }

    pub fn portable() -> Self {
        Self {
            device_class: DeviceClass::PortableCuda,
            matmul: MatmulBackend::Q4F32,
            moe: MoeBackend::IndexedGemv,
            attention: AttentionBackend::Causal,
            decode_graph: false,
            prefill_chunk: 64,
            kv_page_tokens: 16,
        }
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} matmul={:?} moe={:?} attention={:?} graph={} prefill={} kv-page={}",
            self.device_class, self.matmul, self.moe, self.attention,
            self.decode_graph, self.prefill_chunk, self.kv_page_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_plan_is_conservative() {
        let plan = ExecutionPlan::portable();
        assert_eq!(plan.matmul, MatmulBackend::Q4F32);
        assert!(!plan.decode_graph);
    }
}
