#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice;
use mrml_runtime::Vector;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cuda(i32),
    Cpu,
}

#[derive(Clone)]
pub struct DeviceManager {
    pub devices: Vector<DeviceType>,
    pub primary_gpu: Option<i32>,
}

impl DeviceManager {
    pub fn new() -> Self {
        let mut devices = Vector::new();
        #[allow(unused_mut)]
        let mut primary_gpu = None;

        #[cfg(feature = "cuda")]
        {
            let count = CudaDevice::count();
            if count > 0 {
                primary_gpu = Some(0);
                for i in 0..count {
                    devices.push(DeviceType::Cuda(i as i32));
                }
            }
        }

        devices.push(DeviceType::Cpu);

        Self {
            devices,
            primary_gpu,
        }
    }

    /// Plan layer assignment across available devices based on free VRAM
    #[allow(unused_variables)]
    pub fn plan_layers(
        &self,
        total_layers: usize,
        estimated_layer_bytes: usize,
    ) -> Vector<DeviceType> {
        let mut assignments = Vector::with_capacity(total_layers).expect("MRML allocation failed");

        #[cfg(feature = "cuda")]
        {
            if let Some(gpu0) = self.primary_gpu {
                if let Ok((free, _total)) = CudaDevice::get_memory_info(gpu0) {
                    // Reserve 1.5 GB for activations and KV cache
                    let usable = free.saturating_sub(1_500_000_000);
                    let max_on_gpu0 = usable / estimated_layer_bytes.max(1);
                    let gpu0_layers = max_on_gpu0.min(total_layers);

                    for _ in 0..gpu0_layers {
                        assignments.push(DeviceType::Cuda(gpu0));
                    }

                    // Check secondary GPUs if needed
                    let count = CudaDevice::count();
                    let mut cur_layer = gpu0_layers;

                    for g in 1..count {
                        if cur_layer >= total_layers {
                            break;
                        }
                        if let Ok((g_free, _)) = CudaDevice::get_memory_info(g as i32) {
                            let g_usable = g_free.saturating_sub(1_000_000_000);
                            let g_layers = (g_usable / estimated_layer_bytes.max(1))
                                .min(total_layers - cur_layer);
                            for _ in 0..g_layers {
                                assignments.push(DeviceType::Cuda(g as i32));
                            }
                            cur_layer += g_layers;
                        }
                    }

                    // Place any remaining layers on CPU
                    while assignments.len() < total_layers {
                        assignments.push(DeviceType::Cpu);
                    }

                    return assignments;
                }
            }
        }

        // Fallback to CPU if CUDA is not enabled or no GPUs
        for _ in 0..total_layers {
            assignments.push(DeviceType::Cpu);
        }

        assignments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_fallback_plans_every_layer() {
        let manager = DeviceManager::new();
        assert!(manager.devices.contains(&DeviceType::Cpu));

        let plan = manager.plan_layers(7, 1024);
        assert_eq!(plan.len(), 7);
        #[cfg(not(feature = "cuda"))]
        assert!(plan.iter().all(|device| *device == DeviceType::Cpu));
    }
}
