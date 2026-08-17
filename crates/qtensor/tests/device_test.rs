use qtensor::device::{DeviceManager, DeviceType};

#[test]
fn test_device_manager_plan_layers() {
    let mgr = DeviceManager::new();
    let total_layers = 30;
    let estimated_bytes_per_layer = 450_000_000; // 450 MB

    let plan = mgr.plan_layers(total_layers, estimated_bytes_per_layer);
    assert_eq!(plan.len(), total_layers);

    // Verify all layers are assigned to a valid device
    for (i, dev) in plan.iter().enumerate() {
        match dev {
            DeviceType::Cuda(id) => println!("Layer {} assigned to CUDA GPU {}", i, id),
            DeviceType::Cpu => println!("Layer {} assigned to CPU Host", i),
        }
    }
}
