use qtensor::device::DeviceType;
use qtensor::kv_cache::KvCacheManager;

#[test]
fn test_256k_kv_cache_initialization() {
    let num_layers = 30;
    let n_kv_heads = 8;
    let head_dim = 256;
    let max_context = 256000;
    let sw_size = 4096;

    let sw_layers: Vec<usize> = (0..num_layers).filter(|i| i % 2 == 0).collect();
    let layer_devices = vec![DeviceType::Cpu; num_layers];

    let mut cache_mgr = KvCacheManager::new(
        num_layers,
        n_kv_heads,
        head_dim,
        max_context,
        &sw_layers,
        sw_size,
        &layer_devices,
    ).expect("Failed to initialize 256k KV cache manager");

    assert_eq!(cache_mgr.layers.len(), num_layers);

    // SWA layer has capacity limited to sliding_window
    assert_eq!(cache_mgr.layers[0].is_sliding_window, true);
    assert_eq!(cache_mgr.layers[0].max_capacity, sw_size);

    // Global attention layer has full 256k capacity
    assert_eq!(cache_mgr.layers[1].is_sliding_window, false);
    assert_eq!(cache_mgr.layers[1].max_capacity, max_context);

    // Step increment
    cache_mgr.layers[0].step_increment();
    assert_eq!(cache_mgr.layers[0].cur_seq_len, 1);

    cache_mgr.clear();
    assert_eq!(cache_mgr.layers[0].cur_seq_len, 0);
}
