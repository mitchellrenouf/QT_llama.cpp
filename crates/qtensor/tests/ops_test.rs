use qtensor::ops::*;

#[test]
fn test_rms_norm() {
    let x = vec![2.0f32, 2.0, 2.0, 2.0];
    let mut out = vec![0.0f32; 4];
    rms_norm(&x, None, 1e-6, &mut out);

    for v in out {
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_swiglu() {
    let gate = vec![0.0f32, 2.0, -2.0];
    let up = vec![1.0f32, 3.0, 4.0];
    let mut out = vec![0.0f32; 3];

    swiglu(&gate, &up, &mut out);

    assert_eq!(out[0], 0.0);
    assert!((out[1] - 5.28478).abs() < 1e-3);
}

#[test]
fn test_mat_mul_f32() {
    let a = vec![
        1.0f32, 2.0,
        3.0, 4.0,
    ];
    let b = vec![
        5.0f32, 6.0,
        7.0, 8.0,
    ];
    let mut c = vec![0.0f32; 4];

    mat_mul_f32(&a, &b, &mut c, 2, 2, 2);

    assert_eq!(c[0], 1.0 * 5.0 + 2.0 * 7.0);
    assert_eq!(c[1], 1.0 * 6.0 + 2.0 * 8.0);
    assert_eq!(c[2], 3.0 * 5.0 + 4.0 * 7.0);
    assert_eq!(c[3], 3.0 * 6.0 + 4.0 * 8.0);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_ops() {
    use qtensor::cuda::{CudaBuffer, CudaDevice};

    if !CudaDevice::is_available() {
        println!("Skipping CUDA tests: No CUDA device found");
        return;
    }

    let dev = CudaDevice::new().expect("Failed to create CudaDevice");

    let x_host = vec![2.0f32; 128];
    let d_x = CudaBuffer::from_host(&x_host).unwrap();
    let mut d_out = CudaBuffer::alloc(128).unwrap();

    dev.rms_norm(&d_x, None, &mut d_out, 1e-6);
    dev.sync().unwrap();

    let mut out_host = vec![0.0f32; 128];
    d_out.copy_to_host(&mut out_host).unwrap();

    for val in out_host {
        assert!((val - 1.0).abs() < 1e-4);
    }
}
