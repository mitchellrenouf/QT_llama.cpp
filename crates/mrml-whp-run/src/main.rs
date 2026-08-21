#![no_std]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(test, allow(dead_code))]

#[cfg(target_os = "windows")]
use mrml_crypto::{LAMPORT_PUBLIC_KEY_BYTES, Sha3_512};
use mrml_error::{Result, anyhow};
#[cfg(target_os = "windows")]
use mrml_kernel::{
    ArtifactKind, GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES, GpuCommandRing, GpuQueueIdentity,
    GpuQueueReceiver, GpuResourceResponse, GpuResourceResponseSender, GpuSharedQueueLayout,
    GpuVmmQueueBridge, ResourceCommand, SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact, TrustRoot,
    VerifiedGpuKernelBundle, VmBackend, VmExit, verify_gpu_kernel_bundle,
};
#[cfg(any(test, target_os = "windows"))]
use mrml_kernel::{
    FramebufferInfo, MemoryKind, MemoryRegion, PhysAddr, PixelFormat, encode_handoff,
};
#[cfg(target_os = "windows")]
use mrml_runtime::{Instant, Vector, mrml_println as println};
#[cfg(target_os = "windows")]
use mrml_whp::{PreparedWhpGuest, WhpLaunchLayout, WhpSystem};

#[cfg(target_os = "windows")]
const MAX_KERNEL_BUNDLE: usize = SIGNED_ARTIFACT_OVERHEAD_BYTES + 16 * 1024 * 1024;
#[cfg(any(test, target_os = "windows"))]
const FRAMEBUFFER: u64 = 0x00a0_0000;
#[cfg(target_os = "windows")]
const COMMAND_BASE: u64 = 0x00b0_0000;
#[cfg(target_os = "windows")]
const COMPLETION_BASE: u64 = 0x00b0_1000;
#[cfg(target_os = "windows")]
const SERVICE_FRAME_PORT: u16 = 0x4d59;
#[cfg(target_os = "windows")]
const SERVICE_PROBE_PORT: u16 = 0x4d58;
#[cfg(target_os = "windows")]
const SERVICE_CALL_PORT: u16 = 0x4d5a;
#[cfg(target_os = "windows")]
const TIMER_READY_PORT: u16 = 0x4d54;
#[cfg(target_os = "windows")]
const TIMER_TICK_PORT: u16 = 0x4d55;
#[cfg(target_os = "windows")]
const SERVICE_PHYSICAL: u64 = 0x0060_0000;
#[cfg(target_os = "windows")]
const SERVICE_VIRTUAL: u64 = 0x0000_0001_4000_0000;
#[cfg(target_os = "windows")]
const SERVICE_STACK_PHYSICAL: u64 = 0x0080_0000;
#[cfg(target_os = "windows")]
const SERVICE_STACK_VIRTUAL: u64 = 0x0070_0000;
#[cfg(target_os = "windows")]
const SERVICE_TABLE_PHYSICAL: u64 = 0x00c0_0000;
#[cfg(target_os = "windows")]
const SERVICE_B_PHYSICAL: u64 = 0x0090_0000;
#[cfg(target_os = "windows")]
const SERVICE_B_STACK_PHYSICAL: u64 = 0x00e0_0000;
#[cfg(target_os = "windows")]
const SERVICE_B_TABLE_PHYSICAL: u64 = 0x00d0_0000;

#[cfg(target_os = "windows")]
fn application_main() -> Result<()> {
    let total_started = Instant::now();
    let arguments = mrml_runtime::command_arguments();
    if arguments.len() == 3 && arguments[1] == "--export-cuda-bundle" {
        mrml_runtime::write_file(&arguments[2], mrml_tensor::cuda::embedded_cuda_bundle())?;
        println!("wrote exact embedded CUDA PTX bundle to {}", arguments[2]);
        return Ok(());
    }
    let service_mode = arguments.len() == 8 && arguments[4] == "service-probe";
    let timer_mode = arguments.len() == 5 && arguments[4] == "timer-probe";
    if arguments.len() != 7 && !service_mode && !timer_mode {
        return Err(anyhow!(
            "usage: mrml-whp-run KERNEL.signed RELEASE.public MINIMUM_VERSION CUDA.signed CUDA.public CUDA_MINIMUM_VERSION\n       mrml-whp-run KERNEL.signed RELEASE.public MINIMUM_VERSION timer-probe\n       mrml-whp-run KERNEL.signed RELEASE.public MINIMUM_VERSION service-probe SERVICE.signed SERVICE.public SERVICE_MINIMUM_VERSION\n       mrml-whp-run --export-cuda-bundle OUTPUT.ptx"
        ));
    }
    let minimum_version = arguments[3]
        .parse::<u64>()
        .ok()
        .filter(|version| *version != 0)
        .ok_or_else(|| anyhow!("minimum version must be a nonzero integer"))?;
    let bundle = mrml_runtime::read_file_bounded(&arguments[1], MAX_KERNEL_BUNDLE)?;
    let public = mrml_runtime::read_file_bounded(&arguments[2], LAMPORT_PUBLIC_KEY_BYTES)?;
    if public.len() != LAMPORT_PUBLIC_KEY_BYTES {
        return Err(anyhow!("invalid release public-key length"));
    }
    let verification_started = Instant::now();
    let root = TrustRoot::new(
        ArtifactKind::Kernel,
        Sha3_512::digest(&public),
        minimum_version,
    );
    let signed = SignedArtifact::decode(&bundle).map_err(|_| anyhow!("invalid signed bundle"))?;
    let executable = signed
        .verify_executable(&root, ArtifactKind::Kernel)
        .map_err(|_| anyhow!("kernel signature or PE policy rejected"))?;
    let service_bundle = if service_mode {
        Some(mrml_runtime::read_file_bounded(
            &arguments[5],
            SIGNED_ARTIFACT_OVERHEAD_BYTES + mrml_kernel::MAX_SERVICE_IMAGE_BYTES as usize,
        )?)
    } else {
        None
    };
    let service_public = if service_mode {
        let public = mrml_runtime::read_file_bounded(&arguments[6], LAMPORT_PUBLIC_KEY_BYTES)?;
        if public.len() != LAMPORT_PUBLIC_KEY_BYTES {
            return Err(anyhow!("invalid service public-key length"));
        }
        Some(public)
    } else {
        None
    };
    let service_executable = match (&service_bundle, &service_public) {
        (Some(bundle), Some(public)) => {
            let minimum = arguments[7]
                .parse::<u64>()
                .ok()
                .filter(|version| *version != 0)
                .ok_or_else(|| anyhow!("service minimum version must be nonzero"))?;
            let root = TrustRoot::new(
                ArtifactKind::ServiceImage,
                Sha3_512::digest(public),
                minimum,
            );
            let signed = SignedArtifact::decode(bundle)
                .map_err(|_| anyhow!("invalid signed service bundle"))?;
            Some(
                signed
                    .verify_executable(&root, ArtifactKind::ServiceImage)
                    .map_err(|_| anyhow!("service signature or PE policy rejected"))?,
            )
        }
        (None, None) => None,
        _ => return Err(anyhow!("incomplete service verification inputs")),
    };
    let cuda_bundle = if service_mode || timer_mode {
        None
    } else {
        Some(verify_cuda_bundle(
            &arguments[4],
            &arguments[5],
            &arguments[6],
        )?)
    };
    let verification_micros = verification_started.elapsed().as_micros();
    let mut entropy = [0u8; 32];
    mrml_runtime::fill_random(&mut entropy)
        .map_err(|_| anyhow!("operating-system boot entropy failed"))?;
    let handoff = boot_handoff(
        executable.artifact().version(),
        entropy,
        *executable.artifact().digest(),
    )?;
    let layout = WhpLaunchLayout::new(
        0x10_0000,
        32,
        0x20_0000,
        0xffff_8001_4000_0000,
        0x30_0000,
        0xffff_8001_5000_0000,
        0x40_0000,
        0xffff_8001_6000_0000,
        8,
        false,
    )
    .map_err(|_| anyhow!("invalid fixed WHP launch layout"))?;
    let queue = GpuSharedQueueLayout::new(COMMAND_BASE, COMPLETION_BASE, 1)
        .map_err(|_| anyhow!("invalid fixed GPU queue layout"))?;
    let system = WhpSystem::open()
        .map_err(|error| anyhow!("Windows Hypervisor Platform is unavailable: {:?}", error))?;
    let preparation_started = Instant::now();
    let mut guest = match (service_mode, timer_mode) {
        (true, false) => system.prepare_isolated_service_kernel(&executable, &handoff, layout),
        (false, true) => system.prepare_timer_kernel(&executable, &handoff, layout),
        (false, false) => system.prepare_kernel_gpu_guest(&executable, &handoff, layout, queue),
        (true, true) => return Err(anyhow!("launch mode is ambiguous")),
    }
    .map_err(|error| anyhow!("verified WHP kernel preparation failed: {:?}", error))?;
    if service_mode {
        guest = guest
            .attach_isolated_service(
                &executable,
                layout,
                service_executable
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing verified service executable"))?,
                SERVICE_PHYSICAL,
                SERVICE_VIRTUAL,
                SERVICE_STACK_PHYSICAL,
                SERVICE_STACK_VIRTUAL,
                2,
                SERVICE_TABLE_PHYSICAL,
                32,
            )
            .map_err(|error| anyhow!("isolated WHP service preparation failed: {:?}", error))?;
        guest = guest
            .attach_isolated_service_at(
                1,
                &executable,
                layout,
                service_executable
                    .as_ref()
                    .ok_or_else(|| anyhow!("missing verified service executable"))?,
                SERVICE_B_PHYSICAL,
                SERVICE_VIRTUAL,
                SERVICE_B_STACK_PHYSICAL,
                SERVICE_STACK_VIRTUAL,
                2,
                SERVICE_B_TABLE_PHYSICAL,
                32,
            )
            .map_err(|error| {
                anyhow!(
                    "second isolated WHP service preparation failed: {:?}",
                    error
                )
            })?;
        if guest.service_entry() != Some(0x0000_0001_4000_1000)
            || guest.service_page_table_root() != PhysAddr::new(SERVICE_TABLE_PHYSICAL).ok()
            || guest.service_entry_at(1) != Some(0x0000_0001_4000_1000)
            || guest.service_page_table_root_at(1) != PhysAddr::new(SERVICE_B_TABLE_PHYSICAL).ok()
        {
            return Err(anyhow!("isolated WHP service layout mismatch"));
        }
    }
    let preparation_micros = preparation_started.elapsed().as_micros();
    let execution_started = Instant::now();
    let exit = match VmBackend::run(&mut guest, 0) {
        Ok(exit) => exit,
        Err(error) => {
            let mut marker = [0u8; 4];
            let _ = VmBackend::read_guest(&guest, FRAMEBUFFER, &mut marker);
            return Err(anyhow!(
                "WHP execution failed: {:?}; framebuffer marker={:02x?}",
                error,
                marker
            ));
        }
    };
    if let VmExit::GuestMemoryFault { guest_address, .. } = exit {
        let walk = guest
            .page_walk(guest_address)
            .map_err(|_| anyhow!("failed to inspect WHP guest execute fault"))?;
        let physical = walk.physical_address(guest_address);
        let mut instruction = [0u8; 8];
        if let Some(address) = physical {
            let _ = VmBackend::read_guest(&guest, address, &mut instruction);
        }
        return Err(anyhow!(
            "WHP guest memory fault at {:#x}: page_entries={:x?} physical={:x?} bytes={:02x?}",
            guest_address,
            walk.entries(),
            physical,
            instruction
        ));
    }
    if timer_mode {
        if exit
            != (VmExit::Io {
                port: TIMER_READY_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!("WHP timer did not initialize: {:?}", exit));
        }
        let exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("WHP local APIC counter wait failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: TIMER_READY_PORT,
                size: 4,
                write: true,
                value: 2,
            })
        {
            return Err(anyhow!("WHP local APIC counter did not elapse: {:?}", exit));
        }
        let exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("WHP local APIC delivery failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: TIMER_TICK_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!("WHP scheduler timer tick mismatch: {:?}", exit));
        }
        println!(
            "verified local-APIC scheduler tick under WHP: verify={}us prepare={}us execute={}us total={}us",
            verification_micros,
            preparation_micros,
            execution_started.elapsed().as_micros(),
            total_started.elapsed().as_micros()
        );
        return Ok(());
    }
    if service_mode {
        if exit
            != (VmExit::Io {
                port: SERVICE_CALL_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!("isolated WHP receiver did not block: {:?}", exit));
        }
        for stage in [2u32, 3] {
            let exit = VmBackend::run(&mut guest, 0)
                .map_err(|error| anyhow!("WHP execution during service IPC failed: {:?}", error))?;
            if exit
                != (VmExit::Io {
                    port: SERVICE_CALL_PORT,
                    size: 4,
                    write: true,
                    value: stage,
                })
            {
                return Err(anyhow!(
                    "isolated WHP IPC stage {} mismatch: {:?}",
                    stage,
                    exit
                ));
            }
        }
        let exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("WHP execution after service IPC failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: SERVICE_FRAME_PORT,
                size: 4,
                write: true,
                value: 0x001b_2303,
            })
        {
            return Err(anyhow!("isolated WHP service frame mismatch: {:?}", exit));
        }
        let exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("WHP execution after service frame failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: SERVICE_PROBE_PORT,
                size: 4,
                write: true,
                value: 3,
            })
        {
            return Err(anyhow!("isolated WHP service proof mismatch: {:?}", exit));
        }
        println!(
            "verified independently signed service under WHP: verify={}us prepare={}us execute={}us total={}us",
            verification_micros,
            preparation_micros,
            execution_started.elapsed().as_micros(),
            total_started.elapsed().as_micros()
        );
        return Ok(());
    }
    service_gpu_benchmark(
        &mut guest,
        queue,
        entropy,
        cuda_bundle
            .as_ref()
            .ok_or_else(|| anyhow!("missing verified CUDA bundle"))?,
        exit,
    )?;
    let exit = VmBackend::run(&mut guest, 0)
        .map_err(|error| anyhow!("WHP execution after GPU completion failed: {:?}", error))?;
    if exit
        != (VmExit::Io {
            port: GPU_DOORBELL_PORT,
            size: 4,
            write: true,
            value: 2,
        })
    {
        return Err(anyhow!(
            "GPU benchmark kernel exited unexpectedly: {:?}",
            exit
        ));
    }
    let mut marker = [0u8; 4];
    VmBackend::read_guest(&guest, FRAMEBUFFER, &mut marker)
        .map_err(|_| anyhow!("kernel framebuffer is unreadable"))?;
    if marker != [0x78, 0xe0, 0x46, 0] {
        return Err(anyhow!(
            "kernel did not authenticate the GPU benchmark completion"
        ));
    }
    println!(
        "verified GPU benchmark kernel completed under WHP: verify={}us prepare={}us execute={}us total={}us",
        verification_micros,
        preparation_micros,
        execution_started.elapsed().as_micros(),
        total_started.elapsed().as_micros()
    );
    Ok(())
}

#[cfg(target_os = "windows")]
fn verify_cuda_bundle(
    bundle_path: &str,
    public_path: &str,
    minimum_version: &str,
) -> Result<VerifiedGpuKernelBundle> {
    let minimum_version = minimum_version
        .parse::<u64>()
        .ok()
        .filter(|version| *version != 0)
        .ok_or_else(|| anyhow!("CUDA minimum version must be a nonzero integer"))?;
    let bundle = mrml_runtime::read_file_bounded(bundle_path, MAX_KERNEL_BUNDLE)?;
    let public = mrml_runtime::read_file_bounded(public_path, LAMPORT_PUBLIC_KEY_BYTES)?;
    verify_gpu_kernel_bundle(
        &bundle,
        &public,
        minimum_version,
        mrml_tensor::cuda::embedded_cuda_bundle(),
    )
    .map_err(|_| anyhow!("CUDA bundle signature, kind, version, or embedded PTX rejected"))
}

#[cfg(target_os = "windows")]
fn service_gpu_benchmark(
    guest: &mut PreparedWhpGuest<'_>,
    layout: GpuSharedQueueLayout,
    entropy: [u8; 32],
    cuda_bundle: &VerifiedGpuKernelBundle,
    exit: VmExit,
) -> Result<()> {
    if exit
        != (VmExit::Io {
            port: GPU_DOORBELL_PORT,
            size: 4,
            write: true,
            value: 1,
        })
    {
        return Err(anyhow!(
            "guest issued a noncanonical GPU doorbell: {:?}",
            exit
        ));
    }
    let identity = GpuQueueIdentity::from_boot_entropy(&entropy)
        .map_err(|_| anyhow!("GPU queue identity derivation failed"))?;
    let mut bridge = GpuVmmQueueBridge::<1>::new(layout)
        .map_err(|_| anyhow!("GPU bridge initialization failed"))?;
    let mut commands = GpuCommandRing::<1>::new()
        .map_err(|_| anyhow!("GPU command queue initialization failed"))?;
    bridge
        .consume_command(guest, &mut commands)
        .map_err(|error| anyhow!("GPU command transfer failed: {:?}", error))?;
    let mut wire = [0u8; GPU_QUEUE_MESSAGE_BYTES];
    commands
        .dequeue(&mut wire)
        .map_err(|_| anyhow!("GPU command queue unexpectedly empty"))?;
    let mut receiver = GpuQueueReceiver::new(identity.session(), identity.key())
        .map_err(|_| anyhow!("GPU receiver initialization failed"))?;
    let (request_id, command) = receiver
        .decode(&wire)
        .map_err(|_| anyhow!("GPU command authentication failed"))?;
    let ResourceCommand::BenchmarkAdd {
        elements,
        iterations,
    } = command
    else {
        return Err(anyhow!("GPU doorbell carried a non-benchmark command"));
    };
    let elapsed_ns = execute_cuda_add_benchmark(cuda_bundle, elements, iterations)?;
    let mut response = [0u8; GPU_QUEUE_MESSAGE_BYTES];
    GpuResourceResponseSender::new(identity.session(), identity.key())
        .and_then(|mut sender| {
            sender.encode(
                GpuResourceResponse::BenchmarkComplete {
                    request_id,
                    elapsed_ns,
                },
                &mut response,
            )
        })
        .map_err(|_| anyhow!("GPU response encoding failed"))?;
    bridge
        .publish_completion(guest, &response)
        .map_err(|error| anyhow!("GPU completion publication failed: {:?}", error))?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn execute_cuda_add_benchmark(
    _: &VerifiedGpuKernelBundle,
    elements: u32,
    iterations: u32,
) -> Result<u64> {
    use mrml_tensor::cuda::{CudaBuffer, CudaDevice};

    let elements = usize::try_from(elements).map_err(|_| anyhow!("element count overflow"))?;
    let mut a = Vector::new();
    let mut b = Vector::new();
    let mut output = Vector::new();
    a.resize(elements, 1.25f32);
    b.resize(elements, 2.5f32);
    output.resize(elements, 0.0f32);
    let device = CudaDevice::new(0).map_err(|_| anyhow!("CUDA device initialization failed"))?;
    let d_a = CudaBuffer::from_host(&a).map_err(|_| anyhow!("CUDA input A upload failed"))?;
    let d_b = CudaBuffer::from_host(&b).map_err(|_| anyhow!("CUDA input B upload failed"))?;
    let mut d_output =
        CudaBuffer::alloc(elements).map_err(|_| anyhow!("CUDA output allocation failed"))?;
    for _ in 0..20 {
        device.add(&d_a, &d_b, &mut d_output);
    }
    device
        .sync()
        .map_err(|_| anyhow!("CUDA warmup synchronization failed"))?;
    let started = Instant::now();
    for _ in 0..iterations {
        device.add(&d_a, &d_b, &mut d_output);
    }
    device
        .sync()
        .map_err(|_| anyhow!("CUDA benchmark synchronization failed"))?;
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos())
        .ok()
        .filter(|value| *value != 0)
        .ok_or_else(|| anyhow!("GPU benchmark duration is invalid"))?;
    d_output
        .copy_to_host(&mut output)
        .map_err(|_| anyhow!("CUDA result download failed"))?;
    if output
        .iter()
        .any(|value| (*value - 3.75).abs() >= f32::EPSILON)
    {
        return Err(anyhow!("GPU benchmark produced an incorrect result"));
    }
    let seconds = elapsed_ns as f64 / 1e9;
    println!(
        "MRML_IN_VM_GPU_BENCH backend=whp elements={} iterations={} total_ns={} kernel_us={:.3} bandwidth_gbps={:.2}",
        elements,
        iterations,
        elapsed_ns,
        elapsed_ns as f64 / 1e3 / iterations as f64,
        elements as f64 * 12.0 * iterations as f64 / seconds / 1e9
    );
    Ok(elapsed_ns)
}

#[cfg(not(target_os = "windows"))]
fn application_main() -> Result<()> {
    Err(anyhow!("mrml-whp-run is available only on Windows hosts"))
}

#[cfg(any(test, target_os = "windows"))]
fn boot_handoff(version: u64, entropy: [u8; 32], measurement: [u8; 64]) -> Result<[u8; 240]> {
    let framebuffer = FramebufferInfo::new(
        FRAMEBUFFER,
        0x1000,
        16,
        16,
        16,
        PixelFormat::BlueGreenRedReserved,
    )
    .map_err(|_| anyhow!("invalid framebuffer description"))?;
    let region = |address, pages, kind| {
        let address = PhysAddr::new(address).map_err(|_| anyhow!("unaligned launch region"))?;
        MemoryRegion::new(address, pages, kind).map_err(|_| anyhow!("invalid launch region"))
    };
    let regions = [
        region(0x1000, 2, MemoryKind::Free)?,
        region(0x3000, 1, MemoryKind::Kernel)?,
        region(FRAMEBUFFER, 1, MemoryKind::Mmio)?,
    ];
    let mut encoded = [0u8; 240];
    let length = encode_handoff(
        version,
        entropy,
        measurement,
        true,
        false,
        false,
        0x9000,
        framebuffer,
        &regions,
        &mut encoded,
    )
    .map_err(|_| anyhow!("handoff construction failed"))?;
    if length != encoded.len() {
        return Err(anyhow!("unexpected handoff length"));
    }
    Ok(encoded)
}

mrml_runtime::mrml_entrypoint!(application_main);

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_kernel::BootHandoff;

    #[test]
    fn handoff_binds_entropy_and_measurement() {
        let encoded = boot_handoff(9, [0x35; 32], [0xa7; 64]).unwrap();
        let decoded = BootHandoff::decode(&encoded, |_| {}).unwrap();
        assert_eq!(decoded.evidence().image_version(), 9);
        assert_eq!(decoded.evidence().entropy(), &[0x35; 32]);
        assert_eq!(decoded.evidence().image_measurement(), &[0xa7; 64]);
    }
}
