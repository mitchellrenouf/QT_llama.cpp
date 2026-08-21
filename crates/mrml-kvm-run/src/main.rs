#![no_std]
#![cfg_attr(not(test), no_main)]

#[cfg(target_os = "linux")]
use mrml_crypto::{LAMPORT_PUBLIC_KEY_BYTES, Sha3_512};
use mrml_error::{Result, anyhow};
#[cfg(target_os = "linux")]
use mrml_kernel::{
    ArtifactKind, GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES, GpuCommandRing, GpuQueueIdentity,
    GpuQueueReceiver, GpuResourceResponse, GpuResourceResponseSender, GpuSharedQueueLayout,
    GpuVmmQueueBridge, ResourceCommand, SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact, TrustRoot,
    VerifiedGpuKernelBundle, VmBackend, VmExit, verify_gpu_kernel_bundle,
};
#[cfg(any(test, target_os = "linux"))]
use mrml_kernel::{
    FramebufferInfo, MemoryKind, MemoryRegion, PhysAddr, PixelFormat, encode_handoff,
};
#[cfg(target_os = "linux")]
use mrml_kvm::{KvmLaunchLayout, KvmSystem};
#[cfg(target_os = "linux")]
use mrml_runtime::Instant;
#[cfg(target_os = "linux")]
use mrml_runtime::mrml_println as println;

#[cfg(target_os = "linux")]
#[cfg_attr(test, allow(dead_code))]
const MAX_KERNEL_BUNDLE: usize = SIGNED_ARTIFACT_OVERHEAD_BYTES + 16 * 1024 * 1024;
#[cfg(any(test, target_os = "linux"))]
const FRAMEBUFFER: u64 = 0x00a0_0000;
#[cfg(target_os = "linux")]
const EXCEPTION_PROBE_PORT: u16 = 0x4d53;
#[cfg(target_os = "linux")]
const TIMER_READY_PORT: u16 = 0x4d54;
#[cfg(target_os = "linux")]
const TIMER_TICK_PORT: u16 = 0x4d55;
#[cfg(target_os = "linux")]
const PREEMPTION_PROBE_PORT: u16 = 0x4d5b;
#[cfg(target_os = "linux")]
const USER_PROBE_PORT: u16 = 0x4d56;
#[cfg(target_os = "linux")]
const USER_CALL_PROBE_PORT: u16 = 0x4d57;
#[cfg(target_os = "linux")]
const SERVICE_PROBE_PORT: u16 = 0x4d58;
#[cfg(target_os = "linux")]
const SERVICE_FRAME_PORT: u16 = 0x4d59;
#[cfg(target_os = "linux")]
const SERVICE_CALL_PORT: u16 = 0x4d5a;
#[cfg(target_os = "linux")]
const SERVICE_PHYSICAL: u64 = 0x0060_0000;
#[cfg(target_os = "linux")]
const SERVICE_VIRTUAL: u64 = 0x0000_0001_4000_0000;
#[cfg(target_os = "linux")]
const SERVICE_STACK_PHYSICAL: u64 = 0x0080_0000;
#[cfg(target_os = "linux")]
const SERVICE_STACK_VIRTUAL: u64 = 0x0070_0000;
#[cfg(target_os = "linux")]
const SERVICE_TABLE_PHYSICAL: u64 = 0x00c0_0000;
#[cfg(target_os = "linux")]
const SERVICE_B_PHYSICAL: u64 = 0x0090_0000;
#[cfg(target_os = "linux")]
const SERVICE_B_STACK_PHYSICAL: u64 = 0x00e0_0000;
#[cfg(target_os = "linux")]
const SERVICE_B_TABLE_PHYSICAL: u64 = 0x00d0_0000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(any(test, target_os = "linux"))]
enum LaunchMode {
    Boot,
    FaultProbe,
    TimerProbe,
    PreemptionProbe,
    UserProbe,
    ServiceProbe,
    GpuBenchmark,
}

#[cfg(any(test, target_os = "linux"))]
impl LaunchMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "boot" => Ok(Self::Boot),
            "fault-probe" => Ok(Self::FaultProbe),
            "timer-probe" => Ok(Self::TimerProbe),
            "preemption-probe" => Ok(Self::PreemptionProbe),
            "user-probe" => Ok(Self::UserProbe),
            "service-probe" => Ok(Self::ServiceProbe),
            "gpu-benchmark" => Ok(Self::GpuBenchmark),
            _ => Err(anyhow!(
                "mode must be boot, fault-probe, timer-probe, preemption-probe, user-probe, service-probe, or gpu-benchmark"
            )),
        }
    }
}

#[cfg(target_os = "linux")]
#[cfg_attr(test, allow(dead_code))]
fn application_main() -> Result<()> {
    let total_started = Instant::now();
    let arguments = mrml_runtime::command_arguments();
    if arguments.len() == 3 && arguments[1] == "--export-cuda-bundle" {
        mrml_runtime::write_file(&arguments[2], mrml_tensor::cuda::embedded_cuda_bundle())?;
        println!("wrote exact embedded CUDA PTX bundle to {}", arguments[2]);
        return Ok(());
    }
    if arguments.len() != 5 && arguments.len() != 8 {
        return Err(anyhow!(
            "usage: mrml-kvm-run KERNEL.signed RELEASE.public MINIMUM_VERSION MODE [CUDA.signed CUDA.public CUDA_MINIMUM_VERSION]\n       mrml-kvm-run --export-cuda-bundle OUTPUT.ptx"
        ));
    }
    let minimum_version = arguments[3]
        .parse::<u64>()
        .ok()
        .filter(|version| *version != 0)
        .ok_or_else(|| anyhow!("minimum version must be a nonzero integer"))?;
    let mode = LaunchMode::parse(&arguments[4])?;
    if matches!(mode, LaunchMode::GpuBenchmark | LaunchMode::ServiceProbe) != (arguments.len() == 8)
    {
        return Err(anyhow!(
            "gpu-benchmark and service-probe require exactly one matching signed artifact, public key, and minimum version"
        ));
    }
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
    let service_bundle = if mode == LaunchMode::ServiceProbe {
        Some(mrml_runtime::read_file_bounded(
            &arguments[5],
            SIGNED_ARTIFACT_OVERHEAD_BYTES + mrml_kernel::MAX_SERVICE_IMAGE_BYTES as usize,
        )?)
    } else {
        None
    };
    let service_public = if mode == LaunchMode::ServiceProbe {
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
    let cuda_bundle = if mode == LaunchMode::GpuBenchmark {
        Some(verify_cuda_bundle(
            &arguments[5],
            &arguments[6],
            &arguments[7],
        )?)
    } else {
        None
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
    let user_probe = matches!(mode, LaunchMode::UserProbe | LaunchMode::PreemptionProbe);
    let (image_virtual, handoff_virtual, stack_virtual) = if user_probe {
        (0x0040_0000, 0x0200_0000, 0x0300_0000)
    } else {
        (
            0xffff_8001_4000_0000,
            0xffff_8001_5000_0000,
            0xffff_8001_6000_0000,
        )
    };
    let layout = KvmLaunchLayout::new(
        0x10_0000,
        32,
        0x20_0000,
        image_virtual,
        0x30_0000,
        handoff_virtual,
        0x40_0000,
        stack_virtual,
        8,
        user_probe,
    )
    .map_err(|_| anyhow!("invalid fixed kernel launch layout"))?;
    let system = KvmSystem::open()
        .map_err(|error| anyhow!("KVM is unavailable or incompatible: {:?}", error))?;
    let preparation_started = Instant::now();
    let queue_layout = GpuSharedQueueLayout::new(0x00b0_0000, 0x00b0_1000, 1)
        .map_err(|_| anyhow!("invalid fixed GPU queue layout"))?;
    let mut guest = match mode {
        LaunchMode::GpuBenchmark => {
            system.prepare_kernel_gpu_guest::<13>(0, &executable, &handoff, layout, queue_layout)
        }
        LaunchMode::TimerProbe | LaunchMode::PreemptionProbe => {
            system.prepare_timer_kernel_guest::<13>(0, &executable, &handoff, layout)
        }
        _ => system.prepare_kernel_guest::<13>(0, &executable, &handoff, layout),
    }
    .map_err(|error| anyhow!("verified kernel launch preparation failed: {:?}", error))?;
    if mode == LaunchMode::ServiceProbe {
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
            .map_err(|error| anyhow!("isolated service preparation failed: {:?}", error))?;
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
            .map_err(|error| anyhow!("second isolated service preparation failed: {:?}", error))?;
        if guest.service_entry() != Some(0x0000_0001_4000_1000)
            || guest.service_page_table_root() != PhysAddr::new(SERVICE_TABLE_PHYSICAL).ok()
            || guest.service_entry_at(1) != Some(0x0000_0001_4000_1000)
            || guest.service_page_table_root_at(1) != PhysAddr::new(SERVICE_B_TABLE_PHYSICAL).ok()
        {
            return Err(anyhow!(
                "isolated service layout did not match the signed probe ABI"
            ));
        }
    }
    let preparation_micros = preparation_started.elapsed().as_micros();
    let execution_started = Instant::now();
    let mut exit = VmBackend::run(&mut guest, 0)
        .map_err(|error| anyhow!("KVM execution failed: {:?}", error))?;
    if mode == LaunchMode::GpuBenchmark {
        service_gpu_benchmark(
            &mut guest,
            queue_layout,
            handoff_entropy(&handoff)?,
            cuda_bundle
                .as_ref()
                .ok_or_else(|| anyhow!("missing verified CUDA bundle"))?,
            exit,
        )?;
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after GPU completion failed: {:?}", error))?;
    } else if mode == LaunchMode::FaultProbe {
        if exit
            != (VmExit::Io {
                port: EXCEPTION_PROBE_PORT,
                size: 4,
                write: true,
                value: 6,
            })
        {
            return Err(anyhow!(
                "fault probe did not reach the checked invalid-opcode dispatcher: {:?}",
                exit
            ));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after exception probe failed: {:?}", error))?;
    } else if mode == LaunchMode::PreemptionProbe {
        if exit
            != (VmExit::Io {
                port: TIMER_READY_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!("preemption probe did not enter CPL3: {:?}", exit));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM timer frame capture failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: PREEMPTION_PROBE_PORT,
                size: 4,
                write: true,
                value: 0x2320,
            })
        {
            return Err(anyhow!("timer frame was not vector 32 at CPL3: {:?}", exit));
        }
        println!("validated CPL3 timer frame");
        for stage in 1u32..=6 {
            exit = VmBackend::run(&mut guest, 0)
                .map_err(|error| anyhow!("KVM preemption stage {} failed: {:?}", stage, error))?;
            if exit
                != (VmExit::Io {
                    port: PREEMPTION_PROBE_PORT,
                    size: 4,
                    write: true,
                    value: stage,
                })
            {
                let state = guest
                    .snapshot()
                    .map_err(|error| anyhow!("preemption failure snapshot failed: {:?}", error))?;
                return Err(anyhow!(
                    "timer preemption stage {} mismatch: {:?}, rip={:#x} rsp={:#x} cr2={:#x} cr3={:#x}",
                    stage,
                    exit,
                    state.instruction_pointer(),
                    state.stack_pointer(),
                    state.fault_address(),
                    state.page_table_root()
                ));
            }
            println!("validated preemption stage {}", stage);
        }
        exit = VmExit::Halted;
    } else if mode == LaunchMode::TimerProbe {
        if exit
            != (VmExit::Io {
                port: TIMER_READY_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!(
                "timer probe did not reach its interruptible wait: {:?}",
                exit
            ));
        }
        exit = VmBackend::run(&mut guest, 0).map_err(|error| {
            anyhow!(
                "KVM execution waiting for local counter failed: {:?}",
                error
            )
        })?;
        if exit
            != (VmExit::Io {
                port: TIMER_READY_PORT,
                size: 4,
                write: true,
                value: 2,
            })
        {
            return Err(anyhow!("local APIC counter did not elapse: {:?}", exit));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution delivering local timer failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: TIMER_TICK_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!(
                "timer probe did not execute one scheduler tick: {:?}",
                exit
            ));
        }
        // The authenticated timer output is the terminal proof. A KVM guest
        // with an in-kernel interrupt controller can remain blocked in HLT,
        // so do not require a subsequent host-visible halt exit.
        exit = VmExit::Halted;
    } else if mode == LaunchMode::UserProbe {
        if exit
            != (VmExit::Io {
                port: USER_CALL_PROBE_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!(
                "user probe did not complete capability IPC through vector 0x80: {:?}",
                exit
            ));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after user call failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: USER_PROBE_PORT,
                size: 4,
                write: true,
                value: 3,
            })
        {
            return Err(anyhow!(
                "user probe did not return through the checked ring-three fault path: {:?}",
                exit
            ));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after user proof failed: {:?}", error))?;
    } else if mode == LaunchMode::ServiceProbe {
        if exit
            != (VmExit::Io {
                port: SERVICE_CALL_PORT,
                size: 4,
                write: true,
                value: 1,
            })
        {
            return Err(anyhow!("isolated receiver did not block: {:?}", exit));
        }
        for stage in [2u32, 3] {
            exit = VmBackend::run(&mut guest, 0)
                .map_err(|error| anyhow!("KVM execution during service IPC failed: {:?}", error))?;
            if exit
                != (VmExit::Io {
                    port: SERVICE_CALL_PORT,
                    size: 4,
                    write: true,
                    value: stage,
                })
            {
                return Err(anyhow!(
                    "isolated service IPC stage {} failed: {:?}",
                    stage,
                    exit
                ));
            }
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after service IPC failed: {:?}", error))?;
        if exit
            != (VmExit::Io {
                port: SERVICE_FRAME_PORT,
                size: 4,
                write: true,
                value: 0x001b_2303,
            })
        {
            return Err(anyhow!(
                "isolated service produced a malformed privilege frame: {:?}",
                exit
            ));
        }
        exit = VmBackend::run(&mut guest, 0).map_err(|error| {
            anyhow!(
                "KVM execution after service frame proof failed: {:?}",
                error
            )
        })?;
        if exit
            != (VmExit::Io {
                port: SERVICE_PROBE_PORT,
                size: 4,
                write: true,
                value: 3,
            })
        {
            let state = guest
                .snapshot()
                .map_err(|error| anyhow!("service failure snapshot failed: {:?}", error))?;
            return Err(anyhow!(
                "isolated signed service did not return through its CPL3 breakpoint: {:?}, rip={:#x} cs={:#x} rsp={:#x} cr2={:#x} cr3={:#x}",
                exit,
                state.instruction_pointer(),
                state.code_selector(),
                state.stack_pointer(),
                state.fault_address(),
                state.page_table_root()
            ));
        }
        exit = VmBackend::run(&mut guest, 0)
            .map_err(|error| anyhow!("KVM execution after service proof failed: {:?}", error))?;
    }
    let execution_micros = execution_started.elapsed().as_micros();
    if exit != VmExit::Halted {
        let state = guest.snapshot().map_err(|error| {
            anyhow!("kernel exit {:?}; state capture failed: {:?}", exit, error)
        })?;
        let walk = guest.page_walk(state.fault_address()).map_err(|error| {
            anyhow!(
                "kernel exit {:?}; page walk for {:#x} failed: {:?}",
                exit,
                state.fault_address(),
                error
            )
        })?;
        let physical = walk
            .physical_address(state.fault_address())
            .ok_or_else(|| anyhow!("fault address has no 4 KiB physical translation"))?;
        let mut fault_bytes = [0u8; 8];
        VmBackend::read_guest(&guest, physical, &mut fault_bytes)
            .map_err(|_| anyhow!("translated fault bytes are unreadable"))?;
        let invalid_opcode_gate = state
            .idt_base()
            .checked_add(6 * 16)
            .ok_or_else(|| anyhow!("invalid-opcode gate address overflow"))?;
        let gate_walk = guest
            .page_walk(invalid_opcode_gate)
            .map_err(|_| anyhow!("invalid-opcode gate page walk failed"))?;
        let gate_physical = gate_walk
            .physical_address(invalid_opcode_gate)
            .ok_or_else(|| anyhow!("invalid-opcode gate is not mapped"))?;
        let mut gate = [0u8; 16];
        VmBackend::read_guest(&guest, gate_physical, &mut gate)
            .map_err(|_| anyhow!("invalid-opcode gate bytes are unreadable"))?;
        return Err(anyhow!(
            "kernel exit {:?}: rip={:#x} rsp={:#x} rflags={:#x} cr2={:#x}->{:#x} bytes={:02x?} cr3={:#x} cs={:#x} gdt={:#x}/{:#x} idt={:#x}/{:#x} ud_gate={:02x?} walk={:x?}",
            exit,
            state.instruction_pointer(),
            state.stack_pointer(),
            state.flags(),
            state.fault_address(),
            physical,
            fault_bytes,
            state.page_table_root(),
            state.code_selector(),
            state.gdt_base(),
            state.gdt_limit(),
            state.idt_base(),
            state.idt_limit(),
            gate,
            walk.entries()
        ));
    }
    let mut marker = [0u8; 4];
    VmBackend::read_guest(&guest, FRAMEBUFFER, &mut marker)
        .map_err(|_| anyhow!("kernel framebuffer is unreadable"))?;
    match mode {
        LaunchMode::Boot if marker != [0x57, 0xc8, 0xff, 0] => {
            return Err(anyhow!(
                "kernel did not paint its authenticated boot marker"
            ));
        }
        LaunchMode::FaultProbe if marker != [0; 4] => {
            return Err(anyhow!("fault probe unexpectedly modified the framebuffer"));
        }
        LaunchMode::TimerProbe if marker != [0; 4] => {
            return Err(anyhow!("timer probe unexpectedly modified the framebuffer"));
        }
        LaunchMode::UserProbe if marker != [0; 4] => {
            return Err(anyhow!("user probe unexpectedly modified the framebuffer"));
        }
        LaunchMode::ServiceProbe if marker != [0; 4] => {
            return Err(anyhow!(
                "service probe unexpectedly modified the framebuffer"
            ));
        }
        LaunchMode::GpuBenchmark if marker != [0x78, 0xe0, 0x46, 0] => {
            return Err(anyhow!(
                "kernel did not authenticate the GPU benchmark completion"
            ));
        }
        _ => {}
    }
    println!(
        "verified {:?} kernel reached its expected halt under KVM: verify={}us prepare={}us execute={}us total={}us",
        mode,
        verification_micros,
        preparation_micros,
        execution_micros,
        total_started.elapsed().as_micros()
    );
    Ok(())
}

#[cfg(target_os = "linux")]
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

#[cfg(target_os = "linux")]
fn handoff_entropy(handoff: &[u8]) -> Result<[u8; 32]> {
    let decoded = mrml_kernel::BootHandoff::decode(handoff, |_| {})
        .map_err(|_| anyhow!("invalid benchmark handoff"))?;
    Ok(*decoded.evidence().entropy())
}

#[cfg(target_os = "linux")]
fn service_gpu_benchmark<const N: usize>(
    guest: &mut mrml_kvm::PreparedKvmGuest<N>,
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
            "GPU benchmark guest did not issue the canonical doorbell: {:?}",
            exit
        ));
    }
    let identity = GpuQueueIdentity::from_boot_entropy(&entropy)
        .map_err(|_| anyhow!("GPU queue identity derivation failed"))?;
    let mut bridge = GpuVmmQueueBridge::<1>::new(layout)
        .map_err(|_| anyhow!("GPU queue bridge initialization failed"))?;
    let mut owned = GpuCommandRing::<1>::new()
        .map_err(|_| anyhow!("GPU command queue initialization failed"))?;
    bridge
        .consume_command(guest, &mut owned)
        .map_err(|error| anyhow!("GPU command transfer failed: {:?}", error))?;
    let mut wire = [0u8; GPU_QUEUE_MESSAGE_BYTES];
    owned
        .dequeue(&mut wire)
        .map_err(|_| anyhow!("GPU command queue unexpectedly empty"))?;
    let mut receiver = GpuQueueReceiver::new(identity.session(), identity.key())
        .map_err(|_| anyhow!("GPU command receiver initialization failed"))?;
    let (request_id, command) = receiver
        .decode(&wire)
        .map_err(|_| anyhow!("GPU benchmark command authentication failed"))?;
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
        .map_err(|_| anyhow!("GPU benchmark response encoding failed"))?;
    bridge
        .publish_completion(guest, &response)
        .map_err(|error| anyhow!("GPU completion publication failed: {:?}", error))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn execute_cuda_add_benchmark(
    _: &VerifiedGpuKernelBundle,
    elements: u32,
    iterations: u32,
) -> Result<u64> {
    use mrml_runtime::Vector;
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
    let elapsed = started.elapsed().as_nanos();
    let elapsed_ns = u64::try_from(elapsed)
        .ok()
        .filter(|value| *value != 0)
        .ok_or_else(|| anyhow!("GPU benchmark duration is invalid"))?;
    d_output
        .copy_to_host(&mut output)
        .map_err(|_| anyhow!("CUDA benchmark result download failed"))?;
    if output
        .iter()
        .any(|value| (*value - 3.75).abs() >= f32::EPSILON)
    {
        return Err(anyhow!("GPU benchmark produced an incorrect result"));
    }
    let seconds = elapsed_ns as f64 / 1e9;
    println!(
        "MRML_IN_VM_GPU_BENCH elements={} iterations={} total_ns={} kernel_us={:.3} bandwidth_gbps={:.2}",
        elements,
        iterations,
        elapsed_ns,
        elapsed_ns as f64 / 1e3 / iterations as f64,
        elements as f64 * 12.0 * iterations as f64 / seconds / 1e9
    );
    Ok(elapsed_ns)
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn application_main() -> Result<()> {
    Err(anyhow!("mrml-kvm-run is available only on Linux hosts"))
}

#[cfg(any(test, target_os = "linux"))]
fn boot_handoff(version: u64, entropy: [u8; 32], measurement: [u8; 64]) -> Result<[u8; 240]> {
    let framebuffer = FramebufferInfo::new(
        FRAMEBUFFER,
        0x1000,
        16,
        16,
        16,
        PixelFormat::BlueGreenRedReserved,
    )
    .map_err(|_| anyhow!("invalid fixed framebuffer description"))?;
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
    .map_err(|_| anyhow!("canonical boot handoff construction failed"))?;
    if length != encoded.len() {
        return Err(anyhow!("unexpected canonical boot handoff length"));
    }
    Ok(encoded)
}

mrml_runtime::mrml_entrypoint!(application_main);

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_kernel::BootHandoff;

    #[test]
    fn handoff_binds_verified_version_entropy_and_measurement() {
        let entropy = [0x35; 32];
        let measurement = [0xa7; 64];
        let encoded = boot_handoff(19, entropy, measurement).unwrap();
        let decoded = BootHandoff::decode(&encoded, |_| {}).unwrap();
        assert_eq!(decoded.evidence().image_version(), 19);
        assert_eq!(decoded.evidence().entropy(), &entropy);
        assert_eq!(decoded.evidence().image_measurement(), &measurement);
        assert!(decoded.evidence().secure_boot());
        assert!(!decoded.evidence().measured_boot());
        assert!(!decoded.evidence().rollback_protected());
    }

    #[test]
    fn launch_mode_is_explicit_and_fail_closed() {
        assert_eq!(LaunchMode::parse("boot").unwrap(), LaunchMode::Boot);
        assert_eq!(
            LaunchMode::parse("fault-probe").unwrap(),
            LaunchMode::FaultProbe
        );
        assert_eq!(
            LaunchMode::parse("timer-probe").unwrap(),
            LaunchMode::TimerProbe
        );
        assert_eq!(
            LaunchMode::parse("user-probe").unwrap(),
            LaunchMode::UserProbe
        );
        assert_eq!(
            LaunchMode::parse("service-probe").unwrap(),
            LaunchMode::ServiceProbe
        );
        assert!(LaunchMode::parse("diagnostic").is_err());
        assert_eq!(
            LaunchMode::parse("gpu-benchmark").unwrap(),
            LaunchMode::GpuBenchmark
        );
    }
}
