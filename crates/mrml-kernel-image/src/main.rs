#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
#[cfg(feature = "production-policy")]
use mrml_kernel::BootPolicy;
use mrml_kernel::arch::x86_64::{HardwareTrapFrame, InterruptGate, install_exception_tables};
#[cfg(not(feature = "fault-probe"))]
use mrml_kernel::{
    BootHandoff, Color, EarlyKernelContext, FramebufferSurface, HANDOFF_HEADER_BYTES,
    HANDOFF_REGION_BYTES, MAX_HANDOFF_REGIONS, MemoryKind, MemoryRegion, PhysAddr,
};
#[cfg(all(not(feature = "fault-probe"), feature = "gpu-benchmark"))]
use mrml_kernel::{
    GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES, GpuGuestCommandPublisher, GpuQueueIdentity,
    GpuQueueSender, GpuResourceResponse, GpuResourceResponseReceiver, GpuSharedRingIndices,
    ResourceCommand,
};

#[cfg(not(feature = "fault-probe"))]
const MAX_HANDOFF_BYTES: usize = HANDOFF_HEADER_BYTES + MAX_HANDOFF_REGIONS * HANDOFF_REGION_BYTES;
#[cfg(feature = "gpu-benchmark")]
const GPU_COMMAND_BASE: usize = 0x00b0_0000;
#[cfg(feature = "gpu-benchmark")]
const GPU_COMPLETION_BASE: usize = 0x00b0_1000;
#[cfg(feature = "gpu-benchmark")]
const GPU_BENCHMARK_ELEMENTS: u32 = 1 << 22;
#[cfg(feature = "gpu-benchmark")]
const GPU_BENCHMARK_ITERATIONS: u32 = 1_000;
#[cfg(feature = "fault-probe")]
const EXCEPTION_PROBE_PORT: u16 = 0x4d53;

static GDT: [u64; 8] = [
    0,
    0,
    0,
    0,
    0,
    0,
    0x00cf_9300_0000_ffff,
    0x00af_9b00_0000_ffff,
];
static mut IDT: [InterruptGate; 256] = [InterruptGate::MISSING; 256];

global_asm!(
    r#"
    .section .text
    .global mrml_exception_fail_stop
mrml_exception_fail_stop:
    cli
1:
    hlt
    jmp 1b

    .macro MRML_NO_ERROR vector
    .global mrml_exception_\vector
mrml_exception_\vector:
    push 0
    push \vector
    jmp mrml_exception_common
    .endm

    .macro MRML_ERROR vector
    .global mrml_exception_\vector
mrml_exception_\vector:
    push \vector
    jmp mrml_exception_common
    .endm

    MRML_NO_ERROR 0
    MRML_NO_ERROR 1
    MRML_NO_ERROR 2
    MRML_NO_ERROR 3
    MRML_NO_ERROR 4
    MRML_NO_ERROR 5
    MRML_NO_ERROR 6
    MRML_NO_ERROR 7
    MRML_ERROR 8
    MRML_NO_ERROR 9
    MRML_ERROR 10
    MRML_ERROR 11
    MRML_ERROR 12
    MRML_ERROR 13
    MRML_ERROR 14
    MRML_NO_ERROR 15
    MRML_NO_ERROR 16
    MRML_ERROR 17
    MRML_NO_ERROR 18
    MRML_NO_ERROR 19
    MRML_NO_ERROR 20
    MRML_ERROR 21
    MRML_NO_ERROR 22
    MRML_NO_ERROR 23
    MRML_NO_ERROR 24
    MRML_NO_ERROR 25
    MRML_NO_ERROR 26
    MRML_NO_ERROR 27
    MRML_NO_ERROR 28
    MRML_ERROR 29
    MRML_ERROR 30
    MRML_NO_ERROR 31

mrml_exception_common:
    cld
    push rax
    push rcx
    push rdx
    push rbx
    push rbp
    push rsi
    push rdi
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov rdi, rsp
    and rsp, -16
    call mrml_exception_dispatch
    ud2

    .section .rdata
    .balign 8
    .global mrml_exception_table
mrml_exception_table:
    .quad mrml_exception_0, mrml_exception_1, mrml_exception_2, mrml_exception_3
    .quad mrml_exception_4, mrml_exception_5, mrml_exception_6, mrml_exception_7
    .quad mrml_exception_8, mrml_exception_9, mrml_exception_10, mrml_exception_11
    .quad mrml_exception_12, mrml_exception_13, mrml_exception_14, mrml_exception_15
    .quad mrml_exception_16, mrml_exception_17, mrml_exception_18, mrml_exception_19
    .quad mrml_exception_20, mrml_exception_21, mrml_exception_22, mrml_exception_23
    .quad mrml_exception_24, mrml_exception_25, mrml_exception_26, mrml_exception_27
    .quad mrml_exception_28, mrml_exception_29, mrml_exception_30, mrml_exception_31
    "#
);

unsafe extern "C" {
    fn mrml_exception_fail_stop() -> !;
    static mrml_exception_table: [u64; 32];
}

#[unsafe(no_mangle)]
unsafe extern "sysv64" fn mrml_exception_dispatch(frame: *const HardwareTrapFrame) -> ! {
    let frame = match unsafe { frame.as_ref() } {
        Some(frame) => frame,
        None => halt(),
    };
    let mut ss = 0u16;
    unsafe { asm!("mov {0:x}, ss", out(reg) ss, options(nomem, nostack, preserves_flags)) };
    let normalized = unsafe {
        frame.normalize(
            (frame as *const HardwareTrapFrame as u64)
                + core::mem::size_of::<HardwareTrapFrame>() as u64,
            u64::from(ss),
        )
    };
    let fault_address = if normalized.vector == 14 {
        let address: u64;
        unsafe { asm!("mov {}, cr2", out(reg) address, options(nomem, nostack, preserves_flags)) };
        Some(address)
    } else {
        None
    };
    if normalized.disposition(fault_address).is_err() {
        halt();
    }
    #[cfg(feature = "fault-probe")]
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") EXCEPTION_PROBE_PORT,
            in("eax") normalized.vector as u32,
            options(nomem, nostack)
        )
    };
    halt()
}

#[cfg(feature = "fault-probe")]
#[used]
static RELOCATION_ANCHOR: unsafe extern "C" fn() -> ! = mrml_exception_fail_stop;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    halt()
}

/// Standalone PE32+ kernel entry. The loader calls this only after firmware
/// services have exited and after authenticating both this image and handoff.
///
/// # Safety
///
/// `bytes` must address exactly `length` readable bytes that remain mapped for
/// this non-returning call. The loader must already have installed the final
/// W^X image mappings, a writable kernel stack, and a read-only handoff mapping.
#[unsafe(export_name = "efi_main")]
pub unsafe extern "efiapi" fn kernel_entry(bytes: *const u8, length: usize) -> usize {
    unsafe { install_descriptor_tables() };
    #[cfg(feature = "fault-probe")]
    unsafe {
        let _ = (bytes, length);
        // Keep one absolute pointer live so the minimal diagnostic PE exercises
        // the same DIR64 high-half relocation path as the full kernel.
        let _ = core::ptr::addr_of!(RELOCATION_ANCHOR).read_volatile();
        asm!("ud2", options(noreturn));
    }
    #[cfg(not(feature = "fault-probe"))]
    unsafe {
        run_kernel(bytes, length)
    }
}

#[cfg(not(feature = "fault-probe"))]
unsafe fn run_kernel(bytes: *const u8, length: usize) -> ! {
    if bytes.is_null() || !(HANDOFF_HEADER_BYTES..=MAX_HANDOFF_BYTES).contains(&length) {
        halt();
    }
    let encoded = unsafe { core::slice::from_raw_parts(bytes, length) };
    let placeholder = match MemoryRegion::new(
        match PhysAddr::new(0) {
            Ok(value) => value,
            Err(_) => halt(),
        },
        1,
        MemoryKind::Reserved,
    ) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    let mut regions = [placeholder; MAX_HANDOFF_REGIONS];
    let mut region_count = 0usize;
    let handoff = match BootHandoff::decode(encoded, |region| {
        if region_count < regions.len() {
            regions[region_count] = region;
            region_count += 1;
        }
    }) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    if region_count != handoff.region_count() {
        halt();
    }
    #[cfg(feature = "production-policy")]
    {
        let minimum_version = match embedded_minimum_version() {
            Some(value) => value,
            None => halt(),
        };
        if BootPolicy::production(minimum_version)
            .validate(handoff.evidence())
            .is_err()
        {
            halt();
        }
    }
    let framebuffer = handoff.framebuffer();
    let _early = match EarlyKernelContext::new(
        *handoff.evidence().entropy(),
        handoff.acpi_root(),
        framebuffer,
        &regions[..region_count],
    ) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    let framebuffer_bytes = unsafe {
        core::slice::from_raw_parts_mut(
            framebuffer.base().get() as *mut u8,
            framebuffer.byte_length() as usize,
        )
    };
    let mut surface = match FramebufferSurface::new(framebuffer, framebuffer_bytes) {
        Ok(value) => value,
        Err(_) => halt(),
    };
    if surface
        .fill_rectangle(
            0,
            0,
            framebuffer.width(),
            framebuffer.height(),
            Color {
                red: 0x0b,
                green: 0x3b,
                blue: 0x5a,
            },
        )
        .is_err()
    {
        halt();
    }
    let _ = surface.fill_rectangle(
        0,
        0,
        framebuffer.width().min(96),
        framebuffer.height().min(12),
        Color {
            red: 0xff,
            green: 0xc8,
            blue: 0x57,
        },
    );
    #[cfg(feature = "gpu-benchmark")]
    if run_gpu_benchmark(handoff.evidence().entropy()).is_err() {
        halt();
    }
    #[cfg(feature = "gpu-benchmark")]
    let _ = surface.fill_rectangle(
        0,
        0,
        framebuffer.width().min(96),
        framebuffer.height().min(12),
        Color {
            red: 0x46,
            green: 0xe0,
            blue: 0x78,
        },
    );
    #[cfg(feature = "whp-gpu-benchmark")]
    unsafe {
        asm!("out dx, eax", in("dx") GPU_DOORBELL_PORT, in("eax") 2u32, options(nomem, nostack))
    };
    halt()
}

#[cfg(all(not(feature = "fault-probe"), feature = "gpu-benchmark"))]
fn run_gpu_benchmark(entropy: &[u8; 32]) -> Result<(), ()> {
    let identity = GpuQueueIdentity::from_boot_entropy(entropy).map_err(|_| ())?;
    let mut sender = GpuQueueSender::new(identity.session(), identity.key()).map_err(|_| ())?;
    let mut command = [0u8; GPU_QUEUE_MESSAGE_BYTES];
    sender
        .encode(
            1,
            ResourceCommand::BenchmarkAdd {
                elements: GPU_BENCHMARK_ELEMENTS,
                iterations: GPU_BENCHMARK_ITERATIONS,
            },
            &mut command,
        )
        .map_err(|_| ())?;
    let command_indices = unsafe { &*(GPU_COMMAND_BASE as *const GpuSharedRingIndices) };
    let command_slots = unsafe {
        &mut *((GPU_COMMAND_BASE + core::mem::size_of::<GpuSharedRingIndices>())
            as *mut [[u8; GPU_QUEUE_MESSAGE_BYTES]; 1])
    };
    let mut publisher = GpuGuestCommandPublisher::<1>::new().map_err(|_| ())?;
    publisher
        .publish(command_indices, command_slots, &command)
        .map_err(|_| ())?;
    unsafe {
        asm!("out dx, eax", in("dx") GPU_DOORBELL_PORT, in("eax") 1u32, options(nomem, nostack))
    };

    let published = GPU_COMPLETION_BASE as *const u64;
    while unsafe { published.read_volatile() } != 1 {
        core::hint::spin_loop();
    }
    let slot = (GPU_COMPLETION_BASE + core::mem::size_of::<GpuSharedRingIndices>()) as *const u8;
    let mut completion = [0u8; GPU_QUEUE_MESSAGE_BYTES];
    for (index, byte) in completion.iter_mut().enumerate() {
        *byte = unsafe { slot.add(index).read_volatile() };
    }
    let mut receiver =
        GpuResourceResponseReceiver::new(identity.session(), identity.key()).map_err(|_| ())?;
    match receiver.decode(&completion).map_err(|_| ())? {
        GpuResourceResponse::BenchmarkComplete {
            request_id: 1,
            elapsed_ns,
        } if elapsed_ns != 0 => Ok(()),
        _ => Err(()),
    }
}

#[cfg(all(not(feature = "fault-probe"), feature = "production-policy"))]
fn embedded_minimum_version() -> Option<u64> {
    let bytes = option_env!("MRML_KERNEL_MIN_VERSION")?.as_bytes();
    if bytes.is_empty() || bytes.len() > 20 {
        return None;
    }
    let mut value = 0u64;
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add((byte - b'0') as u64)?;
    }
    (value != 0).then_some(value)
}

unsafe fn install_descriptor_tables() {
    let fallback = mrml_exception_fail_stop as *const () as usize as u64;
    let handlers = unsafe { &mrml_exception_table };
    if unsafe {
        install_exception_tables(
            core::ptr::addr_of!(GDT).cast::<u64>(),
            GDT.len(),
            core::ptr::addr_of_mut!(IDT).cast::<InterruptGate>(),
            256,
            handlers,
            fallback,
            0x38,
        )
    }
    .is_err()
    {
        halt();
    }
}

fn halt() -> ! {
    loop {
        unsafe { asm!("cli", "hlt", options(nomem, nostack)) };
    }
}
