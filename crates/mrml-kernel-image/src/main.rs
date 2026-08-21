#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
#[cfg(feature = "production-policy")]
use mrml_kernel::BootPolicy;
#[cfg(feature = "timer-probe")]
use mrml_kernel::KernelScheduler;
#[cfg(any(feature = "user-probe", feature = "service-probe"))]
use mrml_kernel::SyscallRequest;
use mrml_kernel::UserCallFrame;
#[cfg(feature = "user-probe")]
use mrml_kernel::arch::x86_64::enter_user_context;
#[cfg(feature = "service-probe")]
use mrml_kernel::arch::x86_64::enter_user_context_on_stack;
#[cfg(feature = "timer-probe")]
use mrml_kernel::arch::x86_64::install_external_interrupt_gate;
#[cfg(any(feature = "user-probe", feature = "service-probe"))]
use mrml_kernel::arch::x86_64::install_user_call_gate;
use mrml_kernel::arch::x86_64::{
    AlignedTaskState, HardwareTrapFrame, InterruptGate, TaskStateSegment, install_exception_tables,
    load_task_register, write_task_state_descriptor,
};
#[cfg(any(feature = "user-probe", feature = "service-probe"))]
use mrml_kernel::arch::x86_64::{TrapDisposition, UserContext};
#[cfg(not(feature = "fault-probe"))]
use mrml_kernel::{
    BootHandoff, HANDOFF_HEADER_BYTES, HANDOFF_REGION_BYTES, MAX_HANDOFF_REGIONS, MemoryKind,
    MemoryRegion, PhysAddr,
};
#[cfg(all(
    not(feature = "fault-probe"),
    not(feature = "timer-probe"),
    not(feature = "user-probe"),
    not(feature = "service-probe")
))]
use mrml_kernel::{Color, EarlyKernelContext, FramebufferSurface};
#[cfg(feature = "user-probe")]
use mrml_kernel::{Endpoint, ObjectId, Rights, TaskRuntime};
#[cfg(all(not(feature = "fault-probe"), feature = "gpu-benchmark"))]
use mrml_kernel::{
    GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES, GpuGuestCommandPublisher, GpuQueueIdentity,
    GpuQueueSender, GpuResourceResponse, GpuResourceResponseReceiver, GpuSharedRingIndices,
    ResourceCommand,
};
#[cfg(any(feature = "timer-probe", feature = "user-probe"))]
use mrml_kernel::{Priority, ScheduleOutcome};

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
#[cfg(feature = "timer-probe")]
const TIMER_READY_PORT: u16 = 0x4d54;
#[cfg(feature = "timer-probe")]
const TIMER_TICK_PORT: u16 = 0x4d55;
#[cfg(feature = "timer-probe")]
const TIMER_VECTOR: u8 = 32;
#[cfg(feature = "user-probe")]
const USER_PROBE_PORT: u16 = 0x4d56;
#[cfg(feature = "user-probe")]
const USER_CALL_PROBE_PORT: u16 = 0x4d57;
#[cfg(feature = "service-probe")]
const SERVICE_PROBE_PORT: u16 = 0x4d58;
#[cfg(feature = "service-probe")]
const SERVICE_FRAME_PORT: u16 = 0x4d59;
#[cfg(feature = "service-probe")]
const SERVICE_CALL_PORT: u16 = 0x4d5a;
#[cfg(feature = "service-probe")]
const SERVICE_ROOT: u64 = 0x00c0_0000;
#[cfg(feature = "service-probe")]
const SERVICE_ENTRY: u64 = 0x0000_0001_4000_1000;
#[cfg(feature = "service-probe")]
const SERVICE_STACK_TOP: u64 = 0x0070_2000;

const GDT_ENTRIES: usize = 8;
const TSS_SELECTOR: u16 = 0x08;
const PRIVILEGE_STACK_BYTES: usize = 16 * 1024;

static mut GDT: [u64; GDT_ENTRIES] = [
    0,
    0,
    0,
    0x00cf_f300_0000_ffff,
    0x00af_fb00_0000_ffff,
    0,
    0x00cf_9300_0000_ffff,
    0x00af_9b00_0000_ffff,
];
static mut IDT: [InterruptGate; 256] = [InterruptGate::MISSING; 256];
static mut TSS: AlignedTaskState = AlignedTaskState::zeroed();

#[repr(C, align(4096))]
struct PrivilegeStack([u8; PRIVILEGE_STACK_BYTES]);

static mut KERNEL_ENTRY_STACK: PrivilegeStack = PrivilegeStack([0; PRIVILEGE_STACK_BYTES]);
static mut DOUBLE_FAULT_STACK: PrivilegeStack = PrivilegeStack([0; PRIVILEGE_STACK_BYTES]);
#[cfg(feature = "timer-probe")]
static mut TIMER_SCHEDULER: Option<KernelScheduler<1>> = None;
#[cfg(feature = "user-probe")]
static mut USER_RUNTIME: Option<TaskRuntime<2, 1>> = None;
#[cfg(feature = "user-probe")]
static mut USER_ENDPOINT: Option<Endpoint> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_CONTEXT: Option<UserContext> = None;

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

    .global mrml_timer_interrupt
mrml_timer_interrupt:
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
    and rsp, -16
    call mrml_timer_dispatch
    ud2

    .global mrml_user_probe
mrml_user_probe:
    int 0x80
    ud2
1:
    jmp 1b

    .global mrml_user_replacement_probe
mrml_user_replacement_probe:
    int3
2:
    jmp 2b

    .global mrml_user_call
mrml_user_call:
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
    call mrml_user_call_dispatch
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rdi
    pop rsi
    pop rbp
    pop rbx
    pop rdx
    pop rcx
    pop rax
    iretq
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
    #[cfg(feature = "timer-probe")]
    fn mrml_timer_interrupt() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_probe() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_replacement_probe() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_call() -> !;
    #[cfg(all(feature = "service-probe", not(feature = "user-probe")))]
    fn mrml_user_call() -> !;
    static mrml_exception_table: [u64; 32];
}

#[unsafe(no_mangle)]
unsafe extern "sysv64" fn mrml_user_call_dispatch(frame: *mut UserCallFrame) {
    #[cfg(feature = "user-probe")]
    unsafe {
        let frame = match frame.as_mut() {
            Some(frame) => frame,
            None => halt(),
        };
        let request = match frame.request() {
            Ok(request) => request,
            Err(_) => halt(),
        };
        let (endpoint_capability, receiver) = match &request {
            SyscallRequest::SendInline {
                endpoint, receiver, ..
            } => (*endpoint, *receiver),
            SyscallRequest::Yield => halt(),
        };
        let payload = request.payload();
        let runtime = match (*core::ptr::addr_of_mut!(USER_RUNTIME)).as_mut() {
            Some(runtime) => runtime,
            None => halt(),
        };
        let sender = match runtime.current() {
            Some(sender) => sender,
            None => halt(),
        };
        let endpoint = match (*core::ptr::addr_of_mut!(USER_ENDPOINT)).as_mut() {
            Some(endpoint) => endpoint,
            None => halt(),
        };
        let (sequence, message) = match runtime.send_ipc(
            sender,
            receiver,
            endpoint,
            endpoint_capability,
            payload,
            &[],
        ) {
            Ok(result) => result,
            Err(_) => halt(),
        };
        if sequence != 1 || message.payload() != b"ping" {
            halt();
        }
        frame.complete(sequence);
        asm!(
            "out dx, eax",
            in("dx") USER_CALL_PROBE_PORT,
            in("eax") sequence as u32,
            options(nomem, nostack)
        );
    }
    #[cfg(feature = "service-probe")]
    unsafe {
        let frame = match frame.as_mut() {
            Some(frame) => frame,
            None => halt(),
        };
        if frame.request() != Ok(SyscallRequest::Yield) {
            halt();
        }
        frame.complete(0);
        asm!(
            "out dx, eax",
            in("dx") SERVICE_CALL_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
    }
    #[cfg(not(any(feature = "user-probe", feature = "service-probe")))]
    let _ = frame;
}

#[unsafe(no_mangle)]
unsafe extern "sysv64" fn mrml_timer_dispatch() -> ! {
    #[cfg(feature = "timer-probe")]
    unsafe {
        let scheduler = match (*core::ptr::addr_of_mut!(TIMER_SCHEDULER)).as_mut() {
            Some(scheduler) => scheduler,
            None => halt(),
        };
        if scheduler.timer_tick().is_err() || scheduler.ticks() != 1 {
            halt();
        }
        asm!(
            "out dx, eax",
            in("dx") TIMER_TICK_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
    }
    halt()
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
    #[cfg(feature = "service-probe")]
    unsafe {
        let proof = normalized.vector as u32
            | ((normalized.cs as u32 & 0xff) << 8)
            | ((normalized.ss as u32 & 0xff) << 16);
        asm!(
            "out dx, eax",
            in("dx") SERVICE_FRAME_PORT,
            in("eax") proof,
            options(nomem, nostack)
        );
    }
    let _disposition = match normalized.disposition(fault_address) {
        Ok(disposition) => disposition,
        Err(_) => halt(),
    };
    #[cfg(feature = "fault-probe")]
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") EXCEPTION_PROBE_PORT,
            in("eax") normalized.vector as u32,
            options(nomem, nostack)
        )
    };
    #[cfg(feature = "user-probe")]
    if matches!(_disposition, TrapDisposition::TerminateUser { .. }) {
        let runtime = unsafe {
            match (*core::ptr::addr_of_mut!(USER_RUNTIME)).as_mut() {
                Some(runtime) => runtime,
                None => halt(),
            }
        };
        let retired = match runtime.terminate_current_fault(_disposition) {
            Ok(retired) => retired,
            Err(_) => halt(),
        };
        match retired.next {
            ScheduleOutcome::Switch { to, .. } if retired.vector == 6 => {
                let context = match runtime.context(to) {
                    Ok(context) => context as *const UserContext,
                    Err(_) => halt(),
                };
                unsafe { enter_user_probe_context(&*context) };
            }
            ScheduleOutcome::Idle if retired.vector == 3 => {}
            _ => halt(),
        }
        unsafe {
            asm!(
                "out dx, eax",
                in("dx") USER_PROBE_PORT,
                in("eax") 3u32,
                options(nomem, nostack)
            )
        };
    }
    #[cfg(feature = "service-probe")]
    if _disposition
        == (TrapDisposition::TerminateUser {
            vector: 3,
            address: None,
        })
    {
        unsafe {
            asm!(
                "out dx, eax",
                in("dx") SERVICE_PROBE_PORT,
                in("eax") 3u32,
                options(nomem, nostack)
            )
        };
    }
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
    #[cfg(feature = "service-probe")]
    unsafe {
        let root = match PhysAddr::new(SERVICE_ROOT) {
            Ok(root) => root,
            Err(_) => halt(),
        };
        let context = match UserContext::new(root, SERVICE_ENTRY, SERVICE_STACK_TOP) {
            Ok(context) => context,
            Err(_) => halt(),
        };
        let context_pointer = core::ptr::addr_of_mut!(SERVICE_CONTEXT);
        context_pointer.write(Some(context));
        let context = match (*context_pointer).as_ref() {
            Some(context) => context as *const UserContext,
            None => halt(),
        };
        enter_service_probe_context(&*context);
    }
    #[cfg(feature = "user-probe")]
    unsafe {
        let user_stack: u64;
        let page_table: u64;
        asm!("mov {}, rsp", out(reg) user_stack, options(nomem, nostack, preserves_flags));
        asm!("mov {}, cr3", out(reg) page_table, options(nomem, nostack, preserves_flags));
        let page_table = match PhysAddr::new(page_table) {
            Ok(page_table) => page_table,
            Err(_) => halt(),
        };
        let context = match UserContext::new(
            page_table,
            mrml_user_probe as *const () as usize as u64,
            user_stack & !0xf,
        ) {
            Ok(context) => context,
            Err(_) => halt(),
        };
        let replacement_context = match UserContext::new(
            page_table,
            mrml_user_replacement_probe as *const () as usize as u64,
            (user_stack & !0xf)
                .checked_sub(4096)
                .unwrap_or_else(|| halt()),
        ) {
            Ok(context) => context,
            Err(_) => halt(),
        };
        let mut runtime = match TaskRuntime::<2, 1>::new(1_000, 1) {
            Ok(runtime) => runtime,
            Err(_) => halt(),
        };
        let task = match runtime.create(Priority::NORMAL, context) {
            Ok(task) => task,
            Err(_) => halt(),
        };
        let replacement = match runtime.create(Priority::NORMAL, replacement_context) {
            Ok(task) => task,
            Err(_) => halt(),
        };
        let endpoint_object = ObjectId(0x81);
        let endpoint_capability = match runtime.capabilities_mut(task).and_then(|space| {
            space
                .insert(endpoint_object, Rights::SIGNAL)
                .map_err(|_| mrml_kernel::TaskRuntimeError::IntegrityFailure)
        }) {
            Ok(capability) => capability,
            Err(_) => halt(),
        };
        let context = match runtime.context_mut(task) {
            Ok(context) => context,
            Err(_) => halt(),
        };
        context.rax = 1;
        context.rdi = endpoint_capability.token();
        context.rsi = replacement.token();
        context.rdx = 4;
        context.r10 = u64::from_le_bytes(*b"ping\0\0\0\0");
        core::ptr::addr_of_mut!(USER_ENDPOINT).write(Some(Endpoint::new(endpoint_object)));
        if !matches!(
            runtime.start(),
            ScheduleOutcome::Switch { from: None, to } if to == task
        ) {
            halt();
        }
        let runtime_pointer = core::ptr::addr_of_mut!(USER_RUNTIME);
        runtime_pointer.write(Some(runtime));
        let context_pointer = match (*runtime_pointer).as_ref() {
            Some(runtime) => match runtime.context(task) {
                Ok(context) => context as *const UserContext,
                Err(_) => halt(),
            },
            None => halt(),
        };
        enter_user_probe_context(&*context_pointer);
    }
    #[cfg(feature = "timer-probe")]
    unsafe {
        let mut scheduler = match KernelScheduler::<1>::new(1_000, 1) {
            Ok(scheduler) => scheduler,
            Err(_) => halt(),
        };
        if scheduler.create(Priority::NORMAL).is_err()
            || !matches!(scheduler.start(), ScheduleOutcome::Switch { .. })
        {
            halt();
        }
        core::ptr::addr_of_mut!(TIMER_SCHEDULER).write(Some(scheduler));
        asm!(
            "out dx, eax",
            "sti",
            "hlt",
            "cli",
            in("dx") TIMER_READY_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
    }
    #[cfg(all(
        not(feature = "timer-probe"),
        not(feature = "user-probe"),
        not(feature = "service-probe")
    ))]
    {
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
    }
    halt()
}

#[cfg(feature = "user-probe")]
unsafe fn enter_user_probe_context(context: &UserContext) {
    unsafe { enter_user_context(context) }
}

#[cfg(feature = "service-probe")]
unsafe fn enter_service_probe_context(context: &UserContext) {
    let transition_stack =
        core::ptr::addr_of!(KERNEL_ENTRY_STACK) as u64 + PRIVILEGE_STACK_BYTES as u64;
    unsafe { enter_user_context_on_stack(context, transition_stack) }
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
    let kernel_stack =
        core::ptr::addr_of!(KERNEL_ENTRY_STACK) as u64 + PRIVILEGE_STACK_BYTES as u64;
    let double_fault_stack =
        core::ptr::addr_of!(DOUBLE_FAULT_STACK) as u64 + PRIVILEGE_STACK_BYTES as u64;
    let task_state = match TaskStateSegment::new(kernel_stack, double_fault_stack) {
        Ok(task_state) => task_state,
        Err(_) => halt(),
    };
    let task_state_pointer = unsafe { core::ptr::addr_of_mut!(TSS.0) };
    unsafe { task_state_pointer.write(task_state) };
    if unsafe {
        write_task_state_descriptor(
            core::ptr::addr_of_mut!(GDT).cast::<u64>(),
            GDT_ENTRIES,
            TSS_SELECTOR,
            &*task_state_pointer,
        )
    }
    .is_err()
    {
        halt();
    }
    if unsafe {
        install_exception_tables(
            core::ptr::addr_of!(GDT).cast::<u64>(),
            GDT_ENTRIES,
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
    if unsafe { load_task_register(TSS_SELECTOR) }.is_err() {
        halt();
    }
    #[cfg(any(feature = "user-probe", feature = "service-probe"))]
    if unsafe {
        install_user_call_gate(
            core::ptr::addr_of_mut!(IDT).cast::<InterruptGate>(),
            256,
            mrml_user_call as *const () as usize as u64,
            0x38,
        )
    }
    .is_err()
    {
        halt();
    }
    #[cfg(feature = "timer-probe")]
    if unsafe {
        install_external_interrupt_gate(
            core::ptr::addr_of_mut!(IDT).cast::<InterruptGate>(),
            256,
            TIMER_VECTOR,
            mrml_timer_interrupt as *const () as usize as u64,
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
