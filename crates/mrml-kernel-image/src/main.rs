#![no_std]
#![no_main]

#[cfg(not(feature = "fault-probe"))]
use core::arch::x86_64::{__cpuid, __cpuid_count};
use core::arch::{asm, global_asm};
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
#[cfg(feature = "production-policy")]
use mrml_kernel::BootPolicy;
#[cfg(any(feature = "timer-probe", feature = "smp-scheduler-probe"))]
use mrml_kernel::KernelScheduler;
#[cfg(all(feature = "smp-ipi-probe", not(feature = "smp-periodic-balance-probe")))]
use mrml_kernel::PeriodicBalancer;
#[cfg(any(
    feature = "user-probe",
    feature = "service-probe",
    feature = "smp-service-migration-probe"
))]
use mrml_kernel::SyscallRequest;
#[cfg(any(
    feature = "user-probe",
    feature = "service-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "smp-ipi-probe"
))]
use mrml_kernel::TaskRuntime;
use mrml_kernel::UserCallFrame;
#[cfg(feature = "smp-ipi-probe")]
use mrml_kernel::arch::x86_64::LocalApicController;
#[cfg(any(
    feature = "timer-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "smp-scheduler-probe",
    feature = "smp-ipi-probe"
))]
use mrml_kernel::arch::x86_64::LocalApicTimer;
#[cfg(any(
    feature = "timer-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "smp-scheduler-probe",
    feature = "smp-periodic-balance-probe"
))]
use mrml_kernel::arch::x86_64::TimerDivide;
#[cfg(any(
    feature = "user-probe",
    feature = "service-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe"
))]
use mrml_kernel::arch::x86_64::TrapDisposition;
#[cfg(any(
    feature = "user-probe",
    feature = "service-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "smp-ipi-probe"
))]
use mrml_kernel::arch::x86_64::UserContext;
#[cfg(any(feature = "user-probe", feature = "preemption-probe"))]
use mrml_kernel::arch::x86_64::enter_user_context;
#[cfg(any(
    feature = "service-probe",
    feature = "service-preemption-probe",
    feature = "smp-service-migration-probe"
))]
use mrml_kernel::arch::x86_64::enter_user_context_on_stack;
#[cfg(not(feature = "fault-probe"))]
use mrml_kernel::arch::x86_64::{
    ActiveApTrampolinePage, ActivePageTables, ApStartupTable, ApStartupTiming, ApTrampolineImage,
    ApTrampolinePage, ApicIpi, PerCpuPrivilegeStacks, X86CpuTopology,
};
use mrml_kernel::arch::x86_64::{
    ApOnlineTable, CpuDescriptorState, HardwareTrapFrame, MAX_X86_64_CPUS,
    PRIVILEGE_STACK_ARENA_PAGES, PrivilegeStackLayout,
};
#[cfg(feature = "uefi-service-preemption-probe")]
use mrml_kernel::arch::x86_64::{PreallocatedPageTableStore, ServiceAddressSpace};
#[cfg(not(feature = "fault-probe"))]
use mrml_kernel::{
    BootHandoff, HANDOFF_HEADER_BYTES, MAX_HANDOFF_BYTES, MAX_HANDOFF_REGIONS, MemoryKind,
    MemoryRegion, PhysAddr,
};
#[cfg(feature = "service-probe")]
use mrml_kernel::{
    Capability, CapabilitySpace, RestartPolicy, ServiceError, ServiceId, ServiceSupervisor,
};
#[cfg(all(
    not(feature = "fault-probe"),
    not(feature = "timer-probe"),
    not(feature = "preemption-probe"),
    not(feature = "service-preemption-probe"),
    not(feature = "user-probe"),
    not(feature = "service-probe")
))]
use mrml_kernel::{Color, EarlyKernelContext, FramebufferSurface};
#[cfg(feature = "smp-ipi-probe")]
use mrml_kernel::{DetachedTaskDomain, OwnershipMailbox, SchedulerLoad};
#[cfg(feature = "smp-periodic-balance-probe")]
use mrml_kernel::{DomainBalanceOutcome, PeriodicDomainBalancer};
#[cfg(any(feature = "user-probe", feature = "service-probe"))]
use mrml_kernel::{Endpoint, ObjectId, Rights};
#[cfg(all(not(feature = "fault-probe"), feature = "gpu-benchmark"))]
use mrml_kernel::{
    GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES, GpuGuestCommandPublisher, GpuQueueIdentity,
    GpuQueueSender, GpuResourceResponse, GpuResourceResponseReceiver, GpuSharedRingIndices,
    ResourceCommand,
};
#[cfg(any(
    feature = "timer-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "user-probe",
    feature = "service-probe",
    feature = "smp-scheduler-probe",
    feature = "smp-ipi-probe"
))]
use mrml_kernel::{Priority, ScheduleOutcome};

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
#[cfg(any(
    feature = "timer-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe"
))]
const TIMER_READY_PORT: u16 = 0x4d54;
#[cfg(feature = "timer-probe")]
const TIMER_TICK_PORT: u16 = 0x4d55;
#[cfg(any(
    feature = "timer-probe",
    feature = "preemption-probe",
    feature = "service-preemption-probe",
    feature = "smp-scheduler-probe",
    feature = "smp-periodic-balance-probe"
))]
const TIMER_VECTOR: u8 = 32;
#[cfg(feature = "smp-ipi-probe")]
const RESCHEDULE_VECTOR: u8 = 33;
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
#[cfg(feature = "smp-probe")]
const SMP_PROBE_PORT: u16 = 0x4d5c;
#[cfg(any(
    feature = "whp-smp-probe",
    feature = "whp-smp-scheduler-probe",
    feature = "whp-smp-ipi-probe",
    feature = "whp-smp-service-migration-probe",
    feature = "whp-smp-periodic-balance-probe"
))]
const WHP_SMP_HANDSHAKE_PORT: u16 = 0x4d5d;
#[cfg(feature = "smp-scheduler-probe")]
const SMP_SCHEDULER_READY_PORT: u16 = 0x4d5e;
#[cfg(feature = "smp-scheduler-probe")]
const SMP_SCHEDULER_TICK_PORT: u16 = 0x4d5f;
#[cfg(feature = "smp-ipi-probe")]
const SMP_IPI_READY_PORT: u16 = 0x4d60;
#[cfg(all(
    feature = "smp-ipi-probe",
    not(feature = "smp-service-migration-probe"),
    not(feature = "smp-periodic-balance-probe")
))]
const SMP_IPI_PROOF_PORT: u16 = 0x4d61;
#[cfg(feature = "smp-service-migration-probe")]
const SMP_SERVICE_MIGRATION_PROOF_PORT: u16 = 0x4d62;
#[cfg(feature = "smp-periodic-balance-probe")]
const SMP_PERIODIC_APPLICATION_PORT: u16 = 0x4d63;
#[cfg(feature = "smp-periodic-balance-probe")]
const SMP_PERIODIC_BOOTSTRAP_PORT: u16 = 0x4d64;
#[cfg(any(
    feature = "service-probe",
    feature = "smp-service-migration-probe",
    all(
        feature = "service-preemption-probe",
        not(feature = "uefi-service-preemption-probe")
    )
))]
const SERVICE_ROOT: u64 = 0x00c0_0000;
#[cfg(any(
    feature = "service-probe",
    all(
        feature = "service-preemption-probe",
        not(feature = "uefi-service-preemption-probe")
    )
))]
const SERVICE_B_ROOT: u64 = 0x00d0_0000;
#[cfg(any(
    feature = "service-probe",
    feature = "smp-service-migration-probe",
    all(
        feature = "service-preemption-probe",
        not(feature = "uefi-service-preemption-probe")
    )
))]
const SERVICE_ENTRY: u64 = 0x0000_0001_4000_1000;
#[cfg(any(
    feature = "service-probe",
    all(
        feature = "service-preemption-probe",
        not(feature = "uefi-service-preemption-probe")
    )
))]
const SERVICE_SENDER_ENTRY: u64 = SERVICE_ENTRY + 0x80;
#[cfg(any(
    feature = "service-probe",
    feature = "smp-service-migration-probe",
    all(
        feature = "service-preemption-probe",
        not(feature = "uefi-service-preemption-probe")
    )
))]
const SERVICE_STACK_TOP: u64 = 0x0070_2000;

static mut CPU0_DESCRIPTORS: CpuDescriptorState = CpuDescriptorState::empty(0);
struct ApDescriptorSlots([UnsafeCell<MaybeUninit<CpuDescriptorState>>; MAX_X86_64_CPUS]);
unsafe impl Sync for ApDescriptorSlots {}
// This storage is initialized independently by application processors after
// the image has been sealed. Force it into the writable PE data region;
// interior mutability alone is not a PE section-permission contract.
#[unsafe(link_section = ".data")]
static AP_DESCRIPTORS: ApDescriptorSlots =
    ApDescriptorSlots([const { UnsafeCell::new(MaybeUninit::uninit()) }; MAX_X86_64_CPUS]);
static AP_ONLINE: ApOnlineTable<MAX_X86_64_CPUS> = ApOnlineTable::empty();
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
struct ApSchedulerSlot {
    apic_id: AtomicU32,
    initialized: AtomicBool,
    #[cfg(feature = "smp-scheduler-probe")]
    scheduler: UnsafeCell<MaybeUninit<KernelScheduler<2>>>,
}
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
unsafe impl Sync for ApSchedulerSlot {}
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
impl ApSchedulerSlot {
    const fn empty() -> Self {
        Self {
            apic_id: AtomicU32::new(u32::MAX),
            initialized: AtomicBool::new(false),
            #[cfg(feature = "smp-scheduler-probe")]
            scheduler: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
struct ApSchedulerSlots([ApSchedulerSlot; MAX_X86_64_CPUS]);
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
unsafe impl Sync for ApSchedulerSlots {}
#[cfg(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe"))]
#[unsafe(link_section = ".data")]
static AP_SCHEDULERS: ApSchedulerSlots =
    ApSchedulerSlots([const { ApSchedulerSlot::empty() }; MAX_X86_64_CPUS]);
#[cfg(feature = "smp-ipi-probe")]
static AP_IPI_READY: [AtomicBool; MAX_X86_64_CPUS] =
    [const { AtomicBool::new(false) }; MAX_X86_64_CPUS];
#[cfg(feature = "smp-ipi-probe")]
static AP_RESCHEDULE_REQUEST: [AtomicBool; MAX_X86_64_CPUS] =
    [const { AtomicBool::new(false) }; MAX_X86_64_CPUS];
#[cfg(feature = "smp-ipi-probe")]
static AP_SCHEDULER_LOAD: [AtomicU32; MAX_X86_64_CPUS] =
    [const { AtomicU32::new(0) }; MAX_X86_64_CPUS];
#[cfg(all(feature = "smp-ipi-probe", not(feature = "smp-periodic-balance-probe")))]
struct SmpIpiRuntime(UnsafeCell<MaybeUninit<TaskRuntime<2, 1>>>);
#[cfg(feature = "smp-periodic-balance-probe")]
struct SmpIpiRuntime(UnsafeCell<MaybeUninit<TaskRuntime<3, 1>>>);
#[cfg(feature = "smp-ipi-probe")]
unsafe impl Sync for SmpIpiRuntime {}
#[cfg(feature = "smp-ipi-probe")]
#[unsafe(link_section = ".data")]
static SMP_IPI_RUNTIME: SmpIpiRuntime = SmpIpiRuntime(UnsafeCell::new(MaybeUninit::uninit()));
#[cfg(feature = "smp-ipi-probe")]
static SMP_MIGRATION_MAILBOX: OwnershipMailbox<DetachedTaskDomain<1>> = OwnershipMailbox::new();
#[cfg(feature = "smp-periodic-balance-probe")]
static SMP_LOCAL_MIGRATION_MAILBOX: OwnershipMailbox<DetachedTaskDomain<1>> =
    OwnershipMailbox::new();
#[cfg(feature = "smp-periodic-balance-probe")]
struct SmpPeriodicSource(UnsafeCell<MaybeUninit<TaskRuntime<5, 1>>>);
#[cfg(feature = "smp-periodic-balance-probe")]
unsafe impl Sync for SmpPeriodicSource {}
#[cfg(feature = "smp-periodic-balance-probe")]
#[unsafe(link_section = ".data")]
static SMP_PERIODIC_SOURCE: SmpPeriodicSource =
    SmpPeriodicSource(UnsafeCell::new(MaybeUninit::uninit()));
#[cfg(feature = "smp-periodic-balance-probe")]
struct SmpPeriodicPolicy(UnsafeCell<MaybeUninit<PeriodicDomainBalancer<2, 1>>>);
#[cfg(feature = "smp-periodic-balance-probe")]
unsafe impl Sync for SmpPeriodicPolicy {}
#[cfg(feature = "smp-periodic-balance-probe")]
#[unsafe(link_section = ".data")]
static SMP_PERIODIC_POLICY: SmpPeriodicPolicy =
    SmpPeriodicPolicy(UnsafeCell::new(MaybeUninit::uninit()));
#[cfg(feature = "smp-periodic-balance-probe")]
static SMP_PERIODIC_TARGET_CPU: AtomicU32 = AtomicU32::new(u32::MAX);
#[cfg(feature = "smp-periodic-balance-probe")]
static SMP_PERIODIC_TARGET_APIC: AtomicU32 = AtomicU32::new(u32::MAX);
#[cfg(feature = "smp-periodic-balance-probe")]
#[unsafe(link_section = ".data")]
static SMP_PERIODIC_ROUND: AtomicU32 = AtomicU32::new(0);
#[cfg(feature = "smp-periodic-balance-probe")]
#[unsafe(link_section = ".data")]
static SMP_PERIODIC_TIMER_TICK: AtomicU32 = AtomicU32::new(0);
#[cfg(not(feature = "fault-probe"))]
struct ApStartupWorkspace(UnsafeCell<MaybeUninit<ApStartupTable<MAX_X86_64_CPUS>>>);
#[cfg(not(feature = "fault-probe"))]
unsafe impl Sync for ApStartupWorkspace {}
#[cfg(not(feature = "fault-probe"))]
static AP_STARTUP_WORKSPACE: ApStartupWorkspace =
    ApStartupWorkspace(UnsafeCell::new(MaybeUninit::uninit()));
#[cfg(feature = "timer-probe")]
static mut TIMER_SCHEDULER: Option<KernelScheduler<1>> = None;
#[cfg(any(feature = "preemption-probe", feature = "service-preemption-probe"))]
static mut PREEMPTION_RUNTIME: Option<TaskRuntime<2, 0>> = None;
#[cfg(feature = "user-probe")]
static mut USER_RUNTIME: Option<TaskRuntime<2, 1>> = None;
#[cfg(feature = "user-probe")]
static mut USER_ENDPOINT: Option<Endpoint> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_RUNTIME: Option<TaskRuntime<2, 1>> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_ENDPOINT: Option<Endpoint> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_SUPERVISOR: Option<ServiceSupervisor<2>> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_MANAGEMENT: Option<CapabilitySpace<2>> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_CONTROLS: Option<[Capability; 2]> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_IDS: Option<[ServiceId; 2]> = None;
#[cfg(feature = "service-probe")]
static mut SERVICE_RESTARTED: bool = false;

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
    push 0
    push 32
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
    call mrml_timer_dispatch
    ud2

    .global mrml_reschedule_interrupt
mrml_reschedule_interrupt:
    cld
    push 0
    push 33
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
    call mrml_reschedule_dispatch
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

    .global mrml_preemption_spin
mrml_preemption_spin:
1:
    pause
    jmp 1b

    .global mrml_preemption_replacement
mrml_preemption_replacement:
    int3
3:
    pause
    jmp 3b

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
    #[cfg(any(
        feature = "timer-probe",
        feature = "preemption-probe",
        feature = "service-preemption-probe",
        feature = "smp-scheduler-probe",
        feature = "smp-periodic-balance-probe"
    ))]
    fn mrml_timer_interrupt() -> !;
    #[cfg(feature = "smp-ipi-probe")]
    fn mrml_reschedule_interrupt() -> !;
    #[cfg(feature = "preemption-probe")]
    fn mrml_preemption_spin() -> !;
    #[cfg(feature = "preemption-probe")]
    fn mrml_preemption_replacement() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_probe() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_replacement_probe() -> !;
    #[cfg(feature = "user-probe")]
    fn mrml_user_call() -> !;
    #[cfg(all(
        any(feature = "service-probe", feature = "smp-service-migration-probe"),
        not(feature = "user-probe")
    ))]
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
            SyscallRequest::Yield | SyscallRequest::Receive | SyscallRequest::Exit => halt(),
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
        let request = frame.request().unwrap_or_else(|_| halt());
        let runtime = (*core::ptr::addr_of_mut!(SERVICE_RUNTIME))
            .as_mut()
            .unwrap_or_else(|| halt());
        let current = runtime.current().unwrap_or_else(|| halt());
        let root = runtime
            .context(current)
            .unwrap_or_else(|_| halt())
            .page_table();
        *runtime.context_mut(current).unwrap_or_else(|_| halt()) =
            UserContext::from_user_call(root, frame).unwrap_or_else(|_| halt());
        match request {
            SyscallRequest::Receive => match runtime
                .receive_or_block_current()
                .unwrap_or_else(|_| halt())
            {
                Ok(message) => frame.complete_message(message.payload()),
                Err(ScheduleOutcome::Switch { to, .. }) => {
                    asm!(
                        "out dx, eax",
                        in("dx") SERVICE_CALL_PORT,
                        in("eax") 1u32,
                        options(nomem, nostack)
                    );
                    enter_service_task(runtime, to)
                }
                _ => halt(),
            },
            SyscallRequest::SendInline {
                endpoint,
                receiver,
                payload,
                length,
            } => {
                let endpoint_object = (*core::ptr::addr_of_mut!(SERVICE_ENDPOINT))
                    .as_mut()
                    .unwrap_or_else(|| halt());
                let sequence = runtime
                    .deliver_ipc(
                        current,
                        receiver,
                        endpoint_object,
                        endpoint,
                        &payload[..usize::from(length)],
                        &[],
                    )
                    .unwrap_or_else(|_| halt());
                frame.complete(sequence);
                asm!(
                    "out dx, eax",
                    in("dx") SERVICE_CALL_PORT,
                    in("eax") 2u32,
                    options(nomem, nostack)
                );
            }
            SyscallRequest::Yield => match runtime.yield_current().unwrap_or_else(|_| halt()) {
                ScheduleOutcome::Switch { to, .. } => {
                    let message = runtime.receive_ipc(to).unwrap_or_else(|_| halt());
                    runtime
                        .context_mut(to)
                        .unwrap_or_else(|_| halt())
                        .complete_message(message.payload());
                    asm!(
                        "out dx, eax",
                        in("dx") SERVICE_CALL_PORT,
                        in("eax") 3u32,
                        options(nomem, nostack)
                    );
                    enter_service_task(runtime, to)
                }
                _ => halt(),
            },
            SyscallRequest::Exit => {
                let supervisor = (*core::ptr::addr_of_mut!(SERVICE_SUPERVISOR))
                    .as_mut()
                    .unwrap_or_else(|| halt());
                let terminated = supervisor.exit_current(runtime).unwrap_or_else(|_| halt());
                if terminated.task != current || runtime.context(current).is_ok() {
                    halt();
                }
                match terminated.next {
                    ScheduleOutcome::Switch { to, .. } => {
                        let message = runtime.receive_ipc(to).unwrap_or_else(|_| halt());
                        runtime
                            .context_mut(to)
                            .unwrap_or_else(|_| halt())
                            .complete_message(message.payload());
                        asm!(
                            "out dx, eax",
                            in("dx") SERVICE_CALL_PORT,
                            in("eax") 3u32,
                            options(nomem, nostack)
                        );
                        enter_service_task(runtime, to)
                    }
                    _ => halt(),
                }
            }
        }
    }
    #[cfg(feature = "smp-service-migration-probe")]
    unsafe {
        let frame = frame.as_mut().unwrap_or_else(|| halt());
        if frame.request().unwrap_or_else(|_| halt()) != SyscallRequest::Receive {
            halt();
        }
        let runtime = (&mut *SMP_IPI_RUNTIME.0.get()).assume_init_mut();
        let task = runtime.current().unwrap_or_else(|| halt());
        let context = runtime.context(task).unwrap_or_else(|_| halt());
        if context.page_table() != PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()) {
            halt();
        }
        let apic_id = current_apic_id().unwrap_or_else(|| halt());
        let cpu = AP_SCHEDULERS
            .0
            .iter()
            .enumerate()
            .find(|(_, slot)| {
                slot.initialized.load(Ordering::Acquire)
                    && slot.apic_id.load(Ordering::Acquire) == apic_id
            })
            .map(|(cpu, _)| cpu)
            .unwrap_or_else(|| halt());
        asm!(
            "out dx, eax",
            in("dx") SMP_SERVICE_MIGRATION_PROOF_PORT,
            in("eax") ((cpu as u32) << 16) | task.token() as u32,
            options(nomem, nostack)
        );
        halt()
    }
    #[cfg(not(any(
        feature = "user-probe",
        feature = "service-probe",
        feature = "smp-service-migration-probe"
    )))]
    let _ = frame;
}

#[cfg(feature = "smp-periodic-balance-probe")]
unsafe fn publish_periodic_balance_tick(expected_tick: u32) {
    smp_trace(0x80u8.saturating_add(expected_tick as u8));
    let target_cpu = SMP_PERIODIC_TARGET_CPU.load(Ordering::Acquire) as usize;
    let target_apic = SMP_PERIODIC_TARGET_APIC.load(Ordering::Acquire);
    if target_cpu != 1 || target_apic == u32::MAX {
        halt();
    }
    let remote = AP_SCHEDULER_LOAD[target_cpu].load(Ordering::Acquire);
    let remote_load = SchedulerLoad::new((remote >> 16) as usize, (remote & 0xffff) as usize, 3)
        .unwrap_or_else(|| halt());
    let runtime = unsafe { (&mut *SMP_PERIODIC_SOURCE.0.get()).assume_init_mut() };
    let loads = [runtime.load(), remote_load];
    let mailboxes = [&SMP_LOCAL_MIGRATION_MAILBOX, &SMP_MIGRATION_MAILBOX];
    let policy = unsafe { (&mut *SMP_PERIODIC_POLICY.0.get()).assume_init_mut() };
    let outcome = policy
        .timer_tick_and_publish(0, &loads, runtime, &mailboxes)
        .unwrap_or_else(|_| halt());
    smp_trace(0x82u8.saturating_add(expected_tick as u8));
    if runtime.ticks() != u64::from(expected_tick)
        || !matches!(
            outcome.balancing(),
            DomainBalanceOutcome::Published(target) if target.cpu() == target_cpu
        )
    {
        halt();
    }
    if AP_RESCHEDULE_REQUEST[target_cpu]
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        halt();
    }
    unsafe { ApicIpi::fixed(target_apic, RESCHEDULE_VECTOR).and_then(|ipi| ipi.send()) }
        .unwrap_or_else(|_| halt());
    smp_trace(0x84u8.saturating_add(expected_tick as u8));
    while SMP_PERIODIC_ROUND.load(Ordering::Acquire) < expected_tick {
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
    }
    if SMP_PERIODIC_ROUND.load(Ordering::Acquire) != expected_tick
        || runtime.load().runnable() != 5 - expected_tick as usize
    {
        halt();
    }
    smp_trace(0x86u8.saturating_add(expected_tick as u8));
}

#[unsafe(no_mangle)]
unsafe extern "sysv64" fn mrml_timer_dispatch(frame: *const HardwareTrapFrame) -> ! {
    #[cfg(feature = "smp-periodic-balance-probe")]
    unsafe {
        let frame = frame.as_ref().unwrap_or_else(|| halt());
        let normalized = frame.normalize(0, 0);
        if normalized.vector != u64::from(TIMER_VECTOR) || normalized.cs & 3 != 0 {
            halt();
        }
        let tick = SMP_PERIODIC_TIMER_TICK.fetch_add(1, Ordering::AcqRel) + 1;
        if tick > 2 {
            halt();
        }
        smp_trace(0x70u8.saturating_add(tick as u8));
        LocalApicTimer::acknowledge();
        asm!(
            "sti",
            "2:",
            "hlt",
            "jmp 2b",
            options(noreturn, nomem, nostack)
        );
    }
    #[cfg(feature = "smp-scheduler-probe")]
    unsafe {
        let _ = frame;
        let apic_id = current_apic_id().unwrap_or_else(|| halt());
        let mut owner = None;
        for (cpu, slot) in AP_SCHEDULERS.0.iter().enumerate() {
            if slot.initialized.load(Ordering::Acquire)
                && slot.apic_id.load(Ordering::Acquire) == apic_id
                && owner.replace((cpu, slot)).is_some()
            {
                halt();
            }
        }
        let (cpu, slot) = owner.unwrap_or_else(|| halt());
        let scheduler = (&mut *slot.scheduler.get()).assume_init_mut();
        if scheduler.timer_tick().is_err() || scheduler.ticks() != 1 {
            halt();
        }
        LocalApicTimer::acknowledge();
        asm!(
            "out dx, eax",
            in("dx") SMP_SCHEDULER_TICK_PORT,
            in("eax") ((cpu as u32) << 16) | scheduler.ticks() as u32,
            options(nomem, nostack)
        );
        halt();
    }
    #[cfg(feature = "timer-probe")]
    let _ = frame;
    #[cfg(feature = "timer-probe")]
    unsafe {
        let scheduler = match (*core::ptr::addr_of_mut!(TIMER_SCHEDULER)).as_mut() {
            Some(scheduler) => scheduler,
            None => halt(),
        };
        if scheduler.timer_tick().is_err() || scheduler.ticks() != 1 {
            halt();
        }
        LocalApicTimer::acknowledge();
        asm!(
            "out dx, eax",
            in("dx") TIMER_TICK_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
    }
    #[cfg(any(feature = "preemption-probe", feature = "service-preemption-probe"))]
    unsafe {
        let frame = frame.as_ref().unwrap_or_else(|| halt());
        let normalized = frame.normalize(0, 0);
        asm!(
            "out dx, eax",
            in("dx") 0x4d5bu16,
            in("eax") ((normalized.cs as u32 & 0xff) << 8) | normalized.vector as u32,
            options(nomem, nostack)
        );
        if normalized.vector != u64::from(TIMER_VECTOR) || normalized.cs & 3 != 3 {
            halt();
        }
        #[cfg(feature = "uefi-service-preemption-probe")]
        asm!("out dx, al", in("dx") 0xe9u16, in("al") 0x90u8, options(nomem, nostack));
        asm!("out dx, eax", in("dx") 0x4d5bu16, in("eax") 1u32, options(nomem, nostack));
        let page_table: u64;
        asm!("mov {}, cr3", out(reg) page_table, options(nomem, nostack, preserves_flags));
        let interrupted = UserContext::from_trap(
            PhysAddr::new(page_table).unwrap_or_else(|_| halt()),
            &normalized,
        )
        .unwrap_or_else(|_| halt());
        asm!("out dx, eax", in("dx") 0x4d5bu16, in("eax") 2u32, options(nomem, nostack));
        let runtime_pointer = core::ptr::addr_of_mut!(PREEMPTION_RUNTIME);
        let runtime = (*runtime_pointer).as_mut().unwrap_or_else(|| halt());
        let next = match runtime.preempt_current(interrupted) {
            Ok(ScheduleOutcome::Switch { to, .. }) => to,
            _ => halt(),
        };
        asm!("out dx, eax", in("dx") 0x4d5bu16, in("eax") 3u32, options(nomem, nostack));
        let context = runtime
            .context(next)
            .map(|context| context as *const UserContext)
            .unwrap_or_else(|_| halt());
        asm!("out dx, eax", in("dx") 0x4d5bu16, in("eax") 4u32, options(nomem, nostack));
        LocalApicTimer::acknowledge();
        asm!("out dx, eax", in("dx") 0x4d5bu16, in("eax") 5u32, options(nomem, nostack));
        #[cfg(feature = "preemption-probe")]
        {
            enter_user_context(&*context)
        }
        #[cfg(feature = "service-preemption-probe")]
        {
            enter_service_preemption_context(&*context)
        }
    }
    #[cfg(not(any(
        feature = "preemption-probe",
        feature = "service-preemption-probe",
        feature = "smp-scheduler-probe",
        feature = "smp-periodic-balance-probe"
    )))]
    {
        #[cfg(not(feature = "timer-probe"))]
        let _ = frame;
        halt()
    }
}

#[unsafe(no_mangle)]
unsafe extern "sysv64" fn mrml_reschedule_dispatch(frame: *const HardwareTrapFrame) -> ! {
    #[cfg(feature = "smp-ipi-probe")]
    unsafe {
        #[cfg(feature = "smp-periodic-balance-probe")]
        smp_trace(0x76);
        let frame = frame.as_ref().unwrap_or_else(|| halt());
        let normalized = frame.normalize(0, 0);
        if normalized.vector != u64::from(RESCHEDULE_VECTOR) || normalized.cs & 3 != 0 {
            halt();
        }
        let apic_id = current_apic_id().unwrap_or_else(|| halt());
        let mut owner = None;
        for (cpu, slot) in AP_SCHEDULERS.0.iter().enumerate() {
            if slot.initialized.load(Ordering::Acquire)
                && slot.apic_id.load(Ordering::Acquire) == apic_id
                && owner.replace((cpu, slot)).is_some()
            {
                halt();
            }
        }
        let (cpu, _slot) = owner.unwrap_or_else(|| halt());
        if !AP_RESCHEDULE_REQUEST[cpu].swap(false, Ordering::AcqRel) {
            halt();
        }
        let runtime = (&mut *SMP_IPI_RUNTIME.0.get()).assume_init_mut();
        let mut ticket = SMP_MIGRATION_MAILBOX.take();
        let migration = runtime
            .attach_domain(&mut ticket)
            .unwrap_or_else(|_| halt());
        #[cfg(feature = "smp-periodic-balance-probe")]
        smp_trace(0x76u8.saturating_add(SMP_PERIODIC_ROUND.load(Ordering::Acquire) as u8));
        if ticket.is_some() {
            halt();
        }
        if migration.priority() != Priority::RESPONSIVE {
            halt();
        }
        let load = runtime.load();
        #[cfg(not(feature = "smp-periodic-balance-probe"))]
        if load.occupied() != 2 || load.runnable() != 2 {
            halt();
        }
        #[cfg(feature = "smp-periodic-balance-probe")]
        let round = SMP_PERIODIC_ROUND.fetch_add(1, Ordering::AcqRel) + 1;
        #[cfg(feature = "smp-periodic-balance-probe")]
        if round > 2
            || load.occupied() != round as usize + 1
            || load.runnable() != round as usize + 1
        {
            halt();
        }
        AP_SCHEDULER_LOAD[cpu].store(
            ((load.occupied() as u32) << 16) | load.runnable() as u32,
            Ordering::Release,
        );
        #[cfg(not(feature = "smp-periodic-balance-probe"))]
        let next = match runtime.yield_current().unwrap_or_else(|_| halt()) {
            ScheduleOutcome::Switch { to, .. } => to,
            _ => halt(),
        };
        #[cfg(feature = "smp-periodic-balance-probe")]
        let next = migration.destination();
        #[cfg(not(feature = "smp-periodic-balance-probe"))]
        if next != migration.destination() {
            halt();
        }
        let context = runtime.context(next).unwrap_or_else(|_| halt());
        #[cfg(not(feature = "smp-service-migration-probe"))]
        if context.instruction_pointer() != 0x0050_0000 {
            halt();
        }
        LocalApicTimer::acknowledge();
        #[cfg(feature = "smp-periodic-balance-probe")]
        {
            if round == 1 {
                asm!(
                    "sti",
                    "2:",
                    "hlt",
                    "jmp 2b",
                    options(noreturn, nomem, nostack)
                );
            }
            asm!(
                "out dx, eax",
                in("dx") SMP_PERIODIC_APPLICATION_PORT,
                in("eax") ((cpu as u32) << 16) | round,
                options(nomem, nostack)
            );
            halt();
        }
        #[cfg(feature = "smp-service-migration-probe")]
        {
            if context.instruction_pointer() != SERVICE_ENTRY
                || context.page_table() != PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt())
            {
                halt();
            }
            let descriptor = AP_DESCRIPTORS.0.get(cpu).unwrap_or_else(|| halt()).get();
            let state = (&*descriptor).assume_init_ref();
            let transition_stack =
                CpuDescriptorState::entry_stack_top_from(state).unwrap_or_else(|_| halt());
            enter_user_context_on_stack(context, transition_stack)
        }
        #[cfg(all(
            not(feature = "smp-service-migration-probe"),
            not(feature = "smp-periodic-balance-probe")
        ))]
        asm!(
            "out dx, eax",
            in("dx") SMP_IPI_PROOF_PORT,
            in("eax") ((cpu as u32) << 16) | next.token() as u32,
            options(nomem, nostack)
        );
        #[cfg(all(
            not(feature = "smp-service-migration-probe"),
            not(feature = "smp-periodic-balance-probe")
        ))]
        halt();
    }
    #[cfg(not(feature = "smp-ipi-probe"))]
    {
        let _ = frame;
        halt()
    }
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
    #[cfg(any(feature = "preemption-probe", feature = "service-preemption-probe"))]
    unsafe {
        let proof = if matches!(
            _disposition,
            TrapDisposition::TerminateUser {
                vector: 3,
                address: None
            }
        ) {
            6
        } else {
            0x8000 | normalized.vector as u32
        };
        asm!(
            "out dx, eax",
            in("dx") 0x4d5bu16,
            in("eax") proof,
            options(nomem, nostack)
        );
        if proof == 6 {
            #[cfg(feature = "uefi-service-preemption-probe")]
            asm!("out dx, al", in("dx") 0xe9u16, in("al") 0x91u8, options(nomem, nostack));
            halt();
        }
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
            let runtime = (*core::ptr::addr_of_mut!(SERVICE_RUNTIME))
                .as_mut()
                .unwrap_or_else(|| halt());
            let supervisor = (*core::ptr::addr_of_mut!(SERVICE_SUPERVISOR))
                .as_mut()
                .unwrap_or_else(|| halt());
            let fault = supervisor
                .fault_current(runtime, _disposition)
                .unwrap_or_else(|_| halt());
            if !matches!(fault.retirement.next, ScheduleOutcome::Idle) {
                halt();
            }
            if core::ptr::addr_of!(SERVICE_RESTARTED).read() {
                verify_restart_budget_exhausted(runtime, supervisor);
                asm!(
                    "out dx, eax",
                    in("dx") SERVICE_PROBE_PORT,
                    in("eax") 4u32,
                    options(nomem, nostack)
                );
                halt();
            }
            asm!(
                "out dx, eax",
                in("dx") SERVICE_PROBE_PORT,
                in("eax") 3u32,
                options(nomem, nostack)
            );
            restart_service_pair(runtime, supervisor);
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
/// Both privilege-stack tops must be canonical, 16-byte aligned addresses in
/// distinct writable supervisor mappings whose guard pages are absent.
#[unsafe(export_name = "efi_main")]
pub unsafe extern "efiapi" fn kernel_entry(
    bytes: *const u8,
    length: usize,
    entry_stack_top: u64,
    double_fault_stack_top: u64,
) -> usize {
    unsafe { install_descriptor_tables(entry_stack_top, double_fault_stack_top) };
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

/// Relocated long-mode entry reached only by a sealed AP trampoline. The
/// trampoline uses the SysV register contract explicitly: CPU index in RDI,
/// startup generation in RSI, and the private stack-layout base in RDX.
///
/// # Safety
///
/// The BSP must have armed the exact CPU/generation pair, installed the shared
/// kernel CR3, and supplied the base of that CPU's exclusively owned, mapped
/// privilege-stack layout. The current RSP must be its early-stack top.
#[unsafe(export_name = "mrml_ap_entry")]
pub unsafe extern "sysv64" fn ap_kernel_entry(cpu: u64, generation: u64, stack_base: u64) -> ! {
    let cpu = usize::try_from(cpu)
        .ok()
        .filter(|cpu| *cpu < MAX_X86_64_CPUS)
        .unwrap_or_else(|| halt());
    let generation = u32::try_from(generation).unwrap_or_else(|_| halt());
    if !AP_ONLINE.matches_armed(cpu, generation) {
        halt();
    }
    let stacks = PrivilegeStackLayout::new(stack_base, PRIVILEGE_STACK_ARENA_PAGES)
        .unwrap_or_else(|_| halt());
    let slot = AP_DESCRIPTORS.0[cpu].get();
    unsafe { slot.write(MaybeUninit::new(CpuDescriptorState::empty(cpu as u16))) };
    let state = unsafe { (&mut *slot).assume_init_mut() };
    unsafe {
        install_descriptor_state(
            state,
            stacks.entry_top().unwrap_or_else(|_| halt()),
            stacks.double_fault_top().unwrap_or_else(|_| halt()),
        )
    };
    #[cfg(any(
        feature = "whp-smp-probe",
        feature = "whp-smp-scheduler-probe",
        feature = "whp-smp-ipi-probe",
        feature = "whp-smp-service-migration-probe",
        feature = "whp-smp-periodic-balance-probe"
    ))]
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") WHP_SMP_HANDSHAKE_PORT,
            in("eax") ((cpu as u32) << 16) | generation,
            options(nomem, nostack)
        )
    };
    if AP_ONLINE.acknowledge(cpu, generation).is_err() {
        halt();
    }
    #[cfg(feature = "smp-scheduler-probe")]
    unsafe {
        let mut scheduler = KernelScheduler::<2>::new(1_000, 1).unwrap_or_else(|_| halt());
        if scheduler.create(Priority::NORMAL).is_err()
            || !matches!(scheduler.start(), ScheduleOutcome::Switch { .. })
        {
            halt();
        }
        let apic_id = current_apic_id().unwrap_or_else(|| halt());
        let slot = AP_SCHEDULERS.0.get(cpu).unwrap_or_else(|| halt());
        if slot.initialized.load(Ordering::Acquire)
            || slot
                .apic_id
                .compare_exchange(u32::MAX, apic_id, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            halt();
        }
        (*slot.scheduler.get()).write(scheduler);
        slot.initialized.store(true, Ordering::Release);
        LocalApicTimer::periodic(TIMER_VECTOR, 100_000, TimerDivide::By16)
            .and_then(|timer| timer.enable())
            .unwrap_or_else(|_| halt());
        asm!(
            "out dx, eax",
            in("dx") SMP_SCHEDULER_READY_PORT,
            in("eax") ((cpu as u32) << 16) | generation,
            options(nomem, nostack)
        );
        asm!(
            "sti",
            "2:",
            "pause",
            "jmp 2b",
            options(noreturn, nomem, nostack)
        );
    }
    #[cfg(feature = "smp-ipi-probe")]
    unsafe {
        let page_table: u64;
        asm!("mov {}, cr3", out(reg) page_table, options(nomem, nostack, preserves_flags));
        let context = UserContext::new(
            PhysAddr::new(page_table).unwrap_or_else(|_| halt()),
            0x0040_0000,
            0x0000_7000_0000_0000,
        )
        .unwrap_or_else(|_| halt());
        #[cfg(not(feature = "smp-periodic-balance-probe"))]
        let mut runtime = TaskRuntime::<2, 1>::new(1_000, 1).unwrap_or_else(|_| halt());
        #[cfg(feature = "smp-periodic-balance-probe")]
        let mut runtime = TaskRuntime::<3, 1>::new(1_000, 1).unwrap_or_else(|_| halt());
        let first = runtime
            .create(Priority::NORMAL, context)
            .unwrap_or_else(|_| halt());
        if !matches!(runtime.start(), ScheduleOutcome::Switch { to, .. } if to == first)
            || first.token() as u32 != 0
        {
            halt();
        }
        let apic_id = current_apic_id().unwrap_or_else(|| halt());
        let slot = AP_SCHEDULERS.0.get(cpu).unwrap_or_else(|| halt());
        if slot.initialized.load(Ordering::Acquire)
            || slot
                .apic_id
                .compare_exchange(u32::MAX, apic_id, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            halt();
        }
        (*SMP_IPI_RUNTIME.0.get()).write(runtime);
        slot.initialized.store(true, Ordering::Release);
        AP_SCHEDULER_LOAD[cpu].store(0x0001_0001, Ordering::Release);
        LocalApicController::enable().unwrap_or_else(|_| halt());
        AP_IPI_READY[cpu].store(true, Ordering::Release);
        asm!(
            "out dx, eax",
            in("dx") SMP_IPI_READY_PORT,
            in("eax") ((cpu as u32) << 16) | generation,
            options(nomem, nostack)
        );
        #[cfg(feature = "smp-periodic-balance-probe")]
        {
            while SMP_PERIODIC_TARGET_APIC.load(Ordering::Acquire) == u32::MAX {
                core::sync::atomic::compiler_fence(Ordering::SeqCst);
            }
            LocalApicTimer::periodic(TIMER_VECTOR, 1_000_000, TimerDivide::By16)
                .and_then(|timer| timer.enable())
                .unwrap_or_else(|_| halt());
        }
        asm!(
            "sti",
            "2:",
            "pause",
            "jmp 2b",
            options(noreturn, nomem, nostack)
        );
    }
    #[cfg(feature = "smp-probe")]
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") SMP_PROBE_PORT,
            in("eax") ((cpu as u32) << 16) | generation,
            options(nomem, nostack)
        )
    };
    #[cfg(not(any(feature = "smp-scheduler-probe", feature = "smp-ipi-probe")))]
    halt()
}

#[cfg(not(feature = "fault-probe"))]
unsafe fn start_application_processors(topology: &X86CpuTopology, handoff: &BootHandoff) {
    smp_trace(0x10);
    let bsp_apic_id = current_apic_id().unwrap_or_else(|| halt());
    smp_trace(0x60u8.saturating_add(bsp_apic_id as u8));
    // The BSP is the sole writer during one-way early boot. Keeping the full
    // 256-CPU state table out of its guarded early stack leaves enough space
    // for the sealed trampoline image and nested topology parsing frame.
    let startup_storage = unsafe { &mut *AP_STARTUP_WORKSPACE.0.get() };
    let startup =
        ApStartupTable::<MAX_X86_64_CPUS>::initialize(startup_storage, topology, bsp_apic_id)
            .unwrap_or_else(|_| halt());
    smp_trace(0x11);
    let timing = ap_startup_timing().unwrap_or_else(|| halt());
    smp_trace(0x12);
    let root = unsafe { ActivePageTables::current() }
        .map(|tables| tables.root().get())
        .unwrap_or_else(|_| halt());
    let trampoline = handoff.ap_trampoline().unwrap_or_else(|| halt());
    let stack_base = handoff.ap_stack_arena().unwrap_or_else(|| halt());
    let stacks = PerCpuPrivilegeStacks::<MAX_X86_64_CPUS>::new(
        stack_base,
        stack_base,
        PRIVILEGE_STACK_ARENA_PAGES,
    )
    .unwrap_or_else(|_| halt());
    let mut page =
        unsafe { ActiveApTrampolinePage::current(trampoline) }.unwrap_or_else(|_| halt());
    smp_trace(0x13);

    for cpu in 0..topology.len() {
        let target = topology.cpu(cpu).unwrap_or_else(|_| halt());
        if target.apic_id() == bsp_apic_id {
            continue;
        }
        let token = startup.begin(cpu).unwrap_or_else(|_| halt());
        AP_ONLINE.arm(token).unwrap_or_else(|_| halt());
        smp_trace(0x20);
        let layout = stacks
            .cpu(cpu)
            .map(|slot| slot.virtual_layout())
            .unwrap_or_else(|_| halt());
        let image = ApTrampolineImage::new(
            trampoline,
            root,
            ap_kernel_entry as *const () as usize as u64,
            layout.early_top().unwrap_or_else(|_| halt()),
            cpu,
            token.generation(),
        )
        .unwrap_or_else(|_| halt());
        let installed = image.install(&mut page).unwrap_or_else(|_| halt());
        smp_trace(0x21);
        let destination = startup.destination(token).unwrap_or_else(|_| halt());
        unsafe { ApicIpi::init(destination).and_then(|ipi| ipi.send()) }.unwrap_or_else(|_| halt());
        smp_trace(0x22);
        timing.wait_after_init().unwrap_or_else(|_| halt());
        smp_trace(0x23);
        unsafe { ApicIpi::init_deassert(destination).and_then(|ipi| ipi.send()) }
            .unwrap_or_else(|_| halt());
        startup
            .startup_sent_with_image(token, installed)
            .unwrap_or_else(|_| halt());
        unsafe {
            ApicIpi::startup(destination, installed.startup_vector()).and_then(|ipi| ipi.send())
        }
        .unwrap_or_else(|_| halt());
        smp_trace(0x24);
        timing.wait_after_startup().unwrap_or_else(|_| halt());
        smp_trace(0x25);
        if !AP_ONLINE.is_online(token).unwrap_or_else(|_| halt()) {
            smp_trace(0x28);
            unsafe {
                ApicIpi::startup(destination, installed.startup_vector()).and_then(|ipi| ipi.send())
            }
            .unwrap_or_else(|_| halt());
            timing
                .wait_micros(ap_retry_timeout_micros())
                .unwrap_or_else(|_| halt());
            smp_trace(0x29);
        }
        if !AP_ONLINE.is_online(token).unwrap_or_else(|_| halt()) {
            smp_trace(0x2a);
            if AP_ONLINE.fail(token).is_ok() {
                startup.fail(token).unwrap_or_else(|_| halt());
                #[cfg(feature = "smp-probe")]
                unsafe {
                    asm!(
                        "out dx, eax",
                        in("dx") SMP_PROBE_PORT,
                        in("eax") 0xdead_0001u32,
                        options(nomem, nostack)
                    )
                };
                halt();
            }
            if !AP_ONLINE.is_online(token).unwrap_or_else(|_| halt()) {
                halt();
            }
        }
        smp_trace(0x2b);
        startup
            .acknowledge(token, destination)
            .unwrap_or_else(|_| halt());
        smp_trace(0x26);
        installed.rearm(&mut page).unwrap_or_else(|_| halt());
        smp_trace(0x27);
    }
    if !page.revoke_and_zero(trampoline) {
        halt();
    }
    smp_trace(0x30);
}

#[cfg(all(not(feature = "fault-probe"), feature = "smp-ipi-probe"))]
#[cfg(feature = "smp-periodic-balance-probe")]
unsafe fn initialize_periodic_balance_probe(
    topology_len: usize,
    cpu: usize,
    destination: u32,
    root: PhysAddr,
) {
    if topology_len != 2 || cpu != 1 {
        halt();
    }
    let mut source = TaskRuntime::<5, 1>::new(1_000, 10).unwrap_or_else(|_| halt());
    let first = source
        .create(
            Priority::NORMAL,
            UserContext::new(root, 0x0040_0000, 0x0000_7000_0000_0000).unwrap_or_else(|_| halt()),
        )
        .unwrap_or_else(|_| halt());
    for index in 0..4u64 {
        source
            .create(
                Priority::RESPONSIVE,
                UserContext::new(root, 0x0050_0000, 0x0000_7000_0000_1000 + index * 0x1000)
                    .unwrap_or_else(|_| halt()),
            )
            .unwrap_or_else(|_| halt());
    }
    if !matches!(source.start(), ScheduleOutcome::Switch { to, .. } if to == first) {
        halt();
    }
    unsafe { (*SMP_PERIODIC_SOURCE.0.get()).write(source) };
    unsafe {
        (*SMP_PERIODIC_POLICY.0.get())
            .write(PeriodicDomainBalancer::<2, 1>::new(0, 1).unwrap_or_else(|_| halt()))
    };
    SMP_PERIODIC_TARGET_CPU.store(cpu as u32, Ordering::Release);
    SMP_PERIODIC_TARGET_APIC.store(destination, Ordering::Release);
}

#[cfg(feature = "smp-periodic-balance-probe")]
unsafe fn run_periodic_balance_probe() -> ! {
    unsafe { LocalApicController::enable() }.unwrap_or_else(|_| halt());
    smp_trace(0x70);
    for expected_tick in 1..=2 {
        while SMP_PERIODIC_TIMER_TICK.load(Ordering::Acquire) < expected_tick {
            core::sync::atomic::compiler_fence(Ordering::SeqCst);
        }
        unsafe { publish_periodic_balance_tick(expected_tick) };
    }
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") SMP_PERIODIC_BOOTSTRAP_PORT,
            in("eax") 2u32,
            options(nomem, nostack)
        )
    };
    halt();
}

#[cfg(all(not(feature = "fault-probe"), feature = "smp-ipi-probe"))]
unsafe fn send_reschedule_probe(topology: &X86CpuTopology) {
    let bsp_apic_id = current_apic_id().unwrap_or_else(|| halt());
    let mut target = None;
    let mut bsp_cpu = None;
    for cpu in 0..topology.len() {
        let descriptor = topology.cpu(cpu).unwrap_or_else(|_| halt());
        if descriptor.apic_id() == bsp_apic_id {
            if bsp_cpu.replace(cpu).is_some() {
                halt();
            }
        } else if target.replace((cpu, descriptor.apic_id())).is_some() {
            halt();
        }
    }
    let (cpu, destination) = target.unwrap_or_else(|| halt());
    #[cfg(not(feature = "smp-periodic-balance-probe"))]
    let bsp_cpu = bsp_cpu.unwrap_or_else(|| halt());
    #[cfg(feature = "smp-periodic-balance-probe")]
    if bsp_cpu.is_none() {
        halt();
    }
    let timing = ap_startup_timing().unwrap_or_else(|| halt());
    for _ in 0..1_000 {
        if AP_IPI_READY[cpu].load(Ordering::Acquire) {
            break;
        }
        timing.wait_micros(100).unwrap_or_else(|_| halt());
    }
    if !AP_IPI_READY[cpu].load(Ordering::Acquire) {
        halt();
    }
    #[cfg(not(feature = "smp-periodic-balance-probe"))]
    let remote_load = {
        let remote = AP_SCHEDULER_LOAD[cpu].load(Ordering::Acquire);
        SchedulerLoad::new((remote >> 16) as usize, (remote & 0xffff) as usize, 2)
            .unwrap_or_else(|| halt())
    };
    let page_table: u64;
    unsafe { asm!("mov {}, cr3", out(reg) page_table, options(nomem, nostack, preserves_flags)) };
    let root = PhysAddr::new(page_table).unwrap_or_else(|_| halt());
    #[cfg(feature = "smp-periodic-balance-probe")]
    {
        unsafe {
            initialize_periodic_balance_probe(topology.len(), cpu, destination, root);
            run_periodic_balance_probe();
        }
    }
    #[cfg(not(feature = "smp-periodic-balance-probe"))]
    {
        let mut source = TaskRuntime::<3, 1>::new(1_000, 1).unwrap_or_else(|_| halt());
        let first = source
            .create(
                Priority::NORMAL,
                UserContext::new(root, 0x0040_0000, 0x0000_7000_0000_0000)
                    .unwrap_or_else(|_| halt()),
            )
            .unwrap_or_else(|_| halt());
        #[cfg(feature = "smp-service-migration-probe")]
        let responsive_context = UserContext::new(
            PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()),
            SERVICE_ENTRY,
            SERVICE_STACK_TOP,
        )
        .unwrap_or_else(|_| halt());
        #[cfg(not(feature = "smp-service-migration-probe"))]
        let responsive_context =
            UserContext::new(root, 0x0050_0000, 0x0000_7000_0000_1000).unwrap_or_else(|_| halt());
        source
            .create(Priority::RESPONSIVE, responsive_context)
            .unwrap_or_else(|_| halt());
        source
            .create(
                Priority::BACKGROUND,
                UserContext::new(root, 0x0060_0000, 0x0000_7000_0000_2000)
                    .unwrap_or_else(|_| halt()),
            )
            .unwrap_or_else(|_| halt());
        if !matches!(source.start(), ScheduleOutcome::Switch { to, .. } if to == first) {
            halt();
        }
        if topology.len() != 2 || bsp_cpu >= 2 || cpu >= 2 {
            halt();
        }
        let unavailable = SchedulerLoad::new(0, 0, 0).unwrap_or_else(|| halt());
        let mut loads = [unavailable; 2];
        loads[bsp_cpu] = source.load();
        loads[cpu] = remote_load;
        let mut policy = PeriodicBalancer::<2>::new(0, 1).unwrap_or_else(|_| halt());
        let selected = policy
            .poll(1, bsp_cpu, &loads)
            .unwrap_or_else(|_| halt())
            .unwrap_or_else(|| halt());
        if selected.cpu() != cpu || selected.load() != remote_load {
            halt();
        }
        let detached = source
            .detach_domain_for_rebalance(selected.load())
            .unwrap_or_else(|_| halt())
            .unwrap_or_else(|| halt());
        if source.load().runnable() != 2 || SMP_MIGRATION_MAILBOX.publish(detached).is_err() {
            halt();
        }
        if AP_RESCHEDULE_REQUEST[cpu]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            halt();
        }
        unsafe { ApicIpi::fixed(destination, RESCHEDULE_VECTOR).and_then(|ipi| ipi.send()) }
            .unwrap_or_else(|_| halt());
        unsafe {
            asm!(
                "out dx, eax",
                in("dx") SMP_IPI_READY_PORT,
                in("eax") topology.len() as u32,
                options(nomem, nostack)
            )
        };
    }
}

#[cfg(not(feature = "fault-probe"))]
fn current_apic_id() -> Option<u32> {
    let maximum = __cpuid(0).eax;
    if maximum >= 0x0b {
        for level in 0..2 {
            let topology = __cpuid_count(0x0b, level);
            if topology.ebx & 0xffff != 0 {
                return Some(topology.edx);
            }
        }
    }
    (maximum >= 1).then(|| __cpuid(1).ebx >> 24)
}

#[cfg(not(feature = "fault-probe"))]
fn ap_startup_timing() -> Option<ApStartupTiming> {
    if let Ok(timing) = ApStartupTiming::detect() {
        return Some(timing);
    }
    #[cfg(feature = "emulated-ap-timing")]
    {
        let bytes = option_env!("MRML_EMULATED_TSC_HZ")?.as_bytes();
        if bytes.is_empty() || bytes.len() > 20 {
            return None;
        }
        let mut value = 0u64;
        for byte in bytes {
            if !byte.is_ascii_digit() {
                return None;
            }
            value = value.checked_mul(10)?.checked_add(u64::from(byte - b'0'))?;
        }
        ApStartupTiming::from_tsc_hz(value).ok()
    }
    #[cfg(not(feature = "emulated-ap-timing"))]
    None
}

#[cfg(all(not(feature = "fault-probe"), feature = "emulated-ap-timing"))]
const fn ap_retry_timeout_micros() -> u32 {
    // TCG vCPUs are host threads and can be descheduled well beyond the
    // architectural SIPI delay. This explicitly selected test profile keeps
    // production's hardware timeout unchanged.
    100_000
}

#[cfg(all(not(feature = "fault-probe"), not(feature = "emulated-ap-timing")))]
const fn ap_retry_timeout_micros() -> u32 {
    1_000
}

#[cfg(all(not(feature = "fault-probe"), feature = "ap-startup-trace"))]
fn smp_trace(stage: u8) {
    unsafe { asm!("out dx, al", in("dx") 0xe9u16, in("al") stage, options(nomem, nostack)) };
}

#[cfg(all(not(feature = "fault-probe"), not(feature = "ap-startup-trace")))]
const fn smp_trace(_: u8) {}

#[cfg(not(feature = "fault-probe"))]
#[allow(unreachable_code)]
unsafe fn run_kernel(bytes: *const u8, length: usize) -> ! {
    smp_trace(0x01);
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
    smp_trace(0x02);
    if region_count != handoff.region_count() {
        halt();
    }
    if let Some(madt) = handoff.madt(encoded) {
        smp_trace(0x03);
        let topology = X86CpuTopology::parse_madt(madt).unwrap_or_else(|_| halt());
        smp_trace(0x40u8.saturating_add(topology.len() as u8));
        match (handoff.ap_trampoline(), handoff.ap_stack_arena()) {
            (Some(_), Some(base)) => {
                let stacks = PerCpuPrivilegeStacks::<MAX_X86_64_CPUS>::new(
                    base,
                    base,
                    PRIVILEGE_STACK_ARENA_PAGES,
                )
                .unwrap_or_else(|_| halt());
                for cpu in 0..topology.len() {
                    if stacks.cpu(cpu).is_err() {
                        halt();
                    }
                }
                if topology.len() > 1 {
                    smp_trace(0x04);
                    unsafe { start_application_processors(&topology, &handoff) };
                    smp_trace(0x05);
                    #[cfg(feature = "smp-ipi-probe")]
                    unsafe {
                        send_reschedule_probe(&topology)
                    };
                    #[cfg(feature = "smp-probe")]
                    unsafe {
                        asm!(
                            "out dx, eax",
                            in("dx") SMP_PROBE_PORT,
                            in("eax") topology.len() as u32,
                            options(nomem, nostack)
                        )
                    };
                    #[cfg(feature = "smp-scheduler-probe")]
                    unsafe {
                        asm!(
                            "out dx, eax",
                            in("dx") SMP_SCHEDULER_READY_PORT,
                            in("eax") topology.len() as u32,
                            options(nomem, nostack)
                        )
                    };
                }
            }
            (None, None) if topology.len() == 1 => {}
            _ => halt(),
        }
    }
    #[cfg(feature = "service-probe")]
    unsafe {
        let context_a = UserContext::new(
            PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()),
            SERVICE_ENTRY,
            SERVICE_STACK_TOP,
        )
        .unwrap_or_else(|_| halt());
        let context_b = UserContext::new(
            PhysAddr::new(SERVICE_B_ROOT).unwrap_or_else(|_| halt()),
            SERVICE_SENDER_ENTRY,
            SERVICE_STACK_TOP,
        )
        .unwrap_or_else(|_| halt());
        let mut runtime = TaskRuntime::<2, 1>::new(1_000, 1).unwrap_or_else(|_| halt());
        let receiver = runtime
            .create(Priority::NORMAL, context_a)
            .unwrap_or_else(|_| halt());
        let sender = runtime
            .create(Priority::NORMAL, context_b)
            .unwrap_or_else(|_| halt());
        let mut supervisor = ServiceSupervisor::<2>::new();
        let service_a = supervisor
            .register_with_policy(ObjectId(0xa0), receiver, RestartPolicy::ONCE_IMMEDIATE)
            .unwrap_or_else(|_| halt());
        let service_b = supervisor
            .register_with_policy(ObjectId(0xa1), sender, RestartPolicy::ONCE_IMMEDIATE)
            .unwrap_or_else(|_| halt());
        let mut management = CapabilitySpace::<2>::new();
        let control_a = management
            .insert(ObjectId(0xa0), Rights::CONTROL)
            .unwrap_or_else(|_| halt());
        let control_b = management
            .insert(ObjectId(0xa1), Rights::CONTROL)
            .unwrap_or_else(|_| halt());
        let endpoint_object = ObjectId(0x91);
        let capability = runtime
            .capabilities_mut(sender)
            .and_then(|space| {
                space
                    .insert(endpoint_object, Rights::SIGNAL)
                    .map_err(|_| mrml_kernel::TaskRuntimeError::IntegrityFailure)
            })
            .unwrap_or_else(|_| halt());
        let sender_context = runtime.context_mut(sender).unwrap_or_else(|_| halt());
        sender_context.r13 = capability.token();
        sender_context.r14 = receiver.token();
        if !matches!(runtime.start(), ScheduleOutcome::Switch { from: None, to } if to == receiver)
        {
            halt();
        }
        core::ptr::addr_of_mut!(SERVICE_ENDPOINT).write(Some(Endpoint::new(endpoint_object)));
        core::ptr::addr_of_mut!(SERVICE_SUPERVISOR).write(Some(supervisor));
        core::ptr::addr_of_mut!(SERVICE_MANAGEMENT).write(Some(management));
        core::ptr::addr_of_mut!(SERVICE_CONTROLS).write(Some([control_a, control_b]));
        core::ptr::addr_of_mut!(SERVICE_IDS).write(Some([service_a, service_b]));
        let runtime_pointer = core::ptr::addr_of_mut!(SERVICE_RUNTIME);
        runtime_pointer.write(Some(runtime));
        let context = (*runtime_pointer)
            .as_ref()
            .and_then(|runtime| runtime.context(receiver).ok())
            .map(|context| context as *const UserContext)
            .unwrap_or_else(|| halt());
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
    #[cfg(feature = "service-preemption-probe")]
    unsafe {
        asm!("cli", options(nomem, nostack));
        #[cfg(feature = "uefi-service-preemption-probe")]
        let (service_root, service_entry, service_stack_top) = {
            let service = handoff.service().unwrap_or_else(|| halt());
            let image_length = service
                .image_pages()
                .checked_mul(mrml_kernel::PAGE_SIZE)
                .and_then(|bytes| usize::try_from(bytes).ok())
                .unwrap_or_else(|| halt());
            let image = core::slice::from_raw_parts(
                service.image_physical().get() as *const u8,
                image_length,
            );
            let plan = ServiceAddressSpace::<40>::from_handoff(service, image, &[])
                .unwrap_or_else(|_| halt());
            let store =
                PreallocatedPageTableStore::new(service.table_physical(), service.table_pages())
                    .unwrap_or_else(|_| halt());
            let tables = plan
                .build_page_tables_with_current_kernel(store)
                .unwrap_or_else(|_| halt());
            (tables.root(), plan.entry(), plan.stack_top())
        };
        #[cfg(not(feature = "uefi-service-preemption-probe"))]
        let (service_root, service_entry, service_stack_top) = (
            PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()),
            SERVICE_ENTRY,
            SERVICE_STACK_TOP,
        );
        let first_context = UserContext::new(service_root, service_entry, service_stack_top)
            .unwrap_or_else(|_| halt());
        let second_context = UserContext::new(
            service_root,
            service_entry.checked_add(0x80).unwrap_or_else(|| halt()),
            service_stack_top,
        )
        .unwrap_or_else(|_| halt());
        let mut runtime = TaskRuntime::<2, 0>::new(1_000, 1).unwrap_or_else(|_| halt());
        let first = runtime
            .create(Priority::NORMAL, first_context)
            .unwrap_or_else(|_| halt());
        runtime
            .create(Priority::NORMAL, second_context)
            .unwrap_or_else(|_| halt());
        if !matches!(runtime.start(), ScheduleOutcome::Switch { from: None, to } if to == first) {
            halt();
        }
        core::ptr::addr_of_mut!(PREEMPTION_RUNTIME).write(Some(runtime));
        LocalApicTimer::periodic(TIMER_VECTOR, 100_000, TimerDivide::By16)
            .and_then(|timer| timer.enable())
            .unwrap_or_else(|_| halt());
        let mut previous = LocalApicTimer::current_count();
        loop {
            let current = LocalApicTimer::current_count();
            if current > previous {
                break;
            }
            previous = current;
            core::hint::spin_loop();
        }
        asm!(
            "out dx, eax",
            in("dx") TIMER_READY_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
        let context = (*core::ptr::addr_of!(PREEMPTION_RUNTIME))
            .as_ref()
            .and_then(|runtime| runtime.context(first).ok())
            .map(|context| context as *const UserContext)
            .unwrap_or_else(|| halt());
        enter_service_preemption_context(&*context)
    }
    #[cfg(feature = "preemption-probe")]
    unsafe {
        asm!("cli", options(nomem, nostack));
        let user_stack: u64;
        let page_table: u64;
        asm!("mov {}, rsp", out(reg) user_stack, options(nomem, nostack, preserves_flags));
        asm!("mov {}, cr3", out(reg) page_table, options(nomem, nostack, preserves_flags));
        let root = PhysAddr::new(page_table).unwrap_or_else(|_| halt());
        let first_context = UserContext::new(
            root,
            mrml_preemption_spin as *const () as usize as u64,
            user_stack & !0xf,
        )
        .unwrap_or_else(|_| halt());
        let second_context = UserContext::new(
            root,
            mrml_preemption_replacement as *const () as usize as u64,
            (user_stack & !0xf)
                .checked_sub(4096)
                .unwrap_or_else(|| halt()),
        )
        .unwrap_or_else(|_| halt());
        let mut runtime = TaskRuntime::<2, 0>::new(1_000, 1).unwrap_or_else(|_| halt());
        let first = runtime
            .create(Priority::NORMAL, first_context)
            .unwrap_or_else(|_| halt());
        runtime
            .create(Priority::NORMAL, second_context)
            .unwrap_or_else(|_| halt());
        if !matches!(runtime.start(), ScheduleOutcome::Switch { from: None, to } if to == first) {
            halt();
        }
        core::ptr::addr_of_mut!(PREEMPTION_RUNTIME).write(Some(runtime));
        LocalApicTimer::periodic(TIMER_VECTOR, 100_000, TimerDivide::By16)
            .and_then(|timer| timer.enable())
            .unwrap_or_else(|_| halt());
        let mut previous = LocalApicTimer::current_count();
        loop {
            let current = LocalApicTimer::current_count();
            if current > previous {
                break;
            }
            previous = current;
            core::hint::spin_loop();
        }
        asm!(
            "out dx, eax",
            in("dx") TIMER_READY_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
        let context = (*core::ptr::addr_of!(PREEMPTION_RUNTIME))
            .as_ref()
            .and_then(|runtime| runtime.context(first).ok())
            .map(|context| context as *const UserContext)
            .unwrap_or_else(|| halt());
        enter_user_context(&*context)
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
        let timer = LocalApicTimer::periodic(TIMER_VECTOR, 100_000, TimerDivide::By16)
            .unwrap_or_else(|_| halt());
        timer.enable().unwrap_or_else(|_| halt());
        asm!(
            "out dx, eax",
            in("dx") TIMER_READY_PORT,
            in("eax") 1u32,
            options(nomem, nostack)
        );
        let mut previous = LocalApicTimer::current_count();
        loop {
            let current = LocalApicTimer::current_count();
            if current > previous {
                break;
            }
            previous = current;
            core::hint::spin_loop();
        }
        asm!(
            "out dx, eax",
            "sti",
            "2:",
            "pause",
            "jmp 2b",
            in("dx") TIMER_READY_PORT,
            in("eax") 2u32,
            options(noreturn, nomem, nostack)
        );
    }
    #[cfg(all(
        not(feature = "timer-probe"),
        not(feature = "preemption-probe"),
        not(feature = "service-preemption-probe"),
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
unsafe fn enter_service_probe_context(context: &UserContext) -> ! {
    let transition_stack = unsafe {
        CpuDescriptorState::entry_stack_top_from(core::ptr::addr_of!(CPU0_DESCRIPTORS))
            .unwrap_or_else(|_| halt())
    };
    unsafe { enter_user_context_on_stack(context, transition_stack) }
}

#[cfg(feature = "service-probe")]
unsafe fn restart_service_pair(
    runtime: &mut TaskRuntime<2, 1>,
    supervisor: &mut ServiceSupervisor<2>,
) -> ! {
    let management = unsafe {
        (*core::ptr::addr_of!(SERVICE_MANAGEMENT))
            .as_ref()
            .unwrap_or_else(|| halt())
    };
    let controls = unsafe { (*core::ptr::addr_of!(SERVICE_CONTROLS)).unwrap_or_else(|| halt()) };
    let retired = unsafe { (*core::ptr::addr_of!(SERVICE_IDS)).unwrap_or_else(|| halt()) };
    let receiver_context = UserContext::new(
        PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()),
        SERVICE_ENTRY,
        SERVICE_STACK_TOP,
    )
    .unwrap_or_else(|_| halt());
    let sender_context = UserContext::new(
        PhysAddr::new(SERVICE_B_ROOT).unwrap_or_else(|_| halt()),
        SERVICE_SENDER_ENTRY,
        SERVICE_STACK_TOP,
    )
    .unwrap_or_else(|_| halt());
    // Recreate the sender first so the scheduler's retained round-robin cursor
    // selects the receiver slot. This preserves the proof that an empty receive
    // blocks before the replacement sender is allowed to run.
    let sender_service = supervisor
        .restart(
            retired[1],
            management,
            controls[1],
            runtime,
            Priority::NORMAL,
            sender_context,
        )
        .unwrap_or_else(|_| halt());
    unsafe { service_restart_stage(0x31) };
    let receiver_service = supervisor
        .restart(
            retired[0],
            management,
            controls[0],
            runtime,
            Priority::NORMAL,
            receiver_context,
        )
        .unwrap_or_else(|_| halt());
    unsafe { service_restart_stage(0x32) };
    let receiver = supervisor
        .task(receiver_service)
        .ok()
        .flatten()
        .unwrap_or_else(|| halt());
    let sender = supervisor
        .task(sender_service)
        .ok()
        .flatten()
        .unwrap_or_else(|| halt());
    unsafe { service_restart_stage(0x34) };
    if unsafe { (*core::ptr::addr_of!(SERVICE_ENDPOINT)).as_ref() }.is_none() {
        halt();
    }
    unsafe { service_restart_stage(0x35) };
    let capability = runtime
        .capabilities_mut(sender)
        .and_then(|space| {
            space
                .insert(ObjectId(0x91), Rights::SIGNAL)
                .map_err(|_| mrml_kernel::TaskRuntimeError::IntegrityFailure)
        })
        .unwrap_or_else(|_| halt());
    unsafe { service_restart_stage(0x36) };
    let sender_context = runtime.context_mut(sender).unwrap_or_else(|_| halt());
    sender_context.r13 = capability.token();
    sender_context.r14 = receiver.token();
    unsafe { service_restart_stage(0x37) };
    if !matches!(runtime.start(), ScheduleOutcome::Switch { from: None, to } if to == receiver) {
        halt();
    }
    unsafe {
        core::ptr::addr_of_mut!(SERVICE_IDS).write(Some([receiver_service, sender_service]));
        core::ptr::addr_of_mut!(SERVICE_RESTARTED).write(true);
    }
    unsafe { service_restart_stage(0x33) };
    unsafe { enter_service_task(runtime, receiver) }
}

#[cfg(feature = "service-probe")]
unsafe fn service_restart_stage(stage: u32) {
    unsafe {
        asm!(
            "out dx, eax",
            in("dx") SERVICE_PROBE_PORT,
            in("eax") stage,
            options(nomem, nostack)
        )
    }
}

#[cfg(feature = "service-probe")]
unsafe fn verify_restart_budget_exhausted(
    runtime: &mut TaskRuntime<2, 1>,
    supervisor: &mut ServiceSupervisor<2>,
) {
    let management = unsafe {
        (*core::ptr::addr_of!(SERVICE_MANAGEMENT))
            .as_ref()
            .unwrap_or_else(|| halt())
    };
    let controls = unsafe { (*core::ptr::addr_of!(SERVICE_CONTROLS)).unwrap_or_else(|| halt()) };
    let services = unsafe { (*core::ptr::addr_of!(SERVICE_IDS)).unwrap_or_else(|| halt()) };
    let context = UserContext::new(
        PhysAddr::new(SERVICE_ROOT).unwrap_or_else(|_| halt()),
        SERVICE_ENTRY,
        SERVICE_STACK_TOP,
    )
    .unwrap_or_else(|_| halt());
    if supervisor.restart(
        services[0],
        management,
        controls[0],
        runtime,
        Priority::NORMAL,
        context,
    ) != Err(ServiceError::RestartLimit)
    {
        halt();
    }
}

#[cfg(feature = "service-preemption-probe")]
unsafe fn enter_service_preemption_context(context: &UserContext) -> ! {
    let transition_stack = unsafe {
        CpuDescriptorState::entry_stack_top_from(core::ptr::addr_of!(CPU0_DESCRIPTORS))
            .unwrap_or_else(|_| halt())
    };
    unsafe { enter_user_context_on_stack(context, transition_stack) }
}

#[cfg(feature = "service-probe")]
unsafe fn enter_service_task(runtime: &TaskRuntime<2, 1>, task: mrml_kernel::TaskId) -> ! {
    let context = runtime
        .context(task)
        .map(|context| context as *const UserContext)
        .unwrap_or_else(|_| halt());
    unsafe { enter_service_probe_context(&*context) }
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

unsafe fn install_descriptor_tables(kernel_stack: u64, double_fault_stack: u64) {
    let state = unsafe {
        core::ptr::addr_of_mut!(CPU0_DESCRIPTORS)
            .as_mut()
            .unwrap_or_else(|| halt())
    };
    unsafe { install_descriptor_state(state, kernel_stack, double_fault_stack) };
}

unsafe fn install_descriptor_state(
    state: &mut CpuDescriptorState,
    kernel_stack: u64,
    double_fault_stack: u64,
) {
    let fallback = mrml_exception_fail_stop as *const () as usize as u64;
    let handlers = unsafe { &mrml_exception_table };
    if unsafe { state.install(kernel_stack, double_fault_stack, handlers, fallback) }.is_err() {
        halt();
    }
    #[cfg(any(
        feature = "user-probe",
        feature = "service-probe",
        feature = "smp-service-migration-probe"
    ))]
    if unsafe { state.install_user_call(mrml_user_call as *const () as usize as u64) }.is_err() {
        halt();
    }
    #[cfg(any(
        feature = "timer-probe",
        feature = "preemption-probe",
        feature = "service-preemption-probe",
        feature = "smp-scheduler-probe",
        feature = "smp-periodic-balance-probe"
    ))]
    if unsafe {
        state.install_external(
            TIMER_VECTOR,
            mrml_timer_interrupt as *const () as usize as u64,
        )
    }
    .is_err()
    {
        halt();
    }
    #[cfg(feature = "smp-ipi-probe")]
    if unsafe {
        state.install_external(
            RESCHEDULE_VECTOR,
            mrml_reschedule_interrupt as *const () as usize as u64,
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
