#![no_std]

//! Security primitives shared by the MRML microkernel and its host VMMs.
//! This crate deliberately contains policy data structures, not privileged
//! machine code. Architecture and hypervisor implementations sit behind these
//! narrow types so their unsafe boundaries can be reviewed independently.

pub mod arch;
mod artifact;
mod boot;
mod capability;
mod early;
mod framebuffer;
mod grant;
mod handoff;
mod ipc;
mod memory;
mod pe;
mod platform;
mod policy;
mod scheduler;
mod service_lifecycle;
mod syscall;
mod task_runtime;
mod virtual_gpu;
mod vm;
mod vm_dispatch;
mod vm_interrupt;

pub use artifact::{
    ArtifactError, ArtifactKind, BOOTSTRAP_STATE_BYTES, BootstrapState, ExecutableArtifactError,
    MAX_KERNEL_IMAGE_BYTES, MAX_SERVICE_IMAGE_BYTES, MAX_VM_IMAGE_BYTES, MonotonicStateStore,
    RELEASE_MANIFEST_BYTES, ReleaseManifest, SIGNED_ARTIFACT_HEADER_BYTES,
    SIGNED_ARTIFACT_OVERHEAD_BYTES, SignedArtifact, TrustRoot, VerifiedArtifact,
    VerifiedExecutable, VerifiedRelease, artifact_statement, executable_image_limit,
};
pub use boot::{BootEvidence, BootPolicy, BootValidationError};
pub use capability::{Capability, CapabilityError, CapabilitySpace, ObjectId, Rights};
pub use early::{EarlyKernelContext, EarlyKernelError};
pub use framebuffer::{Color, FramebufferError, FramebufferInfo, FramebufferSurface, PixelFormat};
pub use grant::{DirectoryGrant, GrantError, GrantMode, MAX_HOST_PATH};
pub use handoff::{
    BootHandoff, HANDOFF_HEADER_BYTES, HANDOFF_REGION_BYTES, HandoffError, MAX_HANDOFF_BYTES,
    MAX_HANDOFF_MADT_BYTES, MAX_HANDOFF_REGIONS, encode_handoff, encode_handoff_with_madt,
    encode_handoff_with_smp,
};
pub use ipc::{
    Endpoint, IpcError, MAX_CAPABILITIES, MAX_INLINE_PAYLOAD, MAX_WIRE_MESSAGE, Message, Receiver,
    WIRE_HEADER_LENGTH,
};
pub use memory::{
    FrameAllocator, MemoryError, MemoryKind, MemoryMap, MemoryRegion, PAGE_SIZE, PhysAddr,
};
pub use pe::{
    MAX_PE_IMAGE_BYTES, MAX_PE_RELOCATIONS, MAX_PE_SECTIONS, PeAllocatedRegion, PeAllocationError,
    PeError, PeImage, PeLoadRegion, PeSection,
};
pub use platform::{Architecture, Hypervisor, IsolationClass, VmRole};
pub use policy::{
    DeviceAddress, DeviceGrant, HostDevice, IommuTopology, MAX_VM_NAME, PolicyError, SystemPolicy,
    VmName, VmPolicy,
};
pub use scheduler::{
    BalanceOutcome, DetachedTask, KernelScheduleError, KernelScheduler, MigrationMailbox,
    MigrationMailboxError, Priority, ScheduleOutcome, Scheduler, SchedulerError, SchedulerLoad,
    TaskAttachError, TaskId, TaskMigration, TaskState,
};
pub use service_lifecycle::{
    RestartPolicy, ServiceError, ServiceFault, ServiceId, ServiceRetirement, ServiceState,
    ServiceSupervisor,
};
pub use syscall::{
    MAX_SYSCALL_INLINE_PAYLOAD, SyscallError, SyscallRequest, UserCallFrame, X86_USER_CALL_VECTOR,
};
pub use task_runtime::{
    DetachedTaskDomain, FaultRetirement, TASK_INBOX_MESSAGES, TaskRuntime, TaskRuntimeError,
    TaskTermination,
};
pub use virtual_gpu::{
    BatchedDispatch, BufferAccess, BufferId, BufferMode, ControlBufferId, ControlBufferTable,
    Dispatch, DispatchId, DispatchTable, GPU_DOORBELL_PORT, GPU_QUEUE_MESSAGE_BYTES,
    GpuBatchExecutor, GpuCommandRing, GpuCommandServiceError, GpuCompletion, GpuCompletionReceiver,
    GpuCompletionRing, GpuCompletionSender, GpuCompletionStatus, GpuDispatchBatch, GpuError,
    GpuGuestCommandPublisher, GpuHostBackend, GpuLifecycleError, GpuQueueIdentity,
    GpuQueueReceiver, GpuQueueSender, GpuResourceBackend, GpuResourceOutcome, GpuResourceResponse,
    GpuResourceResponseReceiver, GpuResourceResponseSender, GpuRingConsumer, GpuRingProducer,
    GpuRingTicket, GpuSharedQueueLayout, GpuSharedRingIndices, GpuSubmitError, GpuVmmMemory,
    GpuVmmQueueBridge, GpuVmmQueueError, KernelId, MAX_ACTIVE_EXPERTS, MAX_BATCH_DISPATCHES,
    MAX_DISPATCH_BUFFERS, MAX_DISPATCH_SCALARS, MAX_GPU_BENCHMARK_ELEMENTS,
    MAX_GPU_BENCHMARK_ITERATIONS, MAX_GPU_CONTROL_BYTES, MAX_GPU_QUEUE_SLOTS, MediatedGpuExecutor,
    PreparedGpuBatch, PreparedGpuDispatch, ResourceCommand, ScalarArg, ScalarKind,
    ValidatedExpertSelection, ValidatedGpuBatch, ValidatedKernelLaunch, ValidatedMoeKernelLaunch,
    VerifiedGpuKernelBundle, VirtualGpuSession, process_gpu_resource_command,
    process_gpu_resource_command_with_response, submit_gpu_batch,
    submit_gpu_batch_to_completion_ring, submit_gpu_batch_with_completions,
    submit_gpu_control_batch, verify_gpu_kernel_bundle,
};
pub use vm::{
    GuestAccess, GuestMappingId, GuestMemory, GuestRegion, HYPERCALL_BYTES, Hypercall,
    HypercallOperation, HypercallRouter, MAX_GUEST_REGIONS, RoutedHypercall, VmBackend, VmError,
    VmExit, VmId, VmRunError, VmState, VmTable, decode_hypercall_exit,
};
pub use vm_dispatch::{
    ExitDisposition, IoDirection, IoPortPolicy, IoPortRule, MAX_IO_PORT_RULES, dispatch_vm_exit,
    run_vm_once,
};
pub use vm_interrupt::{InterruptPolicy, inject_vm_interrupt};
