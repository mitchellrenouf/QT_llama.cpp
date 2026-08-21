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
    BootHandoff, HANDOFF_HEADER_BYTES, HANDOFF_REGION_BYTES, HandoffError, MAX_HANDOFF_REGIONS,
    encode_handoff,
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
pub use scheduler::{Priority, Scheduler, SchedulerError, TaskId, TaskState};
pub use virtual_gpu::{
    BufferAccess, BufferId, BufferMode, Dispatch, DispatchId, DispatchTable,
    GPU_QUEUE_MESSAGE_BYTES, GpuCommandRing, GpuError, GpuQueueReceiver, GpuQueueSender,
    GpuRingConsumer, GpuRingProducer, GpuRingTicket, KernelId, MAX_DISPATCH_BUFFERS,
    MAX_GPU_QUEUE_SLOTS, ResourceCommand, VirtualGpuSession,
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
