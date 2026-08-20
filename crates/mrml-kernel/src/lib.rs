#![no_std]

//! Security primitives shared by the MRML microkernel and its host VMMs.
//! This crate deliberately contains policy data structures, not privileged
//! machine code. Architecture and hypervisor implementations sit behind these
//! narrow types so their unsafe boundaries can be reviewed independently.

mod capability;
mod grant;
mod ipc;
mod platform;
mod policy;

pub use capability::{Capability, CapabilityError, CapabilitySpace, ObjectId, Rights};
pub use grant::{DirectoryGrant, GrantError, GrantMode, MAX_HOST_PATH};
pub use ipc::{
    Endpoint, IpcError, MAX_CAPABILITIES, MAX_INLINE_PAYLOAD, MAX_WIRE_MESSAGE, Message, Receiver,
    WIRE_HEADER_LENGTH,
};
pub use platform::{Architecture, Hypervisor, IsolationClass, VmRole};
pub use policy::{DeviceAddress, DeviceGrant, MAX_VM_NAME, PolicyError, VmName, VmPolicy};
