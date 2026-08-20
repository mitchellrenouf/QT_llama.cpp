#![no_std]

//! Security primitives shared by the MRML microkernel and its host VMMs.
//! This crate deliberately contains policy data structures, not privileged
//! machine code. Architecture and hypervisor implementations sit behind these
//! narrow types so their unsafe boundaries can be reviewed independently.

mod capability;
mod grant;
mod platform;

pub use capability::{Capability, CapabilityError, CapabilitySpace, ObjectId, Rights};
pub use grant::{DirectoryGrant, GrantError, GrantMode, MAX_HOST_PATH};
pub use platform::{Architecture, Hypervisor, IsolationClass, VmRole};
