#![no_std]

mod abi;
#[cfg(target_os = "windows")]
mod native;

pub use abi::{GuestRange, MapPermissions, WHP_EXIT_CONTEXT_BYTES, WhpError, decode_exit_context};
#[cfg(target_os = "windows")]
pub use native::{PreparedWhpGuest, WhpLaunchLayout, WhpPageWalk, WhpSystem};
