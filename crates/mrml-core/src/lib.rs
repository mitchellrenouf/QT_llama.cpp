#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(test, not(feature = "std")))]
extern crate std;

pub mod modes;

#[cfg(feature = "runtime")]
pub mod agent;
#[cfg(feature = "runtime")]
pub mod client;
#[cfg(feature = "runtime")]
pub mod config;
#[cfg(feature = "runtime")]
pub mod hf;
#[cfg(feature = "runtime")]
pub mod rules;

#[cfg(feature = "runtime")]
pub use mrml_tools as tools;
#[cfg(feature = "runtime")]
pub use mrml_tools::{diff, encoding};
#[cfg(feature = "runtime")]
pub use mrml_tools::{fs_walk, markdown, platform};

#[cfg(feature = "runtime")]
pub use agent::MrmlAgent;
#[cfg(feature = "runtime")]
pub use config::Config;
pub use modes::{AgentMode, BackendChoice};

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_model::clear_cuda_allocation_pool();
}
