#![cfg_attr(not(feature = "std"), no_std)]

pub mod modes;

#[cfg(feature = "std")]
pub mod agent;
#[cfg(feature = "std")]
pub mod client;
#[cfg(feature = "std")]
pub mod config;
#[cfg(feature = "std")]
pub mod hf;
#[cfg(feature = "std")]
pub mod rules;

#[cfg(feature = "std")]
pub use mrml_tools as tools;
#[cfg(feature = "std")]
pub use mrml_tools::{diff, encoding};
#[cfg(feature = "std")]
pub use mrml_tools::{fs_walk, markdown, platform};

#[cfg(feature = "std")]
pub use agent::MrmlAgent;
#[cfg(feature = "std")]
pub use config::Config;
pub use modes::{AgentMode, BackendChoice};

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_model::clear_cuda_allocation_pool();
}
