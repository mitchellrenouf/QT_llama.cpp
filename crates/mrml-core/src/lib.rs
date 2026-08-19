pub mod agent;
pub mod client;
pub mod config;
pub mod hf;
pub mod rules;

pub use mrml_tools as tools;
pub use mrml_tools::{diff, encoding, fs_walk, markdown, platform};

pub use agent::MrmlAgent;
pub use config::{AgentMode, BackendChoice, Config};

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_model::clear_cuda_allocation_pool();
}
