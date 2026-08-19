#![no_std]

#[cfg(feature = "runtime")]
pub mod chat;
#[cfg(feature = "runtime")]
pub mod engine;
pub mod error;
pub mod portable;

#[cfg(feature = "runtime")]
pub use chat::*;
#[cfg(feature = "runtime")]
pub use engine::*;
#[cfg(feature = "runtime")]
pub use mrml_tensor;
pub use portable::*;

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_tensor::cuda::clear_cuda_allocation_pool();
}
