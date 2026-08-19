#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod chat;
#[cfg(feature = "std")]
pub mod engine;
pub mod error;

pub use chat::*;
#[cfg(feature = "std")]
pub use engine::*;
#[cfg(feature = "std")]
pub use mrml_tensor;

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_tensor::cuda::clear_cuda_allocation_pool();
}
