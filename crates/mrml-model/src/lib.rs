#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(test)]
extern crate std;

#[cfg(feature = "alloc")]
pub mod chat;
#[cfg(feature = "std")]
pub mod engine;
pub mod error;
pub mod portable;

#[cfg(feature = "alloc")]
pub use chat::*;
#[cfg(feature = "std")]
pub use engine::*;
#[cfg(feature = "std")]
pub use mrml_tensor;
pub use portable::*;

#[cfg(feature = "cuda")]
pub fn clear_cuda_allocation_pool() {
    mrml_tensor::cuda::clear_cuda_allocation_pool();
}
