#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod anyhow;
pub mod device;
#[cfg(feature = "std")]
pub mod engine;
pub mod execution_plan;
#[cfg(feature = "std")]
pub mod gguf;
pub mod graph;
pub mod kv_cache;
#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod model;
pub mod ops;
#[cfg(feature = "std")]
pub mod parallel;
#[cfg(not(feature = "std"))]
#[path = "parallel_nostd.rs"]
pub mod parallel;
pub mod quant;
#[cfg(feature = "std")]
pub mod speculative;
#[doc(hidden)]
pub mod sync;
pub mod tensor;
pub mod types;

#[cfg(all(feature = "std", feature = "cuda"))]
pub mod cuda;

// Re-export common types
pub use device::{DeviceManager, DeviceType};
#[cfg(feature = "std")]
pub use engine::MrmlEngine;
#[cfg(feature = "std")]
pub use gguf::{GgufFile, GgufTensorInfo, GgufValue};
pub use graph::{CGraph, Node, OpType};
pub use kv_cache::{KvCacheFormat, KvCacheManager, KvCacheRow, LayerKvCache, PrefixCache};
#[cfg(feature = "std")]
pub use model::{ModelConfig, MrmlModel};
#[cfg(feature = "std")]
pub use speculative::SpeculativeDecoder;
pub use tensor::{Tensor, TensorStorage};
pub use types::{DType, Shape, Strides};
