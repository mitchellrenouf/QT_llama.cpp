#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(any(feature = "alloc", test))]
extern crate alloc;
#[cfg(test)]
extern crate std;

#[cfg(feature = "alloc")]
pub mod anyhow;
#[cfg(feature = "alloc")]
pub mod device;
#[cfg(feature = "std")]
pub mod engine;
pub mod execution_plan;
#[cfg(feature = "std")]
pub mod gguf;
#[cfg(feature = "alloc")]
pub mod graph;
#[cfg(feature = "alloc")]
pub mod kv_cache;
#[cfg(feature = "std")]
pub mod mmap;
#[cfg(feature = "std")]
pub mod model;
#[cfg(feature = "alloc")]
pub mod ops;
#[cfg(all(feature = "std", feature = "alloc"))]
pub mod parallel;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
#[path = "parallel_nostd.rs"]
pub mod parallel;
pub mod quant;
#[cfg(feature = "std")]
pub mod speculative;
#[doc(hidden)]
pub mod sync;
#[cfg(feature = "alloc")]
pub mod tensor;
pub mod types;

#[cfg(all(feature = "std", feature = "cuda"))]
pub mod cuda;

// Re-export common types
#[cfg(feature = "alloc")]
pub use device::{DeviceManager, DeviceType};
#[cfg(feature = "std")]
pub use engine::MrmlEngine;
#[cfg(feature = "std")]
pub use gguf::{GgufFile, GgufTensorInfo, GgufValue};
#[cfg(feature = "alloc")]
pub use graph::{CGraph, Node, OpType};
#[cfg(feature = "alloc")]
pub use kv_cache::{KvCacheFormat, KvCacheManager, KvCacheRow, LayerKvCache, PrefixCache};
#[cfg(feature = "std")]
pub use model::{ModelConfig, MrmlModel};
#[cfg(feature = "std")]
pub use speculative::SpeculativeDecoder;
#[cfg(feature = "alloc")]
pub use tensor::{Tensor, TensorStorage};
pub use types::{DType, Shape, Strides};
