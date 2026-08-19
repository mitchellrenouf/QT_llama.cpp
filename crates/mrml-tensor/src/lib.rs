#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "cuda", feature(thread_local))]

#[cfg(test)]
extern crate std;

#[cfg(feature = "runtime")]
pub mod anyhow;
#[cfg(feature = "runtime")]
pub mod device;
#[cfg(feature = "runtime")]
pub mod engine;
pub mod execution_plan;
#[cfg(feature = "runtime")]
pub mod gguf;
#[cfg(feature = "runtime")]
pub mod graph;
#[cfg(feature = "runtime")]
pub mod kv_cache;
#[cfg(feature = "runtime")]
pub mod mmap;
#[cfg(feature = "runtime")]
pub mod model;
#[cfg(feature = "runtime")]
pub mod ops;
#[cfg(all(feature = "std", feature = "runtime"))]
pub mod parallel;
#[cfg(all(not(feature = "std"), feature = "runtime"))]
#[path = "parallel_nostd.rs"]
pub mod parallel;
pub mod quant;
#[cfg(feature = "runtime")]
pub mod speculative;
#[doc(hidden)]
pub mod sync;
#[cfg(feature = "runtime")]
pub mod tensor;
pub mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

// Re-export common types
#[cfg(feature = "runtime")]
pub use device::{DeviceManager, DeviceType};
#[cfg(feature = "runtime")]
pub use engine::MrmlEngine;
#[cfg(feature = "runtime")]
pub use gguf::{GgufFile, GgufTensorInfo, GgufValue};
#[cfg(feature = "runtime")]
pub use graph::{CGraph, Node, OpType};
#[cfg(feature = "runtime")]
pub use kv_cache::{KvCacheFormat, KvCacheManager, KvCacheRow, LayerKvCache, PrefixCache};
#[cfg(feature = "runtime")]
pub use model::{ModelConfig, MrmlModel};
#[cfg(feature = "runtime")]
pub use speculative::SpeculativeDecoder;
#[cfg(feature = "runtime")]
pub use tensor::{Tensor, TensorStorage};
pub use types::{DType, Shape, Strides};
