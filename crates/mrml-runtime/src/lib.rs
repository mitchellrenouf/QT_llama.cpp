#![no_std]
#![feature(coerce_unsized, pin_coerce_unsized_trait, unsize)]

#[cfg(test)]
extern crate std;

mod file;
mod map;
mod owned;
mod sync;
mod text;
mod thread;
mod time;
mod vector;

pub use file::{File, FileError};
pub use map::OrderedMap;
pub use owned::Owned;
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use thread::{available_parallelism, spawn_detached, yield_now};
pub use time::Instant;
pub use vector::{TryReserveError, Vector};
