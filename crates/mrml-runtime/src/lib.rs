#![no_std]

#[cfg(test)]
extern crate std;

mod file;
mod map;
mod sync;
mod text;
mod thread;
mod time;
mod vector;

pub use file::{File, FileError};
pub use map::OrderedMap;
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use thread::{available_parallelism, spawn_detached, yield_now};
pub use time::Instant;
pub use vector::{TryReserveError, Vector};
