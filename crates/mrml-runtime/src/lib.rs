#![no_std]

#[cfg(test)]
extern crate std;

mod map;
mod sync;
mod text;
mod vector;

pub use map::OrderedMap;
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use vector::{TryReserveError, Vector};
