#![no_std]

#[cfg(test)]
extern crate std;

mod map;
mod sync;
mod text;
mod time;
mod vector;

pub use map::OrderedMap;
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use time::Instant;
pub use vector::{TryReserveError, Vector};
