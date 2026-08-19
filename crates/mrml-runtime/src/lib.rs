#![no_std]

#[cfg(test)]
extern crate std;

mod map;
mod text;
mod vector;

pub use map::OrderedMap;
pub use text::Text;
pub use vector::{TryReserveError, Vector};
