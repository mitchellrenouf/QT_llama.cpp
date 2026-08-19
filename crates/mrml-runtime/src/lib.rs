#![no_std]

#[cfg(test)]
extern crate std;

mod text;
mod vector;

pub use text::Text;
pub use vector::{TryReserveError, Vector};
