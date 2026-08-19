#![no_std]
#![feature(coerce_unsized, pin_coerce_unsized_trait, unsize)]

#[cfg(test)]
extern crate std;

mod channel;
mod args;
mod env;
mod file;
mod map;
mod owned;
mod stdin;
mod sync;
mod text;
mod thread;
mod time;
mod vector;

pub use channel::{Receiver, RecvError, SendError, Sender, blocking_channel, sync_channel};
pub use args::command_arguments;
pub use env::environment_variable;
pub use file::{File, FileError};
pub use map::OrderedMap;
pub use owned::Owned;
pub use stdin::{StdinError, read_stdin_line, read_stdin_to_end};
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use thread::{available_parallelism, spawn_detached, yield_now};
pub use time::Instant;
pub use vector::{TryReserveError, Vector};
