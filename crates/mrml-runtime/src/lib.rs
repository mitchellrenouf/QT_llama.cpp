#![no_std]
#![feature(coerce_unsized, pin_coerce_unsized_trait, unsize)]

mod channel;
mod args;
mod env;
mod file;
mod map;
mod net;
mod owned;
mod output;
mod process;
mod stdin;
mod sync;
mod text;
mod thread;
mod time;
mod vector;

pub use channel::{Receiver, RecvError, SendError, Sender, blocking_channel, sync_channel};
pub use args::command_arguments;
pub use env::environment_variable;
pub use file::{DirectoryEntry, File, FileError, canonical_path, create_dir_all, join_path, parent_path, path_exists, path_is_absolute, path_is_directory, path_is_file, read_directory, read_file, read_file_text, remove_dir_all, remove_file, rename_file, write_file};
pub use map::OrderedMap;
pub use net::{NetError, TcpListener, TcpStream};
pub use owned::Owned;
pub use output::{write_stderr, write_stdout};
pub use process::{Child, Command, ExitStatus, Output, PipedChild, ProcessError, process_id, temporary_directory};
pub use stdin::{StdinError, read_stdin_line, read_stdin_to_end};
pub use sync::{OnceCell, Shared, SpinMutex, SpinMutexGuard};
pub use text::Text;
pub use thread::{available_parallelism, spawn_detached, yield_now};
pub use time::{Instant, unix_time_seconds};
pub use vector::{TryReserveError, Vector};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RandomError;

impl core::fmt::Display for RandomError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("operating-system cryptographic random generator failed")
    }
}

impl core::error::Error for RandomError {}

pub fn fill_random(output: &mut [u8]) -> Result<(), RandomError> {
    #[cfg(windows)]
    let success = mrml_windows::fill_random(output);
    #[cfg(unix)]
    let success = mrml_linux::fill_random(output);
    success.then_some(()).ok_or(RandomError)
}

#[cfg(windows)]
pub fn visit_root_certificates(visitor: impl FnMut(&[u8]) -> bool) -> bool { mrml_windows::visit_root_certificates(visitor) }

#[cfg(test)]
mod random_tests {
    #[test]
    fn os_random_fills_independent_buffers() {
        let mut first = [0u8; 64];
        let mut second = [0u8; 64];
        super::fill_random(&mut first).unwrap();
        super::fill_random(&mut second).unwrap();
        assert_ne!(first, [0u8; 64]);
        assert_ne!(first, second);
    }
}

pub fn exit_process(status: i32) -> ! {
    #[cfg(windows)]
    {
        mrml_windows::exit_process(status)
    }
    #[cfg(unix)]
    {
        mrml_linux::exit_process(status)
    }
}

#[macro_export]
macro_rules! mrml_entrypoint {
    ($application_main:path) => {
        #[cfg(not(test))]
        #[panic_handler]
        fn mrml_panic(_information: &core::panic::PanicInfo<'_>) -> ! {
            let _ = $crate::write_stderr(format_args!("MRML terminated after a panic\n"));
            $crate::exit_process(101)
        }

        #[cfg(not(test))]
        #[unsafe(no_mangle)]
        pub extern "C" fn rust_eh_personality() {}

        #[cfg(not(test))]
        #[unsafe(no_mangle)]
        pub extern "C" fn main(
            _argument_count: core::ffi::c_int,
            _argument_values: *const *const core::ffi::c_char,
        ) -> core::ffi::c_int {
            match $application_main() {
                Ok(()) => 0,
                Err(error) => {
                    $crate::mrml_eprintln!("error: {error}");
                    1
                }
            }
        }
    };
}
