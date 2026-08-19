#![no_std]
#![feature(coerce_unsized, pin_coerce_unsized_trait, unsize)]

#[cfg(test)]
extern crate std;

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
pub use time::Instant;
pub use vector::{TryReserveError, Vector};

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
