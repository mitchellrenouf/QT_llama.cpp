use core::fmt::{self, Write};

struct NativeOutput {
    stderr: bool,
}

impl Write for NativeOutput {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        #[cfg(windows)]
        let written = if self.stderr {
            mrml_windows::write_stderr_text(text)
        } else {
            mrml_windows::write_stdout_text(text)
        };
        #[cfg(unix)]
        let written = if self.stderr {
            mrml_linux::write_stderr(text.as_bytes())
        } else {
            mrml_linux::write_stdout(text.as_bytes())
        };
        written.then_some(()).ok_or(fmt::Error)
    }
}

pub fn write_stdout(arguments: fmt::Arguments<'_>) -> fmt::Result {
    NativeOutput { stderr: false }.write_fmt(arguments)
}

pub fn write_stderr(arguments: fmt::Arguments<'_>) -> fmt::Result {
    NativeOutput { stderr: true }.write_fmt(arguments)
}

#[macro_export]
macro_rules! mrml_print {
    ($($argument:tt)*) => {{
        let _ = $crate::write_stdout(core::format_args!($($argument)*));
    }};
}

#[macro_export]
macro_rules! mrml_println {
    () => {{ $crate::mrml_print!("\n") }};
    ($($argument:tt)*) => {{
        let _ = $crate::write_stdout(core::format_args!($($argument)*));
        let _ = $crate::write_stdout(core::format_args!("\n"));
    }};
}

#[macro_export]
macro_rules! mrml_eprintln {
    () => {{ let _ = $crate::write_stderr(core::format_args!("\n")); }};
    ($($argument:tt)*) => {{
        let _ = $crate::write_stderr(core::format_args!($($argument)*));
        let _ = $crate::write_stderr(core::format_args!("\n"));
    }};
}
