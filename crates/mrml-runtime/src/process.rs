use crate::{Text, Vector};
use core::fmt;

pub fn process_id() -> u32 {
    #[cfg(windows)]
    {
        mrml_windows::process_id()
    }
    #[cfg(unix)]
    {
        mrml_linux::process_id()
    }
}

pub fn temporary_directory() -> Text {
    #[cfg(windows)]
    {
        crate::environment_variable("TEMP")
            .or_else(|| crate::environment_variable("TMP"))
            .unwrap_or_else(|| ".".into())
    }
    #[cfg(unix)]
    {
        crate::environment_variable("TMPDIR").unwrap_or_else(|| "/tmp".into())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessError {
    InvalidArgument,
    SpawnFailed,
    OutputFailed,
}

impl fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidArgument => "invalid process argument",
            Self::SpawnFailed => "failed to spawn process",
            Self::OutputFailed => "failed to capture process output",
        })
    }
}

impl core::error::Error for ProcessError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub const fn success(self) -> bool { self.0 == 0 }
    pub const fn code(self) -> i32 { self.0 }
}

pub struct Output {
    pub status: ExitStatus,
    pub stdout: Vector<u8>,
    pub stderr: Vector<u8>,
}

pub struct Command {
    program: Text,
    arguments: Vector<Text>,
    current_directory: Option<Text>,
}

impl Command {
    pub fn new(program: &str) -> Self {
        Self { program: program.into(), arguments: Vector::new(), current_directory: None }
    }

    pub fn arg(&mut self, argument: &str) -> &mut Self {
        self.arguments.push(argument.into());
        self
    }

    pub fn args<'a>(&mut self, arguments: impl IntoIterator<Item = &'a str>) -> &mut Self {
        for argument in arguments { self.arg(argument); }
        self
    }

    pub fn current_dir(&mut self, directory: &str) -> &mut Self {
        self.current_directory = Some(directory.into());
        self
    }

    pub fn output(&mut self) -> Result<Output, ProcessError> {
        #[cfg(windows)]
        let mut child = {
            let mut command_line = Vector::new();
            append_windows_argument(&mut command_line, &self.program);
            for argument in &self.arguments {
                command_line.push(' ' as u16);
                append_windows_argument(&mut command_line, argument);
            }
            command_line.push(0);
            let current_directory = self.current_directory.as_ref().map(|directory| {
                let mut encoded = Vector::new();
                encoded.extend(directory.encode_utf16());
                encoded.push(0);
                encoded
            });
            mrml_windows::NativeChild::spawn_captured(
                &mut command_line,
                current_directory.as_deref(),
            )
            .ok_or(ProcessError::SpawnFailed)?
        };
        #[cfg(unix)]
        let mut child = {
            let mut encoded = Vector::new();
            encoded.push(encode_c_string(&self.program)?);
            for argument in &self.arguments { encoded.push(encode_c_string(argument)?); }
            let mut pointers: Vector<*const i8> = encoded
                .iter()
                .map(|argument| argument.as_ptr().cast())
                .collect();
            pointers.push(core::ptr::null());
            let program = core::ffi::CStr::from_bytes_with_nul(&encoded[0])
                .map_err(|_| ProcessError::InvalidArgument)?;
            let directory = self
                .current_directory
                .as_ref()
                .map(|directory| encode_c_string(directory))
                .transpose()?;
            let directory = directory.as_ref().map(|directory| {
                core::ffi::CStr::from_bytes_with_nul(directory).expect("validated C path")
            });
            mrml_linux::NativeChild::spawn_captured(program, &pointers, directory)
                .ok_or(ProcessError::SpawnFailed)?
        };

        let mut stdout = Vector::new();
        let mut stderr = Vector::new();
        let mut buffer = [0u8; 8192];
        loop {
            let stdout_read = child.read_stdout(&mut buffer);
            stdout.try_extend_from_slice(&buffer[..stdout_read]).map_err(|_| ProcessError::OutputFailed)?;
            let stderr_read = child.read_stderr(&mut buffer);
            stderr.try_extend_from_slice(&buffer[..stderr_read]).map_err(|_| ProcessError::OutputFailed)?;
            if let Some(code) = child.try_wait() {
                loop {
                    let out = child.read_stdout(&mut buffer);
                    stdout.try_extend_from_slice(&buffer[..out]).map_err(|_| ProcessError::OutputFailed)?;
                    let err = child.read_stderr(&mut buffer);
                    stderr.try_extend_from_slice(&buffer[..err]).map_err(|_| ProcessError::OutputFailed)?;
                    if out == 0 && err == 0 { break; }
                }
                return Ok(Output { status: ExitStatus(code), stdout, stderr });
            }
            if stdout_read == 0 && stderr_read == 0 {
                #[cfg(windows)]
                mrml_windows::sleep_millis(1);
                #[cfg(unix)]
                mrml_linux::sleep_millis(1);
            }
        }
    }
}

#[cfg(unix)]
fn encode_c_string(value: &str) -> Result<Vector<u8>, ProcessError> {
    if value.as_bytes().contains(&0) { return Err(ProcessError::InvalidArgument); }
    let mut encoded = Vector::with_capacity(value.len() + 1).map_err(|_| ProcessError::InvalidArgument)?;
    encoded.try_extend_from_slice(value.as_bytes()).map_err(|_| ProcessError::InvalidArgument)?;
    encoded.push(0);
    Ok(encoded)
}

#[cfg(windows)]
fn append_windows_argument(output: &mut Vector<u16>, argument: &str) {
    output.push('"' as u16);
    let mut backslashes = 0usize;
    for unit in argument.encode_utf16() {
        if unit == '\\' as u16 {
            backslashes += 1;
        } else {
            if unit == '"' as u16 {
                for _ in 0..=backslashes { output.push('\\' as u16); }
            }
            for _ in 0..backslashes { output.push('\\' as u16); }
            backslashes = 0;
            output.push(unit);
        }
    }
    for _ in 0..backslashes.saturating_mul(2) { output.push('\\' as u16); }
    output.push('"' as u16);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_live_process_and_temporary_directory() {
        assert_eq!(process_id(), std::process::id());
        assert!(crate::path_is_directory(&temporary_directory()));
    }


    #[test]
    fn captures_stdout_stderr_exit_status_and_working_directory() {
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("powershell.exe");
            command.args(["-NoProfile", "-Command", "[Console]::Out.Write('out λ'); [Console]::Error.Write('err 星'); exit 7"]);
            command
        };
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", "printf 'out λ'; printf 'err 星' >&2; exit 7"]);
            command
        };
        command.current_dir(&temporary_directory());
        let output = command.output().unwrap();
        assert_eq!(output.status.code(), 7);
        assert_eq!(core::str::from_utf8(&output.stdout).unwrap(), "out λ");
        assert_eq!(core::str::from_utf8(&output.stderr).unwrap(), "err 星");
    }
}
