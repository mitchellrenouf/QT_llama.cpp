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
    OutputLimit,
    TimedOut,
}

impl fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidArgument => "invalid process argument",
            Self::SpawnFailed => "failed to spawn process",
            Self::OutputFailed => "failed to capture process output",
            Self::OutputLimit => "process output exceeded limit",
            Self::TimedOut => "process exceeded time limit",
        })
    }
}

impl core::error::Error for ProcessError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExitStatus(i32);

impl ExitStatus {
    pub const fn success(self) -> bool {
        self.0 == 0
    }
    pub const fn code(self) -> i32 {
        self.0
    }
}

#[derive(Debug)]
pub struct Output {
    pub status: ExitStatus,
    pub stdout: Vector<u8>,
    pub stderr: Vector<u8>,
}

pub struct Child {
    #[cfg(windows)]
    native: mrml_windows::NativeChild,
    #[cfg(unix)]
    native: mrml_linux::NativeChild,
}

impl Child {
    pub fn try_wait(&mut self) -> Option<ExitStatus> {
        self.native.try_wait().map(ExitStatus)
    }
    pub fn kill(&mut self) -> bool {
        self.native.kill()
    }
    pub fn wait(&mut self) -> Option<ExitStatus> {
        self.native.wait().map(ExitStatus)
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        if self.try_wait().is_none() {
            let _ = self.kill();
        }
        let _ = self.wait();
    }
}

pub struct PipedChild {
    #[cfg(windows)]
    native: mrml_windows::NativePipedChild,
    #[cfg(unix)]
    native: mrml_linux::NativePipedChild,
}

impl PipedChild {
    pub fn write_all(&mut self, bytes: &[u8]) -> Result<(), ProcessError> {
        self.native
            .write_stdin(bytes)
            .then_some(())
            .ok_or(ProcessError::OutputFailed)
    }

    pub fn read_line(&mut self) -> Result<Text, ProcessError> {
        let mut bytes = Vector::new();
        let mut byte = [0u8; 1];
        loop {
            match self.native.read_stdout(&mut byte) {
                Some(0) | None => return Err(ProcessError::OutputFailed),
                Some(_) if byte[0] == b'\n' => break,
                Some(_) => bytes
                    .try_push(byte[0])
                    .map_err(|_| ProcessError::OutputFailed)?,
            }
        }
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        let line = core::str::from_utf8(&bytes).map_err(|_| ProcessError::OutputFailed)?;
        Ok(line.into())
    }
}

pub struct Command {
    program: Text,
    arguments: Vector<Text>,
    current_directory: Option<Text>,
}

impl Command {
    pub fn new(program: &str) -> Self {
        Self {
            program: program.into(),
            arguments: Vector::new(),
            current_directory: None,
        }
    }

    pub fn arg(&mut self, argument: impl AsRef<str>) -> &mut Self {
        self.arguments.push(argument.as_ref().into());
        self
    }

    pub fn args<'a>(&mut self, arguments: impl IntoIterator<Item = &'a str>) -> &mut Self {
        for argument in arguments {
            self.arg(argument);
        }
        self
    }

    pub fn current_dir(&mut self, directory: &str) -> &mut Self {
        self.current_directory = Some(directory.into());
        self
    }

    pub fn spawn_detached(&mut self) -> Result<(), ProcessError> {
        #[cfg(windows)]
        {
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
            if mrml_windows::spawn_detached_process(&mut command_line, current_directory.as_deref())
            {
                Ok(())
            } else {
                Err(ProcessError::SpawnFailed)
            }
        }
        #[cfg(unix)]
        {
            let mut encoded = Vector::new();
            encoded.push(encode_c_string(&self.program)?);
            for argument in &self.arguments {
                encoded.push(encode_c_string(argument)?);
            }
            let mut pointers: Vector<*const i8> =
                encoded.iter().map(|value| value.as_ptr().cast()).collect();
            pointers.push(core::ptr::null());
            let program = core::ffi::CStr::from_bytes_with_nul(&encoded[0])
                .map_err(|_| ProcessError::InvalidArgument)?;
            let directory = self
                .current_directory
                .as_ref()
                .map(|value| encode_c_string(value))
                .transpose()?;
            let directory = directory.as_ref().map(|value| {
                core::ffi::CStr::from_bytes_with_nul(value).expect("validated C path")
            });
            if mrml_linux::spawn_detached_process(program, &pointers, directory) {
                Ok(())
            } else {
                Err(ProcessError::SpawnFailed)
            }
        }
    }

    pub fn spawn_silent(&mut self) -> Result<Child, ProcessError> {
        #[cfg(windows)]
        {
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
            mrml_windows::NativeChild::spawn_silent(&mut command_line, current_directory.as_deref())
                .map(|native| Child { native })
                .ok_or(ProcessError::SpawnFailed)
        }
        #[cfg(unix)]
        {
            let mut encoded = Vector::new();
            encoded.push(encode_c_string(&self.program)?);
            for argument in &self.arguments {
                encoded.push(encode_c_string(argument)?);
            }
            let mut pointers: Vector<*const i8> =
                encoded.iter().map(|value| value.as_ptr().cast()).collect();
            pointers.push(core::ptr::null());
            let program = core::ffi::CStr::from_bytes_with_nul(&encoded[0])
                .map_err(|_| ProcessError::InvalidArgument)?;
            let directory = self
                .current_directory
                .as_ref()
                .map(|value| encode_c_string(value))
                .transpose()?;
            let directory = directory.as_ref().map(|value| {
                core::ffi::CStr::from_bytes_with_nul(value).expect("validated C path")
            });
            mrml_linux::NativeChild::spawn_silent(program, &pointers, directory)
                .map(|native| Child { native })
                .ok_or(ProcessError::SpawnFailed)
        }
    }

    pub fn spawn_piped(&mut self) -> Result<PipedChild, ProcessError> {
        #[cfg(windows)]
        {
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
            mrml_windows::NativePipedChild::spawn(&mut command_line, current_directory.as_deref())
                .map(|native| PipedChild { native })
                .ok_or(ProcessError::SpawnFailed)
        }
        #[cfg(unix)]
        {
            let mut encoded = Vector::new();
            encoded.push(encode_c_string(&self.program)?);
            for argument in &self.arguments {
                encoded.push(encode_c_string(argument)?);
            }
            let mut pointers: Vector<*const i8> =
                encoded.iter().map(|value| value.as_ptr().cast()).collect();
            pointers.push(core::ptr::null());
            let program = core::ffi::CStr::from_bytes_with_nul(&encoded[0])
                .map_err(|_| ProcessError::InvalidArgument)?;
            let directory = self
                .current_directory
                .as_ref()
                .map(|value| encode_c_string(value))
                .transpose()?;
            let directory = directory.as_ref().map(|value| {
                core::ffi::CStr::from_bytes_with_nul(value).expect("validated C path")
            });
            mrml_linux::NativePipedChild::spawn(program, &pointers, directory)
                .map(|native| PipedChild { native })
                .ok_or(ProcessError::SpawnFailed)
        }
    }

    pub fn output(&mut self) -> Result<Output, ProcessError> {
        self.output_with_limits(64 * 1024 * 1024, 300_000)
    }

    pub fn output_with_limits(
        &mut self,
        max_output_bytes: usize,
        timeout_millis: u64,
    ) -> Result<Output, ProcessError> {
        if max_output_bytes == 0 || timeout_millis == 0 {
            return Err(ProcessError::InvalidArgument);
        }
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
            for argument in &self.arguments {
                encoded.push(encode_c_string(argument)?);
            }
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
        let started = crate::Instant::now();
        loop {
            let stdout_read = child.read_stdout(&mut buffer);
            if stdout
                .len()
                .saturating_add(stderr.len())
                .saturating_add(stdout_read)
                > max_output_bytes
            {
                let _ = child.kill();
                return Err(ProcessError::OutputLimit);
            }
            stdout
                .try_extend_from_slice(&buffer[..stdout_read])
                .map_err(|_| ProcessError::OutputFailed)?;
            let stderr_read = child.read_stderr(&mut buffer);
            if stdout
                .len()
                .saturating_add(stderr.len())
                .saturating_add(stderr_read)
                > max_output_bytes
            {
                let _ = child.kill();
                return Err(ProcessError::OutputLimit);
            }
            stderr
                .try_extend_from_slice(&buffer[..stderr_read])
                .map_err(|_| ProcessError::OutputFailed)?;
            if let Some(code) = child.try_wait() {
                loop {
                    let out = child.read_stdout(&mut buffer);
                    if stdout
                        .len()
                        .saturating_add(stderr.len())
                        .saturating_add(out)
                        > max_output_bytes
                    {
                        return Err(ProcessError::OutputLimit);
                    }
                    stdout
                        .try_extend_from_slice(&buffer[..out])
                        .map_err(|_| ProcessError::OutputFailed)?;
                    let err = child.read_stderr(&mut buffer);
                    if stdout
                        .len()
                        .saturating_add(stderr.len())
                        .saturating_add(err)
                        > max_output_bytes
                    {
                        return Err(ProcessError::OutputLimit);
                    }
                    stderr
                        .try_extend_from_slice(&buffer[..err])
                        .map_err(|_| ProcessError::OutputFailed)?;
                    if out == 0 && err == 0 {
                        break;
                    }
                }
                return Ok(Output {
                    status: ExitStatus(code),
                    stdout,
                    stderr,
                });
            }
            if started.elapsed().as_millis() >= timeout_millis as u128 {
                let _ = child.kill();
                return Err(ProcessError::TimedOut);
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
    if value.as_bytes().contains(&0) {
        return Err(ProcessError::InvalidArgument);
    }
    let mut encoded =
        Vector::with_capacity(value.len() + 1).map_err(|_| ProcessError::InvalidArgument)?;
    encoded
        .try_extend_from_slice(value.as_bytes())
        .map_err(|_| ProcessError::InvalidArgument)?;
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
                for _ in 0..=backslashes {
                    output.push('\\' as u16);
                }
            }
            for _ in 0..backslashes {
                output.push('\\' as u16);
            }
            backslashes = 0;
            output.push(unit);
        }
    }
    for _ in 0..backslashes.saturating_mul(2) {
        output.push('\\' as u16);
    }
    output.push('"' as u16);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_processes_enforce_output_and_time_limits() {
        #[cfg(windows)]
        let mut noisy = {
            let mut command = Command::new("powershell.exe");
            command.args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$x='x'*8192;[Console]::Write($x)",
            ]);
            command
        };
        #[cfg(unix)]
        let mut noisy = {
            let mut command = Command::new("sh");
            command.args(["-c", "head -c 8192 /dev/zero"]);
            command
        };
        assert_eq!(
            noisy.output_with_limits(1024, 10_000).unwrap_err(),
            ProcessError::OutputLimit
        );

        #[cfg(windows)]
        let mut slow = {
            let mut command = Command::new("powershell.exe");
            command.args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Sleep -Seconds 2",
            ]);
            command
        };
        #[cfg(unix)]
        let mut slow = {
            let mut command = Command::new("sh");
            command.args(["-c", "sleep 2"]);
            command
        };
        assert_eq!(
            slow.output_with_limits(1024, 20).unwrap_err(),
            ProcessError::TimedOut
        );
    }

    #[test]
    fn discovers_live_process_and_temporary_directory() {
        assert_ne!(process_id(), 0);
        assert!(crate::path_is_directory(&temporary_directory()));
    }

    #[test]
    fn captures_stdout_stderr_exit_status_and_working_directory() {
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("powershell.exe");
            command.args([
                "-NoProfile",
                "-Command",
                "[Console]::Out.Write('out λ'); [Console]::Error.Write('err 星'); exit 7",
            ]);
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

    #[test]
    fn detached_process_runs_to_completion() {
        let marker = crate::join_path(
            &temporary_directory(),
            &crate::mrml_format!("mrml-detached-{}.txt", process_id()),
        );
        let _ = crate::remove_file(&marker);
        #[cfg(windows)]
        let mut command = {
            let escaped = marker.replace("'", "''");
            let mut command = Command::new("powershell.exe");
            command
                .args(["-NoProfile", "-Command"])
                .arg(crate::mrml_format!(
                    "[IO.File]::WriteAllText('{}','ok')",
                    escaped
                ));
            command
        };
        #[cfg(unix)]
        let mut command = {
            let escaped = marker.replace("'", "'\\''");
            let mut command = Command::new("sh");
            command.args(["-c", &crate::mrml_format!("printf ok > '{}'", escaped)]);
            command
        };
        command.spawn_detached().unwrap();
        for _ in 0..200 {
            if crate::path_is_file(&marker) {
                assert_eq!(crate::read_file_text(&marker).unwrap().as_str(), "ok");
                assert!(crate::remove_file(&marker).is_ok());
                return;
            }
            #[cfg(windows)]
            mrml_windows::sleep_millis(5);
            #[cfg(unix)]
            mrml_linux::sleep_millis(5);
        }
        panic!("detached child did not create its marker");
    }

    #[test]
    fn silent_child_can_be_polled_killed_and_waited() {
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("powershell.exe");
            command.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 10"]);
            command
        };
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", "sleep 10"]);
            command
        };
        let mut child = command.spawn_silent().unwrap();
        assert!(child.try_wait().is_none());
        assert!(child.kill());
        let status = child.wait().unwrap();
        assert!(!status.success());
        assert_eq!(child.try_wait(), Some(status));
    }

    #[test]
    fn piped_child_round_trips_a_utf8_line() {
        #[cfg(windows)]
        let mut command = {
            let mut command = Command::new("powershell.exe");
            command.args([
                "-NoProfile",
                "-Command",
                "$utf8=[Text.UTF8Encoding]::new($false); [Console]::InputEncoding=$utf8; [Console]::OutputEncoding=$utf8; $line=[Console]::In.ReadLine(); [Console]::Out.WriteLine($line)",
            ]);
            command
        };
        #[cfg(unix)]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", "IFS= read -r line; printf '%s\\n' \"$line\""]);
            command
        };
        let mut child = command.spawn_piped().unwrap();
        child.write_all("hello λ 星\n".as_bytes()).unwrap();
        assert_eq!(child.read_line().unwrap().as_str(), "hello λ 星");
    }
}
