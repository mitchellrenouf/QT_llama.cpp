#![no_std]

#[cfg(unix)]
use core::alloc::{GlobalAlloc, Layout};
#[cfg(unix)]
use core::ffi::c_void;
#[cfg(unix)]
use core::ffi::{CStr, c_int, c_long};
#[cfg(unix)]
use core::ptr::NonNull;

#[repr(C)]
#[cfg(unix)]
struct Timespec {
    seconds: c_long,
    nanoseconds: c_long,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LocalTime {
    pub year: u16,
    pub month: u16,
    pub day: u16,
    pub weekday: u16,
    pub hour: u16,
    pub minute: u16,
    pub second: u16,
}

#[repr(C)]
#[cfg(unix)]
struct Tm {
    second: c_int,
    minute: c_int,
    hour: c_int,
    day: c_int,
    month: c_int,
    year: c_int,
    weekday: c_int,
    year_day: c_int,
    daylight: c_int,
    utc_offset: c_long,
    zone: *const i8,
}

#[cfg(unix)]
#[link(name = "c")]
unsafe extern "C" {
    fn localtime_r(clock: *const c_long, result: *mut Tm) -> *mut Tm;
    fn malloc(bytes: usize) -> *mut c_void;
    fn realloc(memory: *mut c_void, bytes: usize) -> *mut c_void;
    fn free(memory: *mut c_void);
    fn nanosleep(request: *const Timespec, remaining: *mut Timespec) -> c_int;
    fn clock_gettime(clock: c_int, time: *mut Timespec) -> c_int;
    fn dlopen(name: *const i8, flags: c_int) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn dlclose(module: *mut c_void) -> c_int;
    fn mmap(
        address: *mut c_void,
        len: usize,
        protection: c_int,
        flags: c_int,
        file: c_int,
        offset: isize,
    ) -> *mut c_void;
    fn munmap(address: *mut c_void, len: usize) -> c_int;
    fn _exit(status: c_int) -> !;
    fn isatty(file: c_int) -> c_int;
    fn getenv(name: *const i8) -> *mut i8;
    fn getpid() -> c_int;
    fn open(path: *const i8, flags: c_int, ...) -> c_int;
    fn read(file: c_int, buffer: *mut c_void, count: usize) -> isize;
    fn write(file: c_int, buffer: *const c_void, count: usize) -> isize;
    fn lseek(file: c_int, offset: isize, whence: c_int) -> isize;
    fn close(file: c_int) -> c_int;
    fn mkdir(path: *const i8, mode: u32) -> c_int;
    fn unlink(path: *const i8) -> c_int;
    fn rmdir(path: *const i8) -> c_int;
    fn pipe(files: *mut c_int) -> c_int;
    fn fork() -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
    fn execvp(program: *const i8, arguments: *const *const i8) -> c_int;
    fn chdir(path: *const i8) -> c_int;
    fn waitpid(process: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn kill(process: c_int, signal: c_int) -> c_int;
    fn fcntl(file: c_int, command: c_int, ...) -> c_int;
    fn rename(existing: *const i8, replacement: *const i8) -> c_int;
    fn opendir(path: *const i8) -> *mut c_void;
    fn readdir(directory: *mut c_void) -> *mut Dirent;
    fn closedir(directory: *mut c_void) -> c_int;
    fn realpath(path: *const i8, resolved: *mut i8) -> *mut i8;
    fn socket(domain: c_int, kind: c_int, protocol: c_int) -> c_int;
    fn bind(socket: c_int, address: *const SockAddr, length: u32) -> c_int;
    fn listen(socket: c_int, backlog: c_int) -> c_int;
    fn accept(socket: c_int, address: *mut SockAddr, length: *mut u32) -> c_int;
    fn connect(socket: c_int, address: *const SockAddr, length: u32) -> c_int;
    fn getsockname(socket: c_int, address: *mut SockAddr, length: *mut u32) -> c_int;
    fn setsockopt(socket: c_int, level: c_int, name: c_int, value: *const c_void, length: u32) -> c_int;
    fn pthread_create(
        thread: *mut usize,
        attributes: *const c_void,
        start: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        argument: *mut c_void,
    ) -> c_int;
    fn pthread_detach(thread: usize) -> c_int;
    fn sysconf(name: c_int) -> c_long;
    fn sched_yield() -> c_int;
    fn syscall(number: c_long, ...) -> c_long;
}

#[repr(C)]
#[cfg(unix)]
struct SockAddr {
    family: u16,
    port: [u8; 2],
    address: [u8; 4],
    zero: [u8; 8],
}

#[repr(C)]
#[cfg(unix)]
struct TimeVal { seconds: isize, microseconds: isize }

#[repr(C)]
#[cfg(unix)]
struct Dirent {
    inode: u64,
    offset: i64,
    record_length: u16,
    kind: u8,
    name: [i8; 256],
}

#[cfg(unix)]
pub fn processor_count() -> usize {
    const SC_NPROCESSORS_ONLN: c_int = 84;
    unsafe { sysconf(SC_NPROCESSORS_ONLN) }.max(1) as usize
}

#[cfg(unix)]
pub fn process_id() -> u32 {
    (unsafe { getpid() }) as u32
}

#[cfg(unix)]
fn socket_address(ip: [u8; 4], port: u16) -> SockAddr {
    SockAddr { family: 2, port: port.to_be_bytes(), address: ip, zero: [0; 8] }
}

#[cfg(unix)]
pub struct NativeTcpListener(c_int);

#[cfg(unix)]
impl NativeTcpListener {
    pub fn bind(ip: [u8; 4], port: u16) -> Option<Self> {
        let handle = unsafe { socket(2, 1, 0) };
        if handle < 0 { return None; }
        let address = socket_address(ip, port);
        if unsafe { bind(handle, &address, core::mem::size_of::<SockAddr>() as u32) } != 0
            || unsafe { listen(handle, 128) } != 0
        { let _ = unsafe { close(handle) }; return None; }
        Some(Self(handle))
    }

    pub fn local_port(&self) -> Option<u16> {
        let mut address = socket_address([0; 4], 0);
        let mut length = core::mem::size_of::<SockAddr>() as u32;
        (unsafe { getsockname(self.0, &mut address, &mut length) } == 0).then(|| u16::from_be_bytes(address.port))
    }

    pub fn accept(&self) -> Option<NativeTcpStream> {
        let handle = unsafe { accept(self.0, core::ptr::null_mut(), core::ptr::null_mut()) };
        (handle >= 0).then_some(NativeTcpStream(handle))
    }
}

#[cfg(unix)]
impl Drop for NativeTcpListener { fn drop(&mut self) { let _ = unsafe { close(self.0) }; } }

#[cfg(unix)]
pub struct NativeTcpStream(c_int);

#[cfg(unix)]
unsafe impl Send for NativeTcpStream {}

#[cfg(unix)]
impl NativeTcpStream {
    pub fn connect(ip: [u8; 4], port: u16) -> Option<Self> {
        let handle = unsafe { socket(2, 1, 0) };
        if handle < 0 { return None; }
        let address = socket_address(ip, port);
        if unsafe { connect(handle, &address, core::mem::size_of::<SockAddr>() as u32) } != 0 {
            let _ = unsafe { close(handle) };
            None
        } else { Some(Self(handle)) }
    }

    pub fn read(&mut self, buffer: &mut [u8]) -> Option<usize> {
        let amount = unsafe { read(self.0, buffer.as_mut_ptr().cast(), buffer.len()) };
        (amount >= 0).then_some(amount as usize)
    }

    pub fn write(&mut self, buffer: &[u8]) -> Option<usize> {
        let amount = unsafe { write(self.0, buffer.as_ptr().cast(), buffer.len()) };
        (amount >= 0).then_some(amount as usize)
    }

    pub fn set_timeout_millis(&self, read_timeout: bool, milliseconds: u64) -> bool {
        let value = TimeVal { seconds: (milliseconds / 1000) as isize, microseconds: ((milliseconds % 1000) * 1000) as isize };
        let option = if read_timeout { 20 } else { 21 };
        (unsafe { setsockopt(self.0, 1, option, (&value as *const TimeVal).cast(), core::mem::size_of::<TimeVal>() as u32) }) == 0
    }
}

#[cfg(unix)]
impl Drop for NativeTcpStream { fn drop(&mut self) { let _ = unsafe { close(self.0) }; } }

#[cfg(unix)]
pub fn spawn_detached_process(
    program: &CStr,
    arguments: &[*const i8],
    current_directory: Option<&CStr>,
) -> bool {
    let process = unsafe { fork() };
    if process == 0 {
        let detached = unsafe { fork() };
        if detached < 0 {
            unsafe { _exit(125) };
        }
        if detached > 0 {
            unsafe { _exit(0) };
        }
        if let Some(directory) = current_directory {
            if unsafe { chdir(directory.as_ptr()) } != 0 {
                unsafe { _exit(126) };
            }
        }
        let _ = unsafe { execvp(program.as_ptr(), arguments.as_ptr()) };
        unsafe { _exit(127) };
    }
    if process < 0 {
        return false;
    }
    let mut status = 0;
    (unsafe { waitpid(process, &mut status, 0) }) == process && status == 0
}

#[cfg(unix)]
pub struct NativeChild {
    process: c_int,
    stdout: c_int,
    stderr: c_int,
    status: Option<i32>,
}

#[cfg(unix)]
impl NativeChild {
    pub fn spawn_silent(
        program: &CStr,
        arguments: &[*const i8],
        current_directory: Option<&CStr>,
    ) -> Option<Self> {
        const O_RDWR: c_int = 2;
        let null_file = unsafe { open(c"/dev/null".as_ptr(), O_RDWR) };
        if null_file < 0 { return None; }
        let process = unsafe { fork() };
        if process == 0 {
            let _ = unsafe { dup2(null_file, 0) };
            let _ = unsafe { dup2(null_file, 1) };
            let _ = unsafe { dup2(null_file, 2) };
            let _ = unsafe { close(null_file) };
            if let Some(directory) = current_directory {
                if unsafe { chdir(directory.as_ptr()) } != 0 { unsafe { _exit(126) }; }
            }
            let _ = unsafe { execvp(program.as_ptr(), arguments.as_ptr()) };
            unsafe { _exit(127) };
        }
        let _ = unsafe { close(null_file) };
        (process > 0).then_some(Self { process, stdout: -1, stderr: -1, status: None })
    }

    pub fn spawn_captured(
        program: &CStr,
        arguments: &[*const i8],
        current_directory: Option<&CStr>,
    ) -> Option<Self> {
        let mut stdout_pipe = [-1; 2];
        let mut stderr_pipe = [-1; 2];
        if unsafe { pipe(stdout_pipe.as_mut_ptr()) } != 0
            || unsafe { pipe(stderr_pipe.as_mut_ptr()) } != 0
        {
            for file in stdout_pipe.into_iter().chain(stderr_pipe) {
                if file >= 0 {
                    let _ = unsafe { close(file) };
                }
            }
            return None;
        }
        let process = unsafe { fork() };
        if process == 0 {
            let _ = unsafe { close(stdout_pipe[0]) };
            let _ = unsafe { close(stderr_pipe[0]) };
            let _ = unsafe { dup2(stdout_pipe[1], 1) };
            let _ = unsafe { dup2(stderr_pipe[1], 2) };
            let _ = unsafe { close(stdout_pipe[1]) };
            let _ = unsafe { close(stderr_pipe[1]) };
            if let Some(directory) = current_directory {
                if unsafe { chdir(directory.as_ptr()) } != 0 {
                    unsafe { _exit(126) };
                }
            }
            let _ = unsafe { execvp(program.as_ptr(), arguments.as_ptr()) };
            unsafe { _exit(127) };
        }
        let _ = unsafe { close(stdout_pipe[1]) };
        let _ = unsafe { close(stderr_pipe[1]) };
        if process < 0 {
            let _ = unsafe { close(stdout_pipe[0]) };
            let _ = unsafe { close(stderr_pipe[0]) };
            return None;
        }
        const F_SETFL: c_int = 4;
        const O_NONBLOCK: c_int = 0o4000;
        let _ = unsafe { fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK) };
        let _ = unsafe { fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK) };
        Some(Self { process, stdout: stdout_pipe[0], stderr: stderr_pipe[0], status: None })
    }

    fn read_pipe(file: c_int, buffer: &mut [u8]) -> usize {
        let read = unsafe { read(file, buffer.as_mut_ptr().cast(), buffer.len()) };
        if read > 0 { read as usize } else { 0 }
    }

    pub fn read_stdout(&mut self, buffer: &mut [u8]) -> usize {
        Self::read_pipe(self.stdout, buffer)
    }

    pub fn read_stderr(&mut self, buffer: &mut [u8]) -> usize {
        Self::read_pipe(self.stderr, buffer)
    }

    pub fn try_wait(&mut self) -> Option<i32> {
        if let Some(status) = self.status {
            return Some(status);
        }
        let mut status = 0;
        const WNOHANG: c_int = 1;
        if unsafe { waitpid(self.process, &mut status, WNOHANG) } == self.process {
            let code = if status & 0x7f == 0 {
                (status >> 8) & 0xff
            } else {
                128 + (status & 0x7f)
            };
            self.status = Some(code);
        }
        self.status
    }

    pub fn kill(&mut self) -> bool {
        self.status.is_some() || unsafe { kill(self.process, 9) } == 0
    }

    pub fn wait(&mut self) -> Option<i32> {
        if let Some(status) = self.status { return Some(status); }
        let mut status = 0;
        if unsafe { waitpid(self.process, &mut status, 0) } != self.process { return None; }
        let code = if status & 0x7f == 0 { (status >> 8) & 0xff } else { 128 + (status & 0x7f) };
        self.status = Some(code);
        Some(code)
    }
}

#[cfg(unix)]
impl Drop for NativeChild {
    fn drop(&mut self) {
        if self.stdout >= 0 { let _ = unsafe { close(self.stdout) }; }
        if self.stderr >= 0 { let _ = unsafe { close(self.stderr) }; }
        if self.status.is_none() {
            let mut status = 0;
            let _ = unsafe { waitpid(self.process, &mut status, 0) };
        }
    }
}

#[cfg(unix)]
pub struct NativePipedChild {
    process: c_int,
    stdin: c_int,
    stdout: c_int,
    status: Option<i32>,
}

#[cfg(unix)]
impl NativePipedChild {
    pub fn spawn(program: &CStr, arguments: &[*const i8], current_directory: Option<&CStr>) -> Option<Self> {
        let mut stdin_pipe = [-1; 2];
        let mut stdout_pipe = [-1; 2];
        if unsafe { pipe(stdin_pipe.as_mut_ptr()) } != 0 || unsafe { pipe(stdout_pipe.as_mut_ptr()) } != 0 {
            for file in stdin_pipe.into_iter().chain(stdout_pipe) { if file >= 0 { let _ = unsafe { close(file) }; } }
            return None;
        }
        let process = unsafe { fork() };
        if process == 0 {
            let _ = unsafe { close(stdin_pipe[1]) };
            let _ = unsafe { close(stdout_pipe[0]) };
            let _ = unsafe { dup2(stdin_pipe[0], 0) };
            let _ = unsafe { dup2(stdout_pipe[1], 1) };
            let _ = unsafe { close(stdin_pipe[0]) };
            let _ = unsafe { close(stdout_pipe[1]) };
            if let Some(directory) = current_directory {
                if unsafe { chdir(directory.as_ptr()) } != 0 { unsafe { _exit(126) }; }
            }
            let _ = unsafe { execvp(program.as_ptr(), arguments.as_ptr()) };
            unsafe { _exit(127) };
        }
        let _ = unsafe { close(stdin_pipe[0]) };
        let _ = unsafe { close(stdout_pipe[1]) };
        if process < 0 {
            let _ = unsafe { close(stdin_pipe[1]) };
            let _ = unsafe { close(stdout_pipe[0]) };
            return None;
        }
        Some(Self { process, stdin: stdin_pipe[1], stdout: stdout_pipe[0], status: None })
    }

    pub fn write_stdin(&mut self, bytes: &[u8]) -> bool { write_descriptor(self.stdin, bytes) }

    pub fn read_stdout(&mut self, buffer: &mut [u8]) -> Option<usize> {
        let amount = unsafe { read(self.stdout, buffer.as_mut_ptr().cast(), buffer.len()) };
        (amount >= 0).then_some(amount as usize)
    }

    pub fn kill(&mut self) -> bool { self.status.is_some() || unsafe { kill(self.process, 9) } == 0 }

    pub fn wait(&mut self) -> Option<i32> {
        if let Some(status) = self.status { return Some(status); }
        let mut status = 0;
        if unsafe { waitpid(self.process, &mut status, 0) } != self.process { return None; }
        let code = if status & 0x7f == 0 { (status >> 8) & 0xff } else { 128 + (status & 0x7f) };
        self.status = Some(code);
        Some(code)
    }
}

#[cfg(unix)]
impl Drop for NativePipedChild {
    fn drop(&mut self) {
        let _ = unsafe { close(self.stdin) };
        let _ = unsafe { close(self.stdout) };
        if self.status.is_none() { let _ = self.kill(); let _ = self.wait(); }
    }
}

#[cfg(unix)]
pub fn yield_now() {
    let _ = unsafe { sched_yield() };
}

#[cfg(all(unix, target_arch = "x86_64"))]
const SYS_FUTEX: c_long = 202;
#[cfg(all(unix, target_arch = "aarch64"))]
const SYS_FUTEX: c_long = 98;

#[cfg(unix)]
pub fn wait_on_u32(address: *const u32, expected: u32) {
    const FUTEX_WAIT_PRIVATE: c_int = 128;
    let _ = unsafe {
        syscall(
            SYS_FUTEX,
            address,
            FUTEX_WAIT_PRIVATE,
            expected,
            core::ptr::null::<Timespec>(),
        )
    };
}

#[cfg(unix)]
pub fn wake_one_u32(address: *const u32) {
    const FUTEX_WAKE_PRIVATE: c_int = 129;
    let _ = unsafe { syscall(SYS_FUTEX, address, FUTEX_WAKE_PRIVATE, 1) };
}

#[cfg(unix)]
pub fn wake_all_u32(address: *const u32) {
    const FUTEX_WAKE_PRIVATE: c_int = 129;
    let _ = unsafe { syscall(SYS_FUTEX, address, FUTEX_WAKE_PRIVATE, i32::MAX) };
}

#[cfg(unix)]
pub unsafe fn spawn_detached_thread(
    context: *mut c_void,
    start: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
) -> bool {
    let mut thread = 0usize;
    if unsafe { pthread_create(&mut thread, core::ptr::null(), start, context) } != 0 {
        return false;
    }
    unsafe { pthread_detach(thread) == 0 }
}

#[derive(Debug)]
#[cfg(unix)]
pub struct NativeFile(c_int);

#[cfg(unix)]
impl NativeFile {
    pub fn open_read(path: &CStr) -> Option<Self> {
        let file = unsafe { open(path.as_ptr(), 0) };
        (file >= 0).then_some(Self(file))
    }

    pub fn create_write(path: &CStr) -> Option<Self> {
        const O_WRONLY: c_int = 1;
        const O_CREAT: c_int = 64;
        const O_TRUNC: c_int = 512;
        let file = unsafe { open(path.as_ptr(), O_WRONLY | O_CREAT | O_TRUNC, 0o666) };
        (file >= 0).then_some(Self(file))
    }

    pub fn read(&self, buffer: &mut [u8]) -> Option<usize> {
        let read = unsafe { read(self.0, buffer.as_mut_ptr().cast(), buffer.len()) };
        (read >= 0).then_some(read as usize)
    }

    pub fn write(&self, buffer: &[u8]) -> Option<usize> {
        let result = unsafe { write(self.0, buffer.as_ptr().cast(), buffer.len()) };
        (result >= 0).then_some(result as usize)
    }

    pub fn seek_absolute(&self, position: u64) -> bool {
        position <= isize::MAX as u64 && unsafe { lseek(self.0, position as isize, 0) } >= 0
    }

    pub fn len(&self) -> Option<u64> {
        let current = unsafe { lseek(self.0, 0, 1) };
        if current < 0 {
            return None;
        }
        let end = unsafe { lseek(self.0, 0, 2) };
        let _ = unsafe { lseek(self.0, current, 0) };
        (end >= 0).then_some(end as u64)
    }

    pub fn raw_fd(&self) -> c_int {
        self.0
    }
}

#[cfg(unix)]
impl Drop for NativeFile {
    fn drop(&mut self) {
        let _ = unsafe { close(self.0) };
    }
}

#[cfg(unix)]
pub fn stdout_is_terminal() -> bool {
    (unsafe { isatty(1) }) == 1
}

#[cfg(unix)]
fn write_descriptor(file: c_int, mut bytes: &[u8]) -> bool {
    while !bytes.is_empty() {
        let written = unsafe { write(file, bytes.as_ptr().cast(), bytes.len()) };
        if written <= 0 {
            return false;
        }
        bytes = &bytes[written as usize..];
    }
    true
}

#[cfg(unix)]
pub fn write_stdout(bytes: &[u8]) -> bool {
    write_descriptor(1, bytes)
}

#[cfg(unix)]
pub fn write_stderr(bytes: &[u8]) -> bool {
    write_descriptor(2, bytes)
}

#[cfg(unix)]
pub fn read_stdin(buffer: &mut [u8]) -> Option<usize> {
    let result = unsafe { read(0, buffer.as_mut_ptr().cast(), buffer.len()) };
    (result >= 0).then_some(result as usize)
}

#[cfg(unix)]
pub fn environment_variable_is_set(name: &CStr) -> bool {
    !unsafe { getenv(name.as_ptr()) }.is_null()
}

#[cfg(unix)]
pub fn environment_variable_equals(name: &CStr, expected: &[u8]) -> bool {
    let value = unsafe { getenv(name.as_ptr()) };
    if value.is_null() {
        return false;
    }
    unsafe { CStr::from_ptr(value) }.to_bytes() == expected
}

#[cfg(unix)]
pub fn environment_variable_bytes(name: &CStr, value: &mut [u8]) -> Result<usize, usize> {
    let pointer = unsafe { getenv(name.as_ptr()) };
    if pointer.is_null() {
        return Err(0);
    }
    let bytes = unsafe { CStr::from_ptr(pointer) }.to_bytes();
    if bytes.len() > value.len() {
        return Err(bytes.len());
    }
    value[..bytes.len()].copy_from_slice(bytes);
    Ok(bytes.len())
}

#[cfg(unix)]
pub fn path_is_directory(path: &CStr) -> bool {
    const O_RDONLY: c_int = 0;
    const O_DIRECTORY: c_int = 0o200000;
    let file = unsafe { open(path.as_ptr(), O_RDONLY | O_DIRECTORY) };
    if file < 0 {
        false
    } else {
        let _ = unsafe { close(file) };
        true
    }
}

#[cfg(unix)]
pub fn canonical_path<'a>(path: &CStr, output: &'a mut [u8; 4096]) -> Option<&'a [u8]> {
    NonNull::new(unsafe { realpath(path.as_ptr(), output.as_mut_ptr().cast()) })?;
    let length = output.iter().position(|byte| *byte == 0)?;
    Some(&output[..length])
}

#[cfg(unix)]
pub fn create_directory(path: &CStr) -> bool {
    (unsafe { mkdir(path.as_ptr(), 0o777) }) == 0 || path_is_directory(path)
}

#[cfg(unix)]
pub fn delete_file(path: &CStr) -> bool {
    (unsafe { unlink(path.as_ptr()) }) == 0
}

#[cfg(unix)]
pub fn remove_directory(path: &CStr) -> bool {
    (unsafe { rmdir(path.as_ptr()) }) == 0
}

#[cfg(unix)]
pub fn rename_file(existing: &CStr, replacement: &CStr) -> bool {
    (unsafe { rename(existing.as_ptr(), replacement.as_ptr()) }) == 0
}

#[cfg(unix)]
pub struct NativeDirectory(*mut c_void);

#[cfg(unix)]
impl NativeDirectory {
    pub fn open(path: &CStr) -> Option<Self> {
        NonNull::new(unsafe { opendir(path.as_ptr()) }).map(|directory| Self(directory.as_ptr()))
    }

    pub fn next<'a>(&mut self, name: &'a mut [u8; 256]) -> Option<(&'a [u8], bool, bool)> {
        let entry = unsafe { readdir(self.0).as_ref()? };
        let len = entry.name.iter().position(|byte| *byte == 0)?;
        for (target, source) in name[..len].iter_mut().zip(&entry.name[..len]) {
            *target = *source as u8;
        }
        const DT_DIR: u8 = 4;
        const DT_LNK: u8 = 10;
        Some((&name[..len], entry.kind == DT_DIR, entry.kind == DT_LNK))
    }
}

#[cfg(unix)]
impl Drop for NativeDirectory {
    fn drop(&mut self) {
        let _ = unsafe { closedir(self.0) };
    }
}

#[cfg(unix)]
pub fn exit_process(status: i32) -> ! {
    unsafe { _exit(status) }
}

#[cfg(unix)]
pub unsafe fn map_file_read_only(file: c_int, len: usize) -> Option<NonNull<u8>> {
    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;
    let view = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file, 0) };
    if view as isize == -1 {
        None
    } else {
        NonNull::new(view.cast())
    }
}

#[cfg(unix)]
pub unsafe fn unmap_file(address: NonNull<u8>, len: usize) -> bool {
    unsafe { munmap(address.as_ptr().cast(), len) == 0 }
}

#[derive(Debug)]
#[cfg(unix)]
pub struct DynamicLibrary(*mut c_void);

#[cfg(unix)]
unsafe impl Send for DynamicLibrary {}
#[cfg(unix)]
unsafe impl Sync for DynamicLibrary {}

#[cfg(unix)]
impl DynamicLibrary {
    pub fn open(name: &core::ffi::CStr) -> Option<Self> {
        const RTLD_NOW: c_int = 2;
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        (!handle.is_null()).then_some(Self(handle))
    }

    pub fn symbol(&self, name: &core::ffi::CStr) -> Option<*mut c_void> {
        let symbol = unsafe { dlsym(self.0, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }
}

#[cfg(unix)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        let _ = unsafe { dlclose(self.0) };
    }
}

pub struct SystemAllocator;

#[cfg(unix)]
unsafe impl GlobalAlloc for SystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { malloc(layout.size().max(1)) }.cast()
    }

    unsafe fn dealloc(&self, pointer: *mut u8, _: Layout) {
        unsafe { free(pointer.cast()) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, _: Layout, size: usize) -> *mut u8 {
        unsafe { realloc(pointer.cast(), size.max(1)) }.cast()
    }
}

#[cfg(unix)]
pub fn sleep_millis(milliseconds: u64) {
    let request = Timespec {
        seconds: (milliseconds / 1000) as c_long,
        nanoseconds: ((milliseconds % 1000) * 1_000_000) as c_long,
    };
    let _ = unsafe { nanosleep(&request, core::ptr::null_mut()) };
}

#[cfg(unix)]
pub fn monotonic_nanos() -> u64 {
    const CLOCK_MONOTONIC: c_int = 1;
    let mut value = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    if unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) } != 0 {
        return 0;
    }
    value.seconds as u64 * 1_000_000_000 + value.nanoseconds as u64
}

#[cfg(unix)]
pub fn unix_time_millis() -> u64 {
    const CLOCK_REALTIME: c_int = 0;
    let mut value = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    if unsafe { clock_gettime(CLOCK_REALTIME, &mut value) } != 0 {
        return 0;
    }
    value.seconds as u64 * 1000 + value.nanoseconds as u64 / 1_000_000
}

#[cfg(unix)]
pub fn local_time(epoch_seconds: i64) -> Option<LocalTime> {
    let seconds = epoch_seconds as c_long;
    let mut value = core::mem::MaybeUninit::<Tm>::uninit();
    // SAFETY: localtime_r initializes value or returns null.
    let result = unsafe { localtime_r(&seconds, value.as_mut_ptr()) };
    if result.is_null() {
        return None;
    }
    // SAFETY: a non-null return guarantees initialization.
    let value = unsafe { value.assume_init() };
    Some(LocalTime {
        year: (value.year + 1900) as u16,
        month: (value.month + 1) as u16,
        day: value.day as u16,
        weekday: value.weekday as u16,
        hour: value.hour as u16,
        minute: value.minute as u16,
        second: value.second as u16,
    })
}

#[cfg(not(unix))]
pub fn local_time(_: i64) -> Option<LocalTime> {
    None
}

#[cfg(not(unix))]
pub fn sleep_millis(_: u64) {}

#[cfg(not(unix))]
pub fn monotonic_nanos() -> u64 {
    0
}

#[cfg(not(unix))]
pub fn unix_time_millis() -> u64 {
    0
}

#[cfg(all(test, unix))]
mod tests {
    use core::alloc::{GlobalAlloc, Layout};

    #[test]
    fn native_allocator_and_clock_work() {
        assert!(super::unix_time_millis() > 1_000_000_000_000);
        let layout = Layout::from_size_align(64, 8).unwrap();
        let pointer = unsafe { GlobalAlloc::alloc(&super::SystemAllocator, layout) };
        assert!(!pointer.is_null());
        unsafe {
            pointer.write(0x5a);
            assert_eq!(pointer.read(), 0x5a);
            GlobalAlloc::dealloc(&super::SystemAllocator, pointer, layout);
        }

        let before = super::monotonic_nanos();
        super::sleep_millis(2);
        assert!(super::monotonic_nanos() > before);
    }

    #[test]
    fn reads_process_environment_without_allocation() {
        assert!(super::environment_variable_is_set(c"PATH"));
        let _ = super::stdout_is_terminal();
    }

    #[test]
    fn loads_native_library_symbols() {
        let library = super::DynamicLibrary::open(c"libc.so.6").unwrap();
        assert!(library.symbol(c"getpid").is_some());
        assert!(library.symbol(c"definitely_missing_symbol").is_none());
    }

    #[test]
    fn maps_native_file_descriptor_read_only() {
        let path = c"mrml-linux-map-test.bin";
        let file = super::NativeFile::create_write(path).unwrap();
        assert_eq!(file.write(b"native mapping"), Some(14));
        drop(file);
        let file = super::NativeFile::open_read(path).unwrap();
        let mapping = unsafe { super::map_file_read_only(file.raw_fd(), 14) }.unwrap();
        assert_eq!(
            unsafe { core::slice::from_raw_parts(mapping.as_ptr(), 14) },
            b"native mapping"
        );
        assert!(unsafe { super::unmap_file(mapping, 14) });
        drop(file);
        assert!(super::delete_file(path));
    }
}
