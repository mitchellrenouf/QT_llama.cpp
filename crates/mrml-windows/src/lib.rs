#![no_std]

#[cfg(windows)]
use core::alloc::{GlobalAlloc, Layout};
#[cfg(windows)]
use core::ffi::{CStr, c_void};
#[cfg(windows)]
use core::ptr::NonNull;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct LocalTime {
    pub year: u16,
    pub month: u16,
    pub weekday: u16,
    pub day: u16,
    pub hour: u16,
    pub minute: u16,
    pub second: u16,
    pub milliseconds: u16,
}

#[repr(C)]
#[cfg(windows)]
struct FileTime {
    low: u32,
    high: u32,
}

#[repr(C)]
#[cfg(windows)]
struct FindDataW {
    attributes: u32,
    creation_time: FileTime,
    access_time: FileTime,
    write_time: FileTime,
    size_high: u32,
    size_low: u32,
    reserved0: u32,
    reserved1: u32,
    file_name: [u16; 260],
    alternate_file_name: [u16; 14],
    file_type: u32,
    creator_type: u32,
    finder_flags: u16,
}

#[repr(C)]
#[cfg(windows)]
struct SecurityAttributes {
    length: u32,
    security_descriptor: *mut c_void,
    inherit_handle: i32,
}

#[repr(C)]
#[cfg(windows)]
struct StartupInfoW {
    size: u32,
    reserved: *mut u16,
    desktop: *mut u16,
    title: *mut u16,
    x: u32,
    y: u32,
    x_size: u32,
    y_size: u32,
    x_count_chars: u32,
    y_count_chars: u32,
    fill_attribute: u32,
    flags: u32,
    show_window: u16,
    reserved_bytes: u16,
    reserved_pointer: *mut u8,
    stdin: *mut c_void,
    stdout: *mut c_void,
    stderr: *mut c_void,
}

#[repr(C)]
#[cfg(windows)]
struct ProcessInformation {
    process: *mut c_void,
    thread: *mut c_void,
    process_id: u32,
    thread_id: u32,
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLocalTime(time: *mut LocalTime);
    fn GetCommandLineW() -> *const u16;
    fn GetSystemTimeAsFileTime(time: *mut FileTime);
    fn GetProcessHeap() -> *mut c_void;
    fn GetCurrentProcessId() -> u32;
    fn CreatePipe(read: *mut *mut c_void, write: *mut *mut c_void, attributes: *const SecurityAttributes, size: u32) -> i32;
    fn SetHandleInformation(handle: *mut c_void, mask: u32, flags: u32) -> i32;
    fn CreateProcessW(
        application: *const u16,
        command_line: *mut u16,
        process_attributes: *const SecurityAttributes,
        thread_attributes: *const SecurityAttributes,
        inherit_handles: i32,
        creation_flags: u32,
        environment: *const c_void,
        current_directory: *const u16,
        startup: *mut StartupInfoW,
        process: *mut ProcessInformation,
    ) -> i32;
    fn PeekNamedPipe(handle: *mut c_void, buffer: *mut c_void, size: u32, read: *mut u32, available: *mut u32, left: *mut u32) -> i32;
    fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
    fn GetExitCodeProcess(process: *mut c_void, code: *mut u32) -> i32;
    fn TerminateProcess(process: *mut c_void, exit_code: u32) -> i32;
    fn HeapAlloc(heap: *mut c_void, flags: u32, bytes: usize) -> *mut c_void;
    fn HeapReAlloc(heap: *mut c_void, flags: u32, memory: *mut c_void, bytes: usize)
    -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, memory: *mut c_void) -> i32;
    fn QueryPerformanceCounter(value: *mut i64) -> i32;
    fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    fn Sleep(milliseconds: u32);
    fn LoadLibraryA(name: *const i8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetStdHandle(kind: u32) -> *mut c_void;
    fn GetConsoleMode(handle: *mut c_void, mode: *mut u32) -> i32;
    fn GetEnvironmentVariableA(name: *const i8, value: *mut i8, capacity: u32) -> u32;
    fn GetEnvironmentVariableW(name: *const u16, value: *mut u16, capacity: u32) -> u32;
    fn GetFileAttributesW(name: *const u16) -> u32;
    fn GetFullPathNameW(name: *const u16, capacity: u32, output: *mut u16, file_part: *mut *mut u16) -> u32;
    fn CreateDirectoryW(name: *const u16, security: *const c_void) -> i32;
    fn DeleteFileW(name: *const u16) -> i32;
    fn RemoveDirectoryW(name: *const u16) -> i32;
    fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    fn FindFirstFileW(pattern: *const u16, data: *mut FindDataW) -> *mut c_void;
    fn FindNextFileW(find: *mut c_void, data: *mut FindDataW) -> i32;
    fn FindClose(find: *mut c_void) -> i32;
    fn GetLastError() -> u32;
    fn SetLastError(error: u32);
    fn CreateFileMappingW(
        file: *mut c_void,
        attributes: *const c_void,
        protection: u32,
        maximum_size_high: u32,
        maximum_size_low: u32,
        name: *const u16,
    ) -> *mut c_void;
    fn MapViewOfFile(
        mapping: *mut c_void,
        access: u32,
        offset_high: u32,
        offset_low: u32,
        bytes: usize,
    ) -> *mut c_void;
    fn UnmapViewOfFile(address: *const c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
    fn CreateFileW(
        name: *const u16,
        access: u32,
        share: u32,
        security: *const c_void,
        creation: u32,
        flags: u32,
        template: *mut c_void,
    ) -> *mut c_void;
    fn ReadFile(
        file: *mut c_void,
        buffer: *mut c_void,
        bytes: u32,
        read: *mut u32,
        overlapped: *mut c_void,
    ) -> i32;
    fn WriteFile(
        file: *mut c_void,
        buffer: *const c_void,
        bytes: u32,
        written: *mut u32,
        overlapped: *mut c_void,
    ) -> i32;
    fn SetFilePointerEx(file: *mut c_void, distance: i64, position: *mut i64, method: u32) -> i32;
    fn GetFileSizeEx(file: *mut c_void, size: *mut i64) -> i32;
    fn CreateThread(
        attributes: *const c_void,
        stack_size: usize,
        start: unsafe extern "system" fn(*mut c_void) -> u32,
        parameter: *mut c_void,
        creation_flags: u32,
        thread_id: *mut u32,
    ) -> *mut c_void;
    fn GetActiveProcessorCount(group_number: u16) -> u32;
    fn SwitchToThread() -> i32;
    fn WaitOnAddress(
        address: *const c_void,
        compare_address: *const c_void,
        address_size: usize,
        milliseconds: u32,
    ) -> i32;
    fn WakeByAddressSingle(address: *const c_void);
    fn WakeByAddressAll(address: *const c_void);
    fn ExitProcess(exit_code: u32) -> !;
}

#[cfg(windows)]
pub fn command_line_wide() -> &'static [u16] {
    let pointer = unsafe { GetCommandLineW() };
    let mut length = 0usize;
    while unsafe { *pointer.add(length) } != 0 {
        length += 1;
    }
    unsafe { core::slice::from_raw_parts(pointer, length) }
}

#[cfg(windows)]
pub fn wide_path_is_file(name: &[u16]) -> bool {
    const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;
    const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
    if name.last().copied() != Some(0) {
        return false;
    }
    let attributes = unsafe { GetFileAttributesW(name.as_ptr()) };
    attributes != INVALID_FILE_ATTRIBUTES && attributes & FILE_ATTRIBUTE_DIRECTORY == 0
}

#[cfg(windows)]
pub fn wide_path_is_directory(name: &[u16]) -> bool {
    const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;
    const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
    if name.last().copied() != Some(0) {
        return false;
    }
    let attributes = unsafe { GetFileAttributesW(name.as_ptr()) };
    attributes != INVALID_FILE_ATTRIBUTES && attributes & FILE_ATTRIBUTE_DIRECTORY != 0
}

#[cfg(windows)]
pub fn full_path_wide(name: &[u16], output: &mut [u16]) -> Option<usize> {
    if name.last().copied() != Some(0) {
        return None;
    }
    let length = unsafe {
        GetFullPathNameW(
            name.as_ptr(),
            output.len().min(u32::MAX as usize) as u32,
            output.as_mut_ptr(),
            core::ptr::null_mut(),
        )
    } as usize;
    (length > 0 && length < output.len()).then_some(length)
}

#[cfg(windows)]
pub fn create_directory_wide(name: &[u16]) -> bool {
    const ERROR_ALREADY_EXISTS: u32 = 183;
    if name.last().copied() != Some(0) {
        return false;
    }
    (unsafe { CreateDirectoryW(name.as_ptr(), core::ptr::null()) }) != 0
        || (unsafe { GetLastError() } == ERROR_ALREADY_EXISTS && wide_path_is_directory(name))
}

#[cfg(windows)]
pub fn delete_file_wide(name: &[u16]) -> bool {
    name.last().copied() == Some(0) && unsafe { DeleteFileW(name.as_ptr()) } != 0
}

#[cfg(windows)]
pub fn remove_directory_wide(name: &[u16]) -> bool {
    name.last().copied() == Some(0) && unsafe { RemoveDirectoryW(name.as_ptr()) } != 0
}

#[cfg(windows)]
pub fn rename_file_wide(existing: &[u16], replacement: &[u16]) -> bool {
    const MOVEFILE_REPLACE_EXISTING: u32 = 1;
    existing.last().copied() == Some(0)
        && replacement.last().copied() == Some(0)
        && unsafe {
            MoveFileExW(
                existing.as_ptr(),
                replacement.as_ptr(),
                MOVEFILE_REPLACE_EXISTING,
            )
        } != 0
}

#[cfg(windows)]
pub struct NativeDirectory {
    find: *mut c_void,
    data: FindDataW,
    first: bool,
}

#[cfg(windows)]
impl NativeDirectory {
    pub fn open(pattern: &[u16]) -> Option<Self> {
        const INVALID_HANDLE_VALUE: *mut c_void = usize::MAX as *mut c_void;
        if pattern.last().copied() != Some(0) {
            return None;
        }
        let mut data = unsafe { core::mem::zeroed::<FindDataW>() };
        let find = unsafe { FindFirstFileW(pattern.as_ptr(), &mut data) };
        (find != INVALID_HANDLE_VALUE).then_some(Self { find, data, first: true })
    }

    pub fn next(&mut self, name: &mut [u16; 260]) -> Option<(usize, bool, bool)> {
        if self.first {
            self.first = false;
        } else if unsafe { FindNextFileW(self.find, &mut self.data) } == 0 {
            return None;
        }
        let len = self.data.file_name.iter().position(|unit| *unit == 0)?;
        name[..len].copy_from_slice(&self.data.file_name[..len]);
        const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        Some((
            len,
            self.data.attributes & FILE_ATTRIBUTE_DIRECTORY != 0,
            self.data.attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0,
        ))
    }
}

#[cfg(windows)]
impl Drop for NativeDirectory {
    fn drop(&mut self) {
        let _ = unsafe { FindClose(self.find) };
    }
}

#[cfg(windows)]
pub fn processor_count() -> usize {
    unsafe { GetActiveProcessorCount(u16::MAX) }.max(1) as usize
}

#[cfg(windows)]
pub fn process_id() -> u32 {
    unsafe { GetCurrentProcessId() }
}

#[cfg(windows)]
pub fn spawn_detached_process(
    command_line: &mut [u16],
    current_directory: Option<&[u16]>,
) -> bool {
    if command_line.last().copied() != Some(0)
        || current_directory.is_some_and(|path| path.last().copied() != Some(0))
    {
        return false;
    }
    let mut startup = unsafe { core::mem::zeroed::<StartupInfoW>() };
    startup.size = core::mem::size_of::<StartupInfoW>() as u32;
    let mut information = unsafe { core::mem::zeroed::<ProcessInformation>() };
    let created = unsafe {
        CreateProcessW(
            core::ptr::null(),
            command_line.as_mut_ptr(),
            core::ptr::null(),
            core::ptr::null(),
            0,
            0,
            core::ptr::null(),
            current_directory.map_or(core::ptr::null(), |path| path.as_ptr()),
            &mut startup,
            &mut information,
        )
    };
    if created == 0 {
        return false;
    }
    let _ = unsafe { CloseHandle(information.thread) };
    let _ = unsafe { CloseHandle(information.process) };
    true
}

#[cfg(windows)]
pub struct NativeChild {
    process: *mut c_void,
    stdout: *mut c_void,
    stderr: *mut c_void,
    status: Option<i32>,
}

#[cfg(windows)]
unsafe impl Send for NativeChild {}

#[cfg(windows)]
impl NativeChild {
    pub fn spawn_silent(command_line: &mut [u16], current_directory: Option<&[u16]>) -> Option<Self> {
        if command_line.last().copied() != Some(0)
            || current_directory.is_some_and(|path| path.last().copied() != Some(0))
        { return None; }
        const GENERIC_READ: u32 = 0x8000_0000;
        const GENERIC_WRITE: u32 = 0x4000_0000;
        const OPEN_EXISTING: u32 = 3;
        let null_name = [b'N' as u16, b'U' as u16, b'L' as u16, 0];
        let attributes = SecurityAttributes {
            length: core::mem::size_of::<SecurityAttributes>() as u32,
            security_descriptor: core::ptr::null_mut(),
            inherit_handle: 1,
        };
        let open_null = || unsafe { CreateFileW(null_name.as_ptr(), GENERIC_READ | GENERIC_WRITE, 3, (&attributes as *const SecurityAttributes).cast(), OPEN_EXISTING, 0x80, core::ptr::null_mut()) };
        let stdin = open_null();
        let stdout = open_null();
        let stderr = open_null();
        if [stdin, stdout, stderr].iter().any(|handle| *handle as isize == -1) {
            for handle in [stdin, stdout, stderr] { if handle as isize != -1 { let _ = unsafe { CloseHandle(handle) }; } }
            return None;
        }
        let mut startup = unsafe { core::mem::zeroed::<StartupInfoW>() };
        startup.size = core::mem::size_of::<StartupInfoW>() as u32;
        startup.flags = 0x100;
        startup.stdin = stdin;
        startup.stdout = stdout;
        startup.stderr = stderr;
        let mut information = unsafe { core::mem::zeroed::<ProcessInformation>() };
        let created = unsafe { CreateProcessW(core::ptr::null(), command_line.as_mut_ptr(), core::ptr::null(), core::ptr::null(), 1, 0, core::ptr::null(), current_directory.map_or(core::ptr::null(), |path| path.as_ptr()), &mut startup, &mut information) };
        for handle in [stdin, stdout, stderr] { let _ = unsafe { CloseHandle(handle) }; }
        if created == 0 { return None; }
        let _ = unsafe { CloseHandle(information.thread) };
        Some(Self { process: information.process, stdout: core::ptr::null_mut(), stderr: core::ptr::null_mut(), status: None })
    }

    pub fn spawn_captured(command_line: &mut [u16], current_directory: Option<&[u16]>) -> Option<Self> {
        if command_line.last().copied() != Some(0)
            || current_directory.is_some_and(|path| path.last().copied() != Some(0))
        {
            return None;
        }
        let attributes = SecurityAttributes {
            length: core::mem::size_of::<SecurityAttributes>() as u32,
            security_descriptor: core::ptr::null_mut(),
            inherit_handle: 1,
        };
        let mut stdout_read = core::ptr::null_mut();
        let mut stdout_write = core::ptr::null_mut();
        let mut stderr_read = core::ptr::null_mut();
        let mut stderr_write = core::ptr::null_mut();
        if unsafe { CreatePipe(&mut stdout_read, &mut stdout_write, &attributes, 0) } == 0
            || unsafe { CreatePipe(&mut stderr_read, &mut stderr_write, &attributes, 0) } == 0
        {
            for handle in [stdout_read, stdout_write, stderr_read, stderr_write] {
                if !handle.is_null() {
                    let _ = unsafe { CloseHandle(handle) };
                }
            }
            return None;
        }
        const HANDLE_FLAG_INHERIT: u32 = 1;
        let _ = unsafe { SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0) };
        let _ = unsafe { SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0) };
        let mut startup = unsafe { core::mem::zeroed::<StartupInfoW>() };
        startup.size = core::mem::size_of::<StartupInfoW>() as u32;
        startup.flags = 0x100;
        startup.stdin = unsafe { GetStdHandle(u32::MAX - 9) };
        startup.stdout = stdout_write;
        startup.stderr = stderr_write;
        let mut information = unsafe { core::mem::zeroed::<ProcessInformation>() };
        let created = unsafe {
            CreateProcessW(
                core::ptr::null(),
                command_line.as_mut_ptr(),
                core::ptr::null(),
                core::ptr::null(),
                1,
                0,
                core::ptr::null(),
                current_directory.map_or(core::ptr::null(), |path| path.as_ptr()),
                &mut startup,
                &mut information,
            )
        };
        let _ = unsafe { CloseHandle(stdout_write) };
        let _ = unsafe { CloseHandle(stderr_write) };
        if created == 0 {
            let _ = unsafe { CloseHandle(stdout_read) };
            let _ = unsafe { CloseHandle(stderr_read) };
            return None;
        }
        let _ = unsafe { CloseHandle(information.thread) };
        Some(Self { process: information.process, stdout: stdout_read, stderr: stderr_read, status: None })
    }

    fn read_pipe(handle: *mut c_void, buffer: &mut [u8]) -> usize {
        let mut available = 0u32;
        if unsafe {
            PeekNamedPipe(
                handle,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
                &mut available,
                core::ptr::null_mut(),
            )
        } == 0 || available == 0
        {
            return 0;
        }
        let mut read = 0u32;
        let count = buffer.len().min(available as usize).min(u32::MAX as usize) as u32;
        if unsafe { ReadFile(handle, buffer.as_mut_ptr().cast(), count, &mut read, core::ptr::null_mut()) } == 0 {
            0
        } else {
            read as usize
        }
    }

    pub fn read_stdout(&mut self, buffer: &mut [u8]) -> usize { Self::read_pipe(self.stdout, buffer) }
    pub fn read_stderr(&mut self, buffer: &mut [u8]) -> usize { Self::read_pipe(self.stderr, buffer) }

    pub fn try_wait(&mut self) -> Option<i32> {
        if self.status.is_none() && unsafe { WaitForSingleObject(self.process, 0) } == 0 {
            let mut code = 0;
            if unsafe { GetExitCodeProcess(self.process, &mut code) } != 0 {
                self.status = Some(code as i32);
            }
        }
        self.status
    }

    pub fn kill(&mut self) -> bool {
        self.status.is_some() || unsafe { TerminateProcess(self.process, 1) } != 0
    }

    pub fn wait(&mut self) -> Option<i32> {
        if self.status.is_none() {
            if unsafe { WaitForSingleObject(self.process, u32::MAX) } != 0 { return None; }
            let mut code = 0;
            if unsafe { GetExitCodeProcess(self.process, &mut code) } == 0 { return None; }
            self.status = Some(code as i32);
        }
        self.status
    }
}

#[cfg(windows)]
impl Drop for NativeChild {
    fn drop(&mut self) {
        if self.status.is_none() {
            let _ = unsafe { WaitForSingleObject(self.process, u32::MAX) };
        }
        for handle in [self.process, self.stdout, self.stderr] {
            if !handle.is_null() { let _ = unsafe { CloseHandle(handle) }; }
        }
    }
}

#[cfg(windows)]
pub fn yield_now() {
    let _ = unsafe { SwitchToThread() };
}

#[cfg(windows)]
pub fn wait_on_u32(address: *const u32, expected: u32) {
    let _ = unsafe {
        WaitOnAddress(
            address.cast(),
            (&expected as *const u32).cast(),
            core::mem::size_of::<u32>(),
            u32::MAX,
        )
    };
}

#[cfg(windows)]
pub fn wake_one_u32(address: *const u32) {
    unsafe { WakeByAddressSingle(address.cast()) };
}

#[cfg(windows)]
pub fn wake_all_u32(address: *const u32) {
    unsafe { WakeByAddressAll(address.cast()) };
}

#[cfg(windows)]
pub unsafe fn spawn_detached_thread(
    context: *mut c_void,
    start: unsafe extern "system" fn(*mut c_void) -> u32,
) -> bool {
    let handle = unsafe {
        CreateThread(
            core::ptr::null(),
            0,
            start,
            context,
            0,
            core::ptr::null_mut(),
        )
    };
    if handle.is_null() {
        false
    } else {
        let _ = unsafe { CloseHandle(handle) };
        true
    }
}

#[derive(Debug)]
#[cfg(windows)]
pub struct NativeFile(*mut c_void);

#[cfg(windows)]
unsafe impl Send for NativeFile {}

#[cfg(windows)]
impl NativeFile {
    pub fn open_read(path: &[u16]) -> Option<Self> {
        const GENERIC_READ: u32 = 0x8000_0000;
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_WRITE: u32 = 2;
        const FILE_SHARE_DELETE: u32 = 4;
        const OPEN_EXISTING: u32 = 3;
        const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                core::ptr::null(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                core::ptr::null_mut(),
            )
        };
        (handle as isize != -1).then_some(Self(handle))
    }

    pub fn create_write(path: &[u16]) -> Option<Self> {
        const GENERIC_WRITE: u32 = 0x4000_0000;
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_WRITE: u32 = 2;
        const FILE_SHARE_DELETE: u32 = 4;
        const CREATE_ALWAYS: u32 = 2;
        const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                core::ptr::null(),
                CREATE_ALWAYS,
                FILE_ATTRIBUTE_NORMAL,
                core::ptr::null_mut(),
            )
        };
        (handle as isize != -1).then_some(Self(handle))
    }

    pub fn read(&self, buffer: &mut [u8]) -> Option<usize> {
        let amount = buffer.len().min(u32::MAX as usize) as u32;
        let mut read = 0;
        (unsafe {
            ReadFile(
                self.0,
                buffer.as_mut_ptr().cast(),
                amount,
                &mut read,
                core::ptr::null_mut(),
            )
        } != 0)
            .then_some(read as usize)
    }

    pub fn write(&self, buffer: &[u8]) -> Option<usize> {
        let amount = buffer.len().min(u32::MAX as usize) as u32;
        let mut written = 0;
        (unsafe {
            WriteFile(
                self.0,
                buffer.as_ptr().cast(),
                amount,
                &mut written,
                core::ptr::null_mut(),
            )
        } != 0)
            .then_some(written as usize)
    }

    pub fn seek_absolute(&self, position: u64) -> bool {
        position <= i64::MAX as u64
            && unsafe { SetFilePointerEx(self.0, position as i64, core::ptr::null_mut(), 0) } != 0
    }

    pub fn len(&self) -> Option<u64> {
        let mut size = 0i64;
        (unsafe { GetFileSizeEx(self.0, &mut size) } != 0 && size >= 0).then_some(size as u64)
    }

    pub fn raw_handle(&self) -> *mut c_void {
        self.0
    }
}

#[cfg(windows)]
impl Drop for NativeFile {
    fn drop(&mut self) {
        let _ = unsafe { CloseHandle(self.0) };
    }
}

#[cfg(windows)]
pub fn stdout_is_terminal() -> bool {
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    let handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
    let mut mode = 0;
    !handle.is_null() && unsafe { GetConsoleMode(handle, &mut mode) } != 0
}

#[cfg(windows)]
fn write_standard_handle(kind: u32, mut bytes: &[u8]) -> bool {
    let handle = unsafe { GetStdHandle(kind) };
    if handle.is_null() || handle == usize::MAX as *mut c_void {
        return false;
    }
    while !bytes.is_empty() {
        let mut written = 0u32;
        let count = bytes.len().min(u32::MAX as usize) as u32;
        if unsafe {
            WriteFile(
                handle,
                bytes.as_ptr().cast(),
                count,
                &mut written,
                core::ptr::null_mut(),
            )
        } == 0 || written == 0
        {
            return false;
        }
        bytes = &bytes[written as usize..];
    }
    true
}

#[cfg(windows)]
pub fn write_stdout(bytes: &[u8]) -> bool {
    write_standard_handle(u32::MAX - 10, bytes)
}

#[cfg(windows)]
pub fn write_stderr(bytes: &[u8]) -> bool {
    write_standard_handle(u32::MAX - 11, bytes)
}

#[cfg(windows)]
pub fn read_stdin(buffer: &mut [u8]) -> Option<usize> {
    const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    let handle = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
    if handle.is_null() || handle as isize == -1 {
        return None;
    }
    let amount = buffer.len().min(u32::MAX as usize) as u32;
    let mut read = 0;
    (unsafe {
        ReadFile(
            handle,
            buffer.as_mut_ptr().cast(),
            amount,
            &mut read,
            core::ptr::null_mut(),
        )
    } != 0)
        .then_some(read as usize)
}

#[cfg(windows)]
pub fn environment_variable_is_set(name: &CStr) -> bool {
    const ERROR_ENVVAR_NOT_FOUND: u32 = 203;
    unsafe { SetLastError(0) };
    let length = unsafe { GetEnvironmentVariableA(name.as_ptr(), core::ptr::null_mut(), 0) };
    length != 0 || unsafe { GetLastError() } != ERROR_ENVVAR_NOT_FOUND
}

#[cfg(windows)]
pub fn environment_variable_equals(name: &CStr, expected: &[u8]) -> bool {
    let mut value = [0i8; 64];
    let length =
        unsafe { GetEnvironmentVariableA(name.as_ptr(), value.as_mut_ptr(), value.len() as u32) }
            as usize;
    length == expected.len()
        && length < value.len()
        && value[..length]
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| actual as u8 == expected)
}

#[cfg(windows)]
pub fn environment_variable_wide(name: &[u16], value: &mut [u16]) -> Result<usize, usize> {
    const ERROR_ENVVAR_NOT_FOUND: u32 = 203;
    if name.last().copied() != Some(0) {
        return Err(0);
    }
    unsafe { SetLastError(0) };
    let length = unsafe {
        GetEnvironmentVariableW(
            name.as_ptr(),
            value.as_mut_ptr(),
            value.len().min(u32::MAX as usize) as u32,
        )
    } as usize;
    if length == 0 {
        if unsafe { GetLastError() } == ERROR_ENVVAR_NOT_FOUND {
            Err(0)
        } else {
            Ok(0)
        }
    } else if length >= value.len() {
        Err(length)
    } else {
        Ok(length)
    }
}

#[cfg(windows)]
pub fn exit_process(exit_code: i32) -> ! {
    unsafe { ExitProcess(exit_code as u32) }
}

#[cfg(windows)]
pub unsafe fn map_file_read_only(file: *mut c_void, len: usize) -> Option<NonNull<u8>> {
    const PAGE_READONLY: u32 = 2;
    const FILE_MAP_READ: u32 = 4;
    let mapping = unsafe {
        CreateFileMappingW(
            file,
            core::ptr::null(),
            PAGE_READONLY,
            0,
            0,
            core::ptr::null(),
        )
    };
    if mapping.is_null() {
        return None;
    }
    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
    let _ = unsafe { CloseHandle(mapping) };
    NonNull::new(view.cast())
}

#[cfg(windows)]
pub unsafe fn unmap_file(address: NonNull<u8>, _: usize) -> bool {
    unsafe { UnmapViewOfFile(address.as_ptr().cast()) != 0 }
}

#[derive(Debug)]
#[cfg(windows)]
pub struct DynamicLibrary(*mut c_void);

#[cfg(windows)]
unsafe impl Send for DynamicLibrary {}
#[cfg(windows)]
unsafe impl Sync for DynamicLibrary {}

#[cfg(windows)]
impl DynamicLibrary {
    pub fn open(name: &core::ffi::CStr) -> Option<Self> {
        let handle = unsafe { LoadLibraryA(name.as_ptr()) };
        (!handle.is_null()).then_some(Self(handle))
    }

    pub fn symbol(&self, name: &core::ffi::CStr) -> Option<*mut c_void> {
        let symbol = unsafe { GetProcAddress(self.0, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }
}

#[cfg(windows)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        let _ = unsafe { FreeLibrary(self.0) };
    }
}

pub struct SystemAllocator;

#[cfg(windows)]
unsafe impl GlobalAlloc for SystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let heap = unsafe { GetProcessHeap() };
        unsafe { HeapAlloc(heap, 0, layout.size().max(1)) }.cast()
    }

    unsafe fn dealloc(&self, pointer: *mut u8, _: Layout) {
        if !pointer.is_null() {
            let heap = unsafe { GetProcessHeap() };
            let _ = unsafe { HeapFree(heap, 0, pointer.cast()) };
        }
    }

    unsafe fn realloc(&self, pointer: *mut u8, _: Layout, size: usize) -> *mut u8 {
        let heap = unsafe { GetProcessHeap() };
        unsafe { HeapReAlloc(heap, 0, pointer.cast(), size.max(1)) }.cast()
    }
}

#[cfg(windows)]
pub fn sleep_millis(milliseconds: u64) {
    let milliseconds = milliseconds.min(u32::MAX as u64) as u32;
    unsafe { Sleep(milliseconds) };
}

#[cfg(windows)]
pub fn monotonic_nanos() -> u64 {
    let mut counter = 0i64;
    let mut frequency = 0i64;
    if unsafe { QueryPerformanceCounter(&mut counter) } == 0
        || unsafe { QueryPerformanceFrequency(&mut frequency) } == 0
        || frequency <= 0
    {
        return 0;
    }
    ((counter as u128 * 1_000_000_000) / frequency as u128) as u64
}

#[cfg(windows)]
pub fn unix_time_millis() -> u64 {
    const WINDOWS_TO_UNIX_100NS: u64 = 116_444_736_000_000_000;
    let mut value = FileTime { low: 0, high: 0 };
    unsafe { GetSystemTimeAsFileTime(&mut value) };
    let ticks = (value.high as u64) << 32 | value.low as u64;
    ticks.saturating_sub(WINDOWS_TO_UNIX_100NS) / 10_000
}

#[cfg(windows)]
pub fn local_time() -> LocalTime {
    let mut value = LocalTime::default();
    // SAFETY: value points to writable storage matching the SYSTEMTIME ABI.
    unsafe { GetLocalTime(&mut value) };
    value
}

#[cfg(not(windows))]
pub fn sleep_millis(_: u64) {}

#[cfg(not(windows))]
pub fn monotonic_nanos() -> u64 {
    0
}

#[cfg(not(windows))]
pub fn unix_time_millis() -> u64 {
    0
}

#[cfg(not(windows))]
pub fn local_time() -> LocalTime {
    LocalTime::default()
}

#[cfg(all(test, windows))]
mod tests {
    extern crate std;

    use core::alloc::{GlobalAlloc, Layout};

    #[test]
    fn native_local_time_has_valid_calendar_fields() {
        let time = super::local_time();
        assert!(time.year >= 2020);
        assert!((1..=12).contains(&time.month));
        assert!((1..=31).contains(&time.day));
        assert!(time.hour < 24 && time.minute < 60 && time.second < 60);
    }

    #[test]
    fn reads_process_environment_without_allocation() {
        assert!(super::environment_variable_is_set(c"PATH"));
        let _ = super::stdout_is_terminal();
    }

    #[test]
    fn monotonic_clock_advances_across_native_sleep() {
        assert!(super::unix_time_millis() > 1_000_000_000_000);
        let before = super::monotonic_nanos();
        super::sleep_millis(2);
        assert!(super::monotonic_nanos() > before);
    }

    #[test]
    fn native_allocator_round_trips_memory() {
        let layout = Layout::from_size_align(64, 8).unwrap();
        let pointer = unsafe { GlobalAlloc::alloc(&super::SystemAllocator, layout) };
        assert!(!pointer.is_null());
        unsafe {
            pointer.write(0x5a);
            assert_eq!(pointer.read(), 0x5a);
            GlobalAlloc::dealloc(&super::SystemAllocator, pointer, layout);
        }
    }

    #[test]
    fn loads_native_library_symbols() {
        let library = super::DynamicLibrary::open(c"kernel32.dll").unwrap();
        assert!(library.symbol(c"GetCurrentProcessId").is_some());
        assert!(library.symbol(c"definitely_missing_symbol").is_none());
    }

    #[test]
    fn maps_native_file_handle_read_only() {
        use std::io::Write;
        use std::os::windows::io::AsRawHandle;

        let path =
            std::env::temp_dir().join(std::format!("mrml-windows-map-{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"native mapping").unwrap();
        drop(file);
        let file = std::fs::File::open(&path).unwrap();
        let mapping =
            unsafe { super::map_file_read_only(file.as_raw_handle().cast(), 14) }.unwrap();
        assert_eq!(
            unsafe { core::slice::from_raw_parts(mapping.as_ptr(), 14) },
            b"native mapping"
        );
        assert!(unsafe { super::unmap_file(mapping, 14) });
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
