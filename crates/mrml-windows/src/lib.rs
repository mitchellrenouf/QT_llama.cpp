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

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLocalTime(time: *mut LocalTime);
    fn GetSystemTimeAsFileTime(time: *mut FileTime);
    fn GetProcessHeap() -> *mut c_void;
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
pub fn processor_count() -> usize {
    unsafe { GetActiveProcessorCount(u16::MAX) }.max(1) as usize
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
