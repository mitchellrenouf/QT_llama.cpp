#![no_std]

#[cfg(windows)]
use core::alloc::{GlobalAlloc, Layout};
#[cfg(windows)]
use core::ffi::c_void;
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
    fn HeapReAlloc(heap: *mut c_void, flags: u32, memory: *mut c_void, bytes: usize) -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, memory: *mut c_void) -> i32;
    fn QueryPerformanceCounter(value: *mut i64) -> i32;
    fn QueryPerformanceFrequency(value: *mut i64) -> i32;
    fn Sleep(milliseconds: u32);
    fn LoadLibraryA(name: *const i8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const i8) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
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
pub fn monotonic_nanos() -> u64 { 0 }

#[cfg(not(windows))]
pub fn unix_time_millis() -> u64 { 0 }

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

        let path = std::env::temp_dir().join(std::format!("mrml-windows-map-{}.bin", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"native mapping").unwrap();
        drop(file);
        let file = std::fs::File::open(&path).unwrap();
        let mapping = unsafe { super::map_file_read_only(file.as_raw_handle().cast(), 14) }.unwrap();
        assert_eq!(unsafe { core::slice::from_raw_parts(mapping.as_ptr(), 14) }, b"native mapping");
        assert!(unsafe { super::unmap_file(mapping, 14) });
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
