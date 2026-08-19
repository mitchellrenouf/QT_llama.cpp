#![no_std]

#[cfg(unix)]
use core::alloc::{GlobalAlloc, Layout};
#[cfg(unix)]
use core::ffi::c_void;
#[cfg(unix)]
use core::ffi::{c_int, c_long};

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
    let mut value = Timespec { seconds: 0, nanoseconds: 0 };
    if unsafe { clock_gettime(CLOCK_MONOTONIC, &mut value) } != 0 {
        return 0;
    }
    value.seconds as u64 * 1_000_000_000 + value.nanoseconds as u64
}

#[cfg(unix)]
pub fn unix_time_millis() -> u64 {
    const CLOCK_REALTIME: c_int = 0;
    let mut value = Timespec { seconds: 0, nanoseconds: 0 };
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
pub fn monotonic_nanos() -> u64 { 0 }

#[cfg(not(unix))]
pub fn unix_time_millis() -> u64 { 0 }

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
    fn loads_native_library_symbols() {
        let library = super::DynamicLibrary::open(c"libc.so.6").unwrap();
        assert!(library.symbol(c"getpid").is_some());
        assert!(library.symbol(c"definitely_missing_symbol").is_none());
    }
}
