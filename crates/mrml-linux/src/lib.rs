#![no_std]

#[cfg(unix)]
use core::ffi::{c_int, c_long};

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
