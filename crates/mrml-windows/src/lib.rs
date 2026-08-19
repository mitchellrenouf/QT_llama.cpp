#![no_std]

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

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetLocalTime(time: *mut LocalTime);
}

#[cfg(windows)]
pub fn local_time() -> LocalTime {
    let mut value = LocalTime::default();
    // SAFETY: value points to writable storage matching the SYSTEMTIME ABI.
    unsafe { GetLocalTime(&mut value) };
    value
}

#[cfg(not(windows))]
pub fn local_time() -> LocalTime {
    LocalTime::default()
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn native_local_time_has_valid_calendar_fields() {
        let time = super::local_time();
        assert!(time.year >= 2020);
        assert!((1..=12).contains(&time.month));
        assert!((1..=31).contains(&time.day));
        assert!(time.hour < 24 && time.minute < 60 && time.second < 60);
    }
}
