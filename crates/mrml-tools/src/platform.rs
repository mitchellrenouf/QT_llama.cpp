//! Native platform paths and timestamps shared by tools and the agent.
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
struct LocalTime {
    year: u16,
    month: u16,
    day: u16,
    weekday: u16,
    hour: u16,
    minute: u16,
    second: u16,
}

pub fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub fn local_date_string() -> String {
    const WEEKDAYS: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let time = local_time();
    let weekday = WEEKDAYS
        .get(time.weekday as usize)
        .copied()
        .unwrap_or("Unknown");
    let month = time
        .month
        .checked_sub(1)
        .and_then(|index| MONTHS.get(index as usize))
        .copied()
        .unwrap_or("Unknown");
    format!("{weekday}, {month} {:2}, {}", time.day, time.year)
}

pub fn local_timestamp_string() -> String {
    let time = local_time();
    format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        time.year, time.month, time.day, time.hour, time.minute, time.second
    )
}

#[cfg(windows)]
fn local_time() -> LocalTime {
    #[repr(C)]
    #[derive(Default)]
    struct SystemTime {
        year: u16,
        month: u16,
        weekday: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
        milliseconds: u16,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLocalTime(time: *mut SystemTime);
    }
    let mut value = SystemTime::default();
    unsafe { GetLocalTime(&mut value) };
    LocalTime {
        year: value.year,
        month: value.month,
        day: value.day,
        weekday: value.weekday,
        hour: value.hour,
        minute: value.minute,
        second: value.second,
    }
}

#[cfg(unix)]
fn local_time() -> LocalTime {
    use std::ffi::{c_int, c_long};
    #[repr(C)]
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
    unsafe extern "C" {
        fn localtime_r(clock: *const c_long, result: *mut Tm) -> *mut Tm;
    }
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as c_long;
    let mut value = std::mem::MaybeUninit::<Tm>::uninit();
    let result = unsafe { localtime_r(&seconds, value.as_mut_ptr()) };
    if result.is_null() {
        return LocalTime {
            year: 1970,
            month: 1,
            day: 1,
            weekday: 4,
            hour: 0,
            minute: 0,
            second: 0,
        };
    }
    let value = unsafe { value.assume_init() };
    LocalTime {
        year: (value.year + 1900) as u16,
        month: (value.month + 1) as u16,
        day: value.day as u16,
        weekday: value.weekday as u16,
        hour: value.hour as u16,
        minute: value.minute as u16,
        second: value.second as u16,
    }
}

pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                let mut home = PathBuf::from(drive);
                home.push(path);
                Some(home)
            })
    }

    #[cfg(not(windows))]
    {
        std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    }
}

pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    }

    #[cfg(target_os = "macos")]
    {
        home_dir().map(|home| home.join("Library").join("Caches"))
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|home| home.join(".cache")))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn discovered_directories_are_absolute() {
        assert!(super::home_dir().is_none_or(|path| path.is_absolute()));
        assert!(super::cache_dir().is_none_or(|path| path.is_absolute()));
    }

    #[test]
    fn timestamps_have_stable_protocol_shapes() {
        let timestamp = super::local_timestamp_string();
        assert_eq!(timestamp.len(), 19);
        assert_eq!(&timestamp[4..5], "-");
        assert_eq!(&timestamp[10..11], "_");
        assert!(super::local_date_string().contains(','));
        assert!(super::unix_timestamp_millis() > 1_000_000_000_000);
    }
}
