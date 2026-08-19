//! Native platform paths and timestamps shared by tools and the agent.
use std::path::PathBuf;

#[cfg(windows)]
use mrml_windows::LocalTime;
#[cfg(unix)]
use mrml_linux::LocalTime;

pub fn unix_timestamp_millis() -> u128 {
    platform_unix_time_millis() as u128
}

pub fn monotonic_timestamp_nanos() -> u64 {
    #[cfg(windows)]
    {
        mrml_windows::monotonic_nanos()
    }
    #[cfg(unix)]
    {
        mrml_linux::monotonic_nanos()
    }
}

#[cfg(windows)]
fn platform_unix_time_millis() -> u64 { mrml_windows::unix_time_millis() }

#[cfg(unix)]
fn platform_unix_time_millis() -> u64 { mrml_linux::unix_time_millis() }

pub fn sleep_millis(milliseconds: u64) {
    #[cfg(windows)]
    mrml_windows::sleep_millis(milliseconds);
    #[cfg(unix)]
    mrml_linux::sleep_millis(milliseconds);
}

pub fn exit_process(status: i32) -> ! {
    #[cfg(windows)]
    {
        mrml_windows::exit_process(status)
    }
    #[cfg(unix)]
    {
        mrml_linux::exit_process(status)
    }
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
    mrml_windows::local_time()
}

#[cfg(unix)]
fn local_time() -> LocalTime {
    let seconds = (mrml_linux::unix_time_millis() / 1000) as i64;
    mrml_linux::local_time(seconds).unwrap_or(LocalTime {
        year: 1970, month: 1, day: 1, weekday: 4, hour: 0, minute: 0, second: 0,
    })
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
