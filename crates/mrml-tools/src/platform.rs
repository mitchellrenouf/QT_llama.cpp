//! Native platform paths and timestamps shared by tools and the agent.
use core::fmt::Write as _;
use mrml_runtime::Text;
use std::path::Path;

pub fn path_is_file(path: &Path) -> bool {
    path.to_str().is_some_and(mrml_runtime::path_is_file)
}

#[cfg(unix)]
use mrml_linux::LocalTime;
#[cfg(windows)]
use mrml_windows::LocalTime;

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
fn platform_unix_time_millis() -> u64 {
    mrml_windows::unix_time_millis()
}

#[cfg(unix)]
fn platform_unix_time_millis() -> u64 {
    mrml_linux::unix_time_millis()
}

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

pub fn stdout_is_terminal() -> bool {
    #[cfg(windows)]
    {
        mrml_windows::stdout_is_terminal()
    }
    #[cfg(unix)]
    {
        mrml_linux::stdout_is_terminal()
    }
}

pub fn local_date_string() -> Text {
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
    let mut output = Text::new();
    write!(output, "{weekday}, {month} {:2}, {}", time.day, time.year)
        .expect("MRML timestamp allocation failed");
    output
}

pub fn local_timestamp_string() -> Text {
    let time = local_time();
    let mut output = Text::new();
    write!(
        output,
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        time.year, time.month, time.day, time.hour, time.minute, time.second
    )
    .expect("MRML timestamp allocation failed");
    output
}

#[cfg(windows)]
fn local_time() -> LocalTime {
    mrml_windows::local_time()
}

#[cfg(unix)]
fn local_time() -> LocalTime {
    let seconds = (mrml_linux::unix_time_millis() / 1000) as i64;
    mrml_linux::local_time(seconds).unwrap_or(LocalTime {
        year: 1970,
        month: 1,
        day: 1,
        weekday: 4,
        hour: 0,
        minute: 0,
        second: 0,
    })
}

pub fn home_dir() -> Option<Text> {
    #[cfg(windows)]
    {
        mrml_runtime::environment_variable("USERPROFILE")
            .filter(|value| !value.is_empty())
            .map(|value| value)
            .or_else(|| {
                let drive = mrml_runtime::environment_variable("HOMEDRIVE")?;
                let path = mrml_runtime::environment_variable("HOMEPATH")?;
                Some(mrml_runtime::join_path(&drive, &path))
            })
    }

    #[cfg(not(windows))]
    {
        mrml_runtime::environment_variable("HOME")
            .filter(|value| !value.is_empty())
            .map(|value| value)
    }
}

pub fn cache_dir() -> Option<Text> {
    #[cfg(windows)]
    {
        mrml_runtime::environment_variable("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .map(|value| value)
    }

    #[cfg(target_os = "macos")]
    {
        home_dir().map(|home| {
            mrml_runtime::join_path(&mrml_runtime::join_path(&home, "Library"), "Caches")
        })
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        mrml_runtime::environment_variable("XDG_CACHE_HOME")
            .filter(|value| !value.is_empty())
            .or_else(|| home_dir().map(|home| mrml_runtime::join_path(&home, ".cache")))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn discovered_directories_are_absolute() {
        assert!(super::home_dir().is_none_or(|path| mrml_runtime::path_is_absolute(&path)));
        assert!(super::cache_dir().is_none_or(|path| mrml_runtime::path_is_absolute(&path)));
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
