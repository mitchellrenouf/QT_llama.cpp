use core::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Instant(u64);

impl Instant {
    pub fn now() -> Self {
        #[cfg(windows)]
        let nanos = mrml_windows::monotonic_nanos();
        #[cfg(unix)]
        let nanos = mrml_linux::monotonic_nanos();
        Self(nanos)
    }

    pub fn elapsed(self) -> Duration {
        let now = Self::now().0;
        Duration::from_nanos(now.saturating_sub(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_elapsed_time_advances() {
        let start = Instant::now();
        #[cfg(windows)]
        mrml_windows::sleep_millis(2);
        #[cfg(unix)]
        mrml_linux::sleep_millis(2);
        assert!(start.elapsed() >= Duration::from_millis(1));
    }
}
