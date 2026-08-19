use mrml_runtime::Vector;

#[inline]
pub fn for_each_range<F>(len: usize, _: usize, operation: F)
where
    F: Fn(usize, usize) + Sync,
{
    if len != 0 {
        operation(0, len);
    }
}

pub fn map<T, F>(len: usize, _: usize, operation: F) -> Vector<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    (0..len).map(operation).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_adapter_visits_ranges_and_collects_values() {
        use core::sync::atomic::{AtomicUsize, Ordering};

        let start_seen = AtomicUsize::new(usize::MAX);
        let end_seen = AtomicUsize::new(usize::MAX);
        for_each_range(9, 4, |start, end| {
            start_seen.store(start, Ordering::Relaxed);
            end_seen.store(end, Ordering::Relaxed);
        });
        assert_eq!(start_seen.load(Ordering::Relaxed), 0);
        assert_eq!(end_seen.load(Ordering::Relaxed), 9);
        assert_eq!(&map(4, 1, |index| index * index)[..], &[0, 1, 4, 9]);
    }
}
