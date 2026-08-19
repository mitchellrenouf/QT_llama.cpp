use alloc::vec::Vec;

#[inline]
pub fn for_each_range<F>(len: usize, _: usize, operation: F)
where
    F: Fn(usize, usize) + Sync,
{
    if len != 0 {
        operation(0, len);
    }
}

pub fn map_collect<T, F>(len: usize, operation: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    (0..len).map(operation).collect()
}
