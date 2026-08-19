use core::alloc::{GlobalAlloc, Layout};
use core::marker::Unsize;
use core::ops::{CoerceUnsized, Deref, DerefMut};
use core::ptr::NonNull;

/// A uniquely owned value allocated by the MRML platform allocator.
///
/// `Owned` supports nightly unsizing so callers can own trait objects without
/// using Rust's global `alloc` crate.
pub struct Owned<T: ?Sized> {
    pointer: NonNull<T>,
}

unsafe impl<T: Send + ?Sized> Send for Owned<T> {}
unsafe impl<T: Sync + ?Sized> Sync for Owned<T> {}
// `Owned` never changes its pointee address, moves out through `DerefMut`, or
// deallocates before dropping the pointee in place.
unsafe impl<T: ?Sized> core::pin::PinSafePointer for Owned<T> {}

impl<T> Owned<T> {
    pub fn new(value: T) -> Self {
        let layout = Layout::new::<T>();
        let pointer = unsafe { allocator().alloc(layout) }.cast::<T>();
        let pointer = NonNull::new(pointer).unwrap_or_else(|| panic!("MRML allocation failed"));
        unsafe { pointer.as_ptr().write(value) };
        Self { pointer }
    }
}

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Owned<U>> for Owned<T> where T: Unsize<U> {}

impl<T: ?Sized> Deref for Owned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.pointer.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for Owned<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.pointer.as_mut() }
    }
}

impl<T: ?Sized> Drop for Owned<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::for_value(self.pointer.as_ref());
            self.pointer.as_ptr().drop_in_place();
            allocator().dealloc(self.pointer.as_ptr().cast(), layout);
        }
    }
}

#[cfg(windows)]
fn allocator() -> mrml_windows::SystemAllocator {
    mrml_windows::SystemAllocator
}

#[cfg(unix)]
fn allocator() -> mrml_linux::SystemAllocator {
    mrml_linux::SystemAllocator
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    trait DynamicValue {
        fn value(&self) -> usize;
    }

    struct Value(usize);

    impl DynamicValue for Value {
        fn value(&self) -> usize {
            self.0
        }
    }

    impl Drop for Value {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn owns_and_drops_dynamic_values() {
        DROPS.store(0, Ordering::Relaxed);
        let value: Owned<dyn DynamicValue> = Owned::new(Value(31));
        assert_eq!(value.value(), 31);
        drop(value);
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);
    }
}
