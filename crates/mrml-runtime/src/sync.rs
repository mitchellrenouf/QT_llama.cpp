use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering, fence};

pub struct SpinMutex<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinMutex<T> {}
unsafe impl<T: Send> Sync for SpinMutex<T> {}

impl<T> SpinMutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    pub fn lock(&self) -> SpinMutexGuard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.locked.load(Ordering::Relaxed) {
                spin_loop();
            }
        }
        SpinMutexGuard { mutex: self }
    }
}

pub struct SpinMutexGuard<'a, T> {
    mutex: &'a SpinMutex<T>,
}

impl<T> Deref for SpinMutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.value.get() }
    }
}
impl<T> DerefMut for SpinMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.value.get() }
    }
}
impl<T> Drop for SpinMutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

pub struct OnceCell<T> {
    state: AtomicU8,
    value: UnsafeCell<MaybeUninit<T>>,
}

unsafe impl<T: Send + Sync> Sync for OnceCell<T> {}
unsafe impl<T: Send> Send for OnceCell<T> {}

impl<T> OnceCell<T> {
    pub const fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if self.state.load(Ordering::Acquire) == 2 {
            Some(unsafe { (&*self.value.get()).assume_init_ref() })
        } else {
            None
        }
    }

    pub fn set(&self, value: T) -> Result<(), T> {
        if self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err(value);
        }
        unsafe { (*self.value.get()).write(value) };
        self.state.store(2, Ordering::Release);
        Ok(())
    }

    pub fn get_or_init(&self, initialize: impl FnOnce() -> T) -> &T {
        if let Some(value) = self.get() {
            return value;
        }
        if self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Acquire)
            .is_ok()
        {
            struct ResetOnPanic<'a>(&'a AtomicU8);
            impl Drop for ResetOnPanic<'_> {
                fn drop(&mut self) {
                    self.0.store(0, Ordering::Release);
                }
            }
            let reset = ResetOnPanic(&self.state);
            unsafe { (*self.value.get()).write(initialize()) };
            self.state.store(2, Ordering::Release);
            core::mem::forget(reset);
        } else {
            while self.state.load(Ordering::Acquire) != 2 {
                spin_loop();
            }
        }
        unsafe { (&*self.value.get()).assume_init_ref() }
    }
}

impl<T> Drop for OnceCell<T> {
    fn drop(&mut self) {
        if *self.state.get_mut() == 2 {
            unsafe { self.value.get_mut().assume_init_drop() };
        }
    }
}

#[repr(C)]
struct SharedInner<T> {
    references: AtomicUsize,
    value: T,
}

pub struct Shared<T> {
    inner: NonNull<SharedInner<T>>,
}

unsafe impl<T: Send + Sync> Send for Shared<T> {}
unsafe impl<T: Send + Sync> Sync for Shared<T> {}

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        let layout = Layout::new::<SharedInner<T>>();
        let pointer = unsafe { allocator().alloc(layout) }.cast::<SharedInner<T>>();
        let inner = NonNull::new(pointer).unwrap_or_else(|| panic!("MRML allocation failed"));
        unsafe {
            inner.as_ptr().write(SharedInner {
                references: AtomicUsize::new(1),
                value,
            });
        }
        Self { inner }
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        unsafe { self.inner.as_ref() }
            .references
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |references| {
                (references < isize::MAX as usize).then_some(references + 1)
            })
            .expect("MRML shared count overflow");
        Self { inner: self.inner }
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &unsafe { self.inner.as_ref() }.value
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        if unsafe { self.inner.as_ref() }
            .references
            .fetch_sub(1, Ordering::Release)
            != 1
        {
            return;
        }
        fence(Ordering::Acquire);
        unsafe {
            self.inner.as_ptr().drop_in_place();
            allocator().dealloc(self.inner.as_ptr().cast(), Layout::new::<SharedInner<T>>());
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

    #[test]
    fn mutex_once_and_shared_ownership_work_without_std_types() {
        let mutex = SpinMutex::new(3);
        *mutex.lock() += 4;
        assert_eq!(*mutex.lock(), 7);

        let once = OnceCell::new();
        assert_eq!(*once.get_or_init(|| 11), 11);
        assert_eq!(*once.get_or_init(|| 99), 11);

        let shared = Shared::new(17);
        let clone = shared.clone();
        assert_eq!(*shared, 17);
        drop(shared);
        assert_eq!(*clone, 17);
    }
}
