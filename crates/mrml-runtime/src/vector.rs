use core::alloc::{GlobalAlloc, Layout};
use core::fmt;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryReserveError {
    CapacityOverflow,
    AllocationFailed,
}

pub struct Vector<T> {
    pointer: NonNull<T>,
    length: usize,
    capacity: usize,
}

impl<T> Vector<T> {
    pub const fn new() -> Self {
        Self {
            pointer: NonNull::dangling(),
            length: 0,
            capacity: if core::mem::size_of::<T>() == 0 {
                usize::MAX
            } else {
                0
            },
        }
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        let mut values = Self::new();
        values.try_reserve_exact(capacity)?;
        Ok(values)
    }

    pub const fn len(&self) -> usize {
        self.length
    }
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn try_push(&mut self, value: T) -> Result<(), TryReserveError> {
        if self.length == self.capacity {
            self.try_reserve(1)?;
        }
        unsafe { self.pointer.as_ptr().add(self.length).write(value) };
        self.length += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        Some(unsafe { self.pointer.as_ptr().add(self.length).read() })
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn truncate(&mut self, length: usize) {
        if length >= self.length {
            return;
        }
        let old_length = self.length;
        self.length = length;
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.pointer.as_ptr().add(length),
                old_length - length,
            ))
        };
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required = self
            .length
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;
        if required <= self.capacity {
            return Ok(());
        }
        let doubled = self.capacity.saturating_mul(2).max(4);
        self.grow(required.max(doubled))
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required = self
            .length
            .checked_add(additional)
            .ok_or(TryReserveError::CapacityOverflow)?;
        if required <= self.capacity {
            Ok(())
        } else {
            self.grow(required)
        }
    }

    pub fn try_extend_from_slice(&mut self, values: &[T]) -> Result<(), TryReserveError>
    where
        T: Clone,
    {
        self.try_reserve(values.len())?;
        for value in values {
            unsafe { self.pointer.as_ptr().add(self.length).write(value.clone()) };
            self.length += 1;
        }
        Ok(())
    }

    pub fn try_resize(&mut self, length: usize, value: T) -> Result<(), TryReserveError>
    where
        T: Clone,
    {
        if length <= self.length {
            self.truncate(length);
            return Ok(());
        }
        self.try_reserve(length - self.length)?;
        while self.length < length {
            unsafe { self.pointer.as_ptr().add(self.length).write(value.clone()) };
            self.length += 1;
        }
        Ok(())
    }

    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let this = ManuallyDrop::new(self);
        (this.pointer.as_ptr(), this.length, this.capacity)
    }

    /// # Safety
    /// `pointer`, `length`, and `capacity` must originate from `Vector::into_raw_parts`
    /// for the same `T`, and ownership must not be retained elsewhere.
    pub unsafe fn from_raw_parts(pointer: *mut T, length: usize, capacity: usize) -> Self {
        Self {
            pointer: NonNull::new(pointer).unwrap_or_else(NonNull::dangling),
            length,
            capacity,
        }
    }

    fn grow(&mut self, capacity: usize) -> Result<(), TryReserveError> {
        if core::mem::size_of::<T>() == 0 {
            self.capacity = usize::MAX;
            return Ok(());
        }
        let new_layout =
            Layout::array::<T>(capacity).map_err(|_| TryReserveError::CapacityOverflow)?;
        let raw = if self.capacity == 0 {
            unsafe { allocator().alloc(new_layout) }
        } else {
            let old_layout =
                Layout::array::<T>(self.capacity).map_err(|_| TryReserveError::CapacityOverflow)?;
            unsafe {
                allocator().realloc(self.pointer.as_ptr().cast(), old_layout, new_layout.size())
            }
        };
        self.pointer = NonNull::new(raw.cast()).ok_or(TryReserveError::AllocationFailed)?;
        self.capacity = capacity;
        Ok(())
    }
}

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> Deref for Vector<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}
impl<T> DerefMut for Vector<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
}
impl<T: fmt::Debug> fmt::Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
impl<T: PartialEq> PartialEq for Vector<T> {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}
impl<T: Eq> Eq for Vector<T> {}

impl<T: Clone> Clone for Vector<T> {
    fn clone(&self) -> Self {
        let mut output =
            Self::with_capacity(self.length).unwrap_or_else(|_| panic!("MRML allocation failed"));
        output
            .try_extend_from_slice(self)
            .unwrap_or_else(|_| panic!("MRML allocation failed"));
        output
    }
}

impl<T> Drop for Vector<T> {
    fn drop(&mut self) {
        self.clear();
        if self.capacity != 0 && core::mem::size_of::<T>() != 0 {
            let layout = Layout::array::<T>(self.capacity).expect("valid vector layout");
            unsafe { allocator().dealloc(self.pointer.as_ptr().cast(), layout) };
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
    use std::string::String;

    #[test]
    fn grows_preserves_and_drops_values() {
        let mut values = Vector::with_capacity(1).unwrap();
        for value in 0..1000 {
            values.try_push(value).unwrap();
        }
        assert_eq!(values.len(), 1000);
        assert_eq!(values[999], 999);
        values.truncate(10);
        assert_eq!(values.pop(), Some(9));
    }

    #[test]
    fn owns_non_copy_values_and_clones() {
        let mut values = Vector::new();
        values.try_push(String::from("MRML")).unwrap();
        values.try_push(String::from("runtime")).unwrap();
        assert_eq!(&values.clone()[..], &["MRML", "runtime"]);
    }

    #[test]
    fn zero_sized_values_need_no_allocation() {
        let mut values = Vector::new();
        for _ in 0..100 {
            values.try_push(()).unwrap();
        }
        assert_eq!(values.len(), 100);
        assert_eq!(values.capacity(), usize::MAX);
    }
}
