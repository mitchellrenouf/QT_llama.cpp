use core::cell::{RefCell, RefMut};

/// Mutable model state guarded by `MrmlEngine`'s outer model mutex.
///
/// This keeps runtime borrow checking for accidental overlapping access without
/// taking another atomic or operating-system lock for every activation buffer.
pub struct Mutex<T>(RefCell<T>);

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self(RefCell::new(value))
    }

    #[inline]
    pub fn lock(&self) -> RefMut<'_, T> {
        self.0.borrow_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::Mutex;

    #[test]
    fn permits_sequential_model_state_updates() {
        let value = Mutex::new(1_u32);
        *value.lock() += 2;
        *value.lock() *= 3;
        assert_eq!(*value.lock(), 9);
    }

    #[test]
    #[should_panic(expected = "already borrowed")]
    fn rejects_overlapping_mutable_access() {
        let value = Mutex::new(0_u32);
        let _first = value.lock();
        let _second = value.lock();
    }
}
