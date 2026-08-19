use crate::{TryReserveError, Vector};
use core::fmt::{self, Write};
use core::ops::Deref;

#[derive(Clone, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Text {
    bytes: Vector<u8>,
}

impl Text {
    pub const fn new() -> Self {
        Self {
            bytes: Vector::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Ok(Self {
            bytes: Vector::with_capacity(capacity)?,
        })
    }

    pub fn try_from_str(value: &str) -> Result<Self, TryReserveError> {
        let mut text = Self::with_capacity(value.len())?;
        text.try_push_str(value)?;
        Ok(text)
    }

    pub fn try_from_utf8(bytes: Vector<u8>) -> Result<Self, Vector<u8>> {
        if core::str::from_utf8(&bytes).is_ok() {
            Ok(Self { bytes })
        } else {
            Err(bytes)
        }
    }

    pub fn try_push_str(&mut self, value: &str) -> Result<(), TryReserveError> {
        self.bytes.try_extend_from_slice(value.as_bytes())
    }
    pub fn push_str(&mut self, value: &str) {
        self.try_push_str(value).expect("MRML allocation failed");
    }

    pub fn try_push(&mut self, value: char) -> Result<(), TryReserveError> {
        let mut encoded = [0; 4];
        self.try_push_str(value.encode_utf8(&mut encoded))
    }
    pub fn push(&mut self, value: char) {
        self.try_push(value).expect("MRML allocation failed");
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
    }
    pub fn len(&self) -> usize {
        self.bytes.len()
    }
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
    pub fn as_str(&self) -> &str {
        // Construction only accepts UTF-8 strings and encoded `char` values.
        unsafe { core::str::from_utf8_unchecked(&self.bytes) }
    }
}

impl Deref for Text {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl core::borrow::Borrow<str> for Text {
    fn borrow(&self) -> &str {
        self
    }
}
impl AsRef<str> for Text {
    fn as_ref(&self) -> &str {
        self
    }
}
impl AsRef<[u8]> for Text {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}
impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}
impl Write for Text {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.try_push_str(value).map_err(|_| fmt::Error)
    }
}
impl PartialEq<str> for Text {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for Text {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl From<&str> for Text {
    fn from(value: &str) -> Self {
        Self::try_from_str(value).expect("MRML allocation failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_and_grows_without_rust_alloc() {
        let mut text = Text::try_from_str("MRML").unwrap();
        write!(text, " {} {}", "runtime", 4).unwrap();
        text.try_push('✓').unwrap();
        assert_eq!(text, "MRML runtime 4✓");
        assert!(core::str::from_utf8(&text.bytes).is_ok());
    }
}
