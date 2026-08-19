use crate::{TryReserveError, Vector};
use core::fmt::{self, Write};
use core::ops::Deref;

#[derive(Clone, Default)]
pub struct Text {
    bytes: Vector<u8>,
    inline: [u8; Self::INLINE_CAPACITY],
    inline_len: u8,
    heap_backed: bool,
}

impl Text {
    pub fn remove(&mut self, index: usize) -> char {
        let character = self.as_str()[index..]
            .chars()
            .next()
            .expect("cannot remove a character at the end of text");
        let width = character.len_utf8();
        if self.heap_backed {
            for _ in 0..width {
                self.bytes.remove(index);
            }
        } else {
            let len = self.inline_len as usize;
            self.inline.copy_within(index + width..len, index);
            self.inline_len -= width as u8;
        }
        character
    }

    const INLINE_CAPACITY: usize = 23;

    pub const fn new() -> Self {
        Self {
            bytes: Vector::new(),
            inline: [0; Self::INLINE_CAPACITY],
            inline_len: 0,
            heap_backed: false,
        }
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        if capacity <= Self::INLINE_CAPACITY {
            Ok(Self::new())
        } else {
            Ok(Self {
                bytes: Vector::with_capacity(capacity)?,
                inline: [0; Self::INLINE_CAPACITY],
                inline_len: 0,
                heap_backed: true,
            })
        }
    }

    pub fn try_from_str(value: &str) -> Result<Self, TryReserveError> {
        let mut text = Self::with_capacity(value.len())?;
        text.try_push_str(value)?;
        Ok(text)
    }

    pub fn try_from_utf8(bytes: Vector<u8>) -> Result<Self, Vector<u8>> {
        if core::str::from_utf8(&bytes).is_err() {
            return Err(bytes);
        }
        if bytes.len() <= Self::INLINE_CAPACITY {
            return Ok(
                Self::try_from_str(unsafe { core::str::from_utf8_unchecked(&bytes) })
                    .expect("inline text cannot fail to allocate"),
            );
        }
        Ok(Self {
            bytes,
            inline: [0; Self::INLINE_CAPACITY],
            inline_len: 0,
            heap_backed: true,
        })
    }

    pub fn try_push_str(&mut self, value: &str) -> Result<(), TryReserveError> {
        if !self.heap_backed {
            let current = self.inline_len as usize;
            let needed = current.saturating_add(value.len());
            if needed <= Self::INLINE_CAPACITY {
                self.inline[current..needed].copy_from_slice(value.as_bytes());
                self.inline_len = needed as u8;
                return Ok(());
            }
            let mut bytes = Vector::with_capacity(needed)?;
            bytes.try_extend_from_slice(&self.inline[..current])?;
            self.bytes = bytes;
            self.heap_backed = true;
        }
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
        if self.heap_backed {
            self.bytes.clear();
        } else {
            self.inline_len = 0;
        }
    }
    pub fn len(&self) -> usize {
        if self.heap_backed {
            self.bytes.len()
        } else {
            self.inline_len as usize
        }
    }
    pub fn capacity(&self) -> usize {
        if self.heap_backed {
            self.bytes.capacity()
        } else {
            Self::INLINE_CAPACITY
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_str(&self) -> &str {
        // Construction only accepts UTF-8 strings and encoded `char` values.
        let bytes = if self.heap_backed {
            &self.bytes[..]
        } else {
            &self.inline[..self.inline_len as usize]
        };
        unsafe { core::str::from_utf8_unchecked(bytes) }
    }

    pub fn replace(&self, needle: &str, replacement: &str) -> Self {
        if needle.is_empty() {
            return self.clone();
        }
        let mut output = Self::with_capacity(self.len()).expect("MRML allocation failed");
        let mut remainder = self.as_str();
        while let Some(index) = remainder.find(needle) {
            output.push_str(&remainder[..index]);
            output.push_str(replacement);
            remainder = &remainder[index + needle.len()..];
        }
        output.push_str(remainder);
        output
    }
}

impl Deref for Text {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl PartialEq for Text {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for Text {}
impl PartialOrd for Text {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Text {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_str().cmp(other.as_str())
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

    #[test]
    fn replaces_substrings_without_rust_alloc() {
        assert_eq!(Text::from("a<>b<>c").replace("<>", "-"), "a-b-c");
    }

    #[test]
    fn stores_short_text_inline_and_promotes_without_data_loss() {
        let mut text = Text::from("short token");
        assert!(!text.heap_backed);
        assert!(!text.is_empty());
        assert_eq!(text.capacity(), Text::INLINE_CAPACITY);

        text.push_str(" that exceeds inline storage");
        assert!(text.heap_backed);
        assert_eq!(text, "short token that exceeds inline storage");

        text.clear();
        text.push_str("reused");
        assert_eq!(text, "reused");
    }

    #[test]
    fn lexical_order_is_independent_of_storage() {
        let inline = Text::from("model.layers");
        let mut promoted = Text::with_capacity(64).unwrap();
        promoted.push_str("model.layers");
        assert_eq!(inline, promoted);
        assert_eq!(inline.cmp(&promoted), core::cmp::Ordering::Equal);
        assert!(Text::from("a") < Text::from("z"));
    }
}
