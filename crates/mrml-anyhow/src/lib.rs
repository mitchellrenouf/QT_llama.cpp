#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(test)]
extern crate std;

use core::error::Error as StdError;
#[cfg(not(feature = "alloc"))]
use core::fmt::Write;
use core::fmt::{self, Display};

#[cfg(feature = "alloc")]
pub type Error = alloc::boxed::Box<dyn StdError + Send + Sync + 'static>;

#[cfg(not(feature = "alloc"))]
#[derive(Clone, Eq, PartialEq)]
pub struct Error {
    bytes: [u8; Self::CAPACITY],
    len: u16,
}

#[cfg(not(feature = "alloc"))]
impl Error {
    pub const CAPACITY: usize = 256;

    pub const fn empty() -> Self {
        Self {
            bytes: [0; Self::CAPACITY],
            len: 0,
        }
    }

    pub fn as_str(&self) -> &str {
        // `write_str` only copies bytes from valid UTF-8 strings.
        unsafe { core::str::from_utf8_unchecked(&self.bytes[..self.len as usize]) }
    }
}

#[cfg(not(feature = "alloc"))]
impl Write for Error {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let available = Self::CAPACITY - self.len as usize;
        let mut count = value.len().min(available);
        while !value.is_char_boundary(count) {
            count -= 1;
        }
        let start = self.len as usize;
        self.bytes[start..start + count].copy_from_slice(&value.as_bytes()[..count]);
        self.len += count as u16;
        Ok(())
    }
}

#[cfg(not(feature = "alloc"))]
impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Error")
            .field(&self.as_str())
            .finish()
    }
}

#[cfg(not(feature = "alloc"))]
impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(not(feature = "alloc"))]
impl StdError for Error {}

pub type Result<T, E = Error> = core::result::Result<T, E>;

#[cfg(feature = "alloc")]
#[derive(Debug)]
struct MessageError(alloc::string::String);

#[cfg(feature = "alloc")]
impl Display for MessageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[cfg(feature = "alloc")]
impl StdError for MessageError {}

#[cfg(feature = "alloc")]
pub fn message(text: impl Into<alloc::string::String>) -> Error {
    alloc::boxed::Box::new(MessageError(text.into()))
}

#[cfg(not(feature = "alloc"))]
pub fn message(text: impl Display) -> Error {
    let mut error = Error::empty();
    let _ = write!(error, "{text}");
    error
}

#[cfg(feature = "alloc")]
pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    alloc::boxed::Box::new(MessageError(alloc::fmt::format(arguments)))
}

#[cfg(not(feature = "alloc"))]
pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    message(arguments)
}

#[cfg(feature = "alloc")]
#[derive(Debug)]
struct ContextError {
    context: alloc::string::String,
    source: Error,
}

#[cfg(feature = "alloc")]
impl Display for ContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.source)
    }
}

#[cfg(feature = "alloc")]
impl StdError for ContextError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}

pub trait Context<T> {
    fn context(self, context: impl Display) -> Result<T>;
    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C;
}

#[cfg(feature = "alloc")]
impl<T, E> Context<T> for core::result::Result<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn context(self, context: impl Display) -> Result<T> {
        self.map_err(|source| {
            alloc::boxed::Box::new(ContextError {
                context: alloc::format!("{context}"),
                source: alloc::boxed::Box::new(source),
            }) as Error
        })
    }

    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C,
    {
        self.map_err(|source| {
            alloc::boxed::Box::new(ContextError {
                context: alloc::format!("{}", context()),
                source: alloc::boxed::Box::new(source),
            }) as Error
        })
    }
}

#[cfg(not(feature = "alloc"))]
impl<T, E: Display> Context<T> for core::result::Result<T, E> {
    fn context(self, context: impl Display) -> Result<T> {
        self.map_err(|source| formatted(format_args!("{context}: {source}")))
    }

    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C,
    {
        self.map_err(|source| formatted(format_args!("{}: {source}", context())))
    }
}

impl<T> Context<T> for Option<T> {
    fn context(self, context: impl Display) -> Result<T> {
        self.ok_or_else(|| formatted(format_args!("{context}")))
    }

    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C,
    {
        self.ok_or_else(|| formatted(format_args!("{}", context())))
    }
}

#[macro_export]
macro_rules! anyhow {
    ($message:literal $(,)?) => { $crate::message($message) };
    ($format:expr, $($argument:tt)*) => { $crate::formatted(format_args!($format, $($argument)*)) };
    ($error:expr $(,)?) => { $crate::formatted(format_args!("{}", $error)) };
}

#[macro_export]
macro_rules! bail {
    ($($argument:tt)*) => { return Err($crate::anyhow!($($argument)*)) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::ToString;

    #[test]
    fn formats_messages_and_bounded_context() {
        assert_eq!(anyhow!("bad {}", 7).to_string(), "bad 7");
        let error = Err::<(), _>(TestError)
            .context("operation failed")
            .unwrap_err();
        assert_eq!(error.to_string(), "operation failed: source failed");
        assert_eq!(
            None::<u8>.context("value missing").unwrap_err().to_string(),
            "value missing"
        );
    }

    #[cfg(not(feature = "alloc"))]
    #[test]
    fn fixed_error_truncates_only_at_utf8_boundaries() {
        let error = message("é".repeat(200));
        assert!(error.as_str().is_char_boundary(error.as_str().len()));
        assert!(error.as_str().len() <= Error::CAPACITY);
    }

    #[derive(Debug)]
    struct TestError;

    impl Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("source failed")
        }
    }

    impl StdError for TestError {}

    #[cfg(feature = "alloc")]
    #[test]
    fn boxed_errors_preserve_standard_sources() {
        fn io_failure() -> Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"))?
        }
        assert_eq!(io_failure().unwrap_err().to_string(), "missing");
        let error = std::fs::read("definitely-missing")
            .context("read failed")
            .unwrap_err();
        assert!(error.to_string().starts_with("read failed:"));
        assert!(StdError::source(error.as_ref()).is_some());
    }
}
