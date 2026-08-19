use alloc::boxed::Box;
use alloc::string::{String, ToString};
use core::error::Error as StdError;
use core::fmt;

#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}
impl Error {
    pub fn msg(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }
    fn with_source(error: impl StdError + Send + Sync + 'static) -> Self {
        Self {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}
impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}
#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::with_source(error)
    }
}
impl From<alloc::string::FromUtf8Error> for Error {
    fn from(error: alloc::string::FromUtf8Error) -> Self {
        Self::with_source(error)
    }
}
impl From<core::num::TryFromIntError> for Error {
    fn from(error: core::num::TryFromIntError) -> Self {
        Self::with_source(error)
    }
}
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    Error::msg(alloc::fmt::format(arguments))
}

#[macro_export]
macro_rules! tensor_anyhow { ($($argument:tt)*) => { $crate::anyhow::formatted(format_args!($($argument)*)) }; }
#[macro_export]
macro_rules! tensor_bail { ($($argument:tt)*) => { return Err($crate::tensor_anyhow!($($argument)*)) }; }
pub use crate::tensor_anyhow as anyhow;
pub use crate::tensor_bail as bail;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_formatted_messages_and_sources() {
        let formatted = anyhow!("CUDA error {}", 7);
        assert_eq!(formatted.to_string(), "CUDA error 7");
    }

    #[cfg(feature = "std")]
    #[test]
    fn preserves_standard_error_sources() {
        let io_error = Error::from(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
        assert_eq!(io_error.to_string(), "missing");
        assert!(std::error::Error::source(&io_error).is_some());
    }
}
