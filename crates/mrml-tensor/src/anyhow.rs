use std::fmt;

#[derive(Debug)]
pub struct Error { message: String, source: Option<Box<dyn std::error::Error + Send + Sync>> }
impl Error {
    pub fn msg(message: impl Into<String>) -> Self { Self { message: message.into(), source: None } }
    fn with_source(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self { message: error.to_string(), source: Some(Box::new(error)) }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(&self.message) }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}
impl From<std::io::Error> for Error { fn from(error: std::io::Error) -> Self { Self::with_source(error) } }
impl From<std::string::FromUtf8Error> for Error { fn from(error: std::string::FromUtf8Error) -> Self { Self::with_source(error) } }
impl From<std::num::TryFromIntError> for Error { fn from(error: std::num::TryFromIntError) -> Self { Self::with_source(error) } }
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[macro_export]
macro_rules! tensor_anyhow { ($($argument:tt)*) => { $crate::anyhow::Error::msg(format!($($argument)*)) }; }
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

        let io_error = Error::from(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
        assert_eq!(io_error.to_string(), "missing");
        assert!(std::error::Error::source(&io_error).is_some());
    }
}
