#![no_std]

#[cfg(any(feature = "std", test))]
extern crate std;

use core::error::Error as StdError;
use core::fmt::{self, Display, Write};
use mrml_runtime::Text;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    message: Text,
    source: Option<SourceError>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceError(Text);

impl Error {
    pub const fn empty() -> Self {
        Self {
            message: Text::new(),
            source: None,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.message
    }

    pub fn with_source(source: impl Display) -> Self {
        let source = format_text(format_args!("{source}"));
        Self {
            message: source.clone(),
            source: Some(SourceError(source)),
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}
impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_ref().map(|source| source as _)
    }
}
impl Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl StdError for SourceError {}

impl From<mrml_json::Error> for Error {
    fn from(error: mrml_json::Error) -> Self {
        Self::with_source(error)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::with_source(error)
    }
}

pub type Result<T, E = Error> = core::result::Result<T, E>;

fn format_text(arguments: fmt::Arguments<'_>) -> Text {
    let mut output = Text::new();
    output.write_fmt(arguments).expect("MRML allocation failed");
    output
}

pub fn message(text: impl Display) -> Error {
    Error {
        message: format_text(format_args!("{text}")),
        source: None,
    }
}

pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    Error {
        message: format_text(arguments),
        source: None,
    }
}

pub trait Context<T> {
    fn context(self, context: impl Display) -> Result<T>;
    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C;
}

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
    fn formats_messages_context_and_sources() {
        assert_eq!(anyhow!("bad {}", 7).to_string(), "bad 7");
        let error = Err::<(), _>(TestError)
            .context("operation failed")
            .unwrap_err();
        assert_eq!(error.to_string(), "operation failed: source failed");
        assert_eq!(
            None::<u8>.context("value missing").unwrap_err().to_string(),
            "value missing"
        );

        let sourced = Error::with_source(TestError);
        assert_eq!(sourced.source().unwrap().to_string(), "source failed");
    }

    #[cfg(feature = "std")]
    #[test]
    fn converts_standard_io_errors() {
        fn io_failure() -> Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"))?
        }
        assert_eq!(io_failure().unwrap_err().to_string(), "missing");
    }

    #[derive(Debug)]
    struct TestError;
    impl Display for TestError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("source failed")
        }
    }
    impl StdError for TestError {}
}
