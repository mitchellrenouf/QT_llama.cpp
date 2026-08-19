#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use core::error::Error as StdError;
use core::fmt::{self, Display};

pub type Error = Box<dyn StdError + Send + Sync + 'static>;
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(Debug)]
pub struct MessageError(String);

impl MessageError {
    pub fn new(message: String) -> Self {
        Self(message)
    }
}

impl Display for MessageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl StdError for MessageError {}

pub fn message(text: impl Into<String>) -> Error {
    Box::new(MessageError::new(text.into()))
}

pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    message(alloc::fmt::format(arguments))
}

#[derive(Debug)]
struct ContextError {
    context: String,
    source: Error,
}

impl Display for ContextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.context, self.source)
    }
}

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

impl<T, E> Context<T> for core::result::Result<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn context(self, context: impl Display) -> Result<T> {
        self.map_err(|source| {
            Box::new(ContextError {
                context: context.to_string(),
                source: Box::new(source),
            }) as Error
        })
    }

    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C,
    {
        self.map_err(|source| {
            Box::new(ContextError {
                context: context().to_string(),
                source: Box::new(source),
            }) as Error
        })
    }
}

impl<T> Context<T> for Option<T> {
    fn context(self, context: impl Display) -> Result<T> {
        self.ok_or_else(|| message(context.to_string()))
    }

    fn with_context<C, F>(self, context: F) -> Result<T>
    where
        C: Display,
        F: FnOnce() -> C,
    {
        self.ok_or_else(|| message(context().to_string()))
    }
}

#[macro_export]
macro_rules! anyhow {
    ($message:literal $(,)?) => {
        $crate::message($message)
    };
    ($format:expr, $($argument:tt)*) => {
        $crate::formatted(format_args!($format, $($argument)*))
    };
    ($error:expr $(,)?) => {
        $crate::message($error.to_string())
    };
}

#[macro_export]
macro_rules! bail {
    ($($argument:tt)*) => {
        return Err($crate::anyhow!($($argument)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io_failure() -> Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"))?
    }

    #[test]
    fn formats_messages_and_converts_standard_errors() {
        assert_eq!(anyhow!("bad {}", 7).to_string(), "bad 7");
        assert_eq!(io_failure().unwrap_err().to_string(), "missing");
    }

    #[test]
    fn adds_context_to_results_and_options() {
        let error = std::fs::read("definitely-missing").context("read failed").unwrap_err();
        assert!(error.to_string().starts_with("read failed:"));
        assert_eq!(None::<u8>.context("value missing").unwrap_err().to_string(), "value missing");
    }
}
