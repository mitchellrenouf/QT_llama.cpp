use std::error::Error as StdError;
use std::fmt::{self, Display};

pub type Error = Box<dyn StdError + Send + Sync + 'static>;
pub type Result<T, E = Error> = std::result::Result<T, E>;

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

#[macro_export]
macro_rules! anyhow {
    ($message:literal $(,)?) => {
        $crate::message($message)
    };
    ($format:expr, $($argument:tt)*) => {
        $crate::message(format!($format, $($argument)*))
    };
    ($error:expr $(,)?) => {
        $crate::message($error.to_string())
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
}
