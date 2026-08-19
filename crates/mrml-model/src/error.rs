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
    pub fn message(text: impl Into<String>) -> Self {
        Self {
            message: text.into(),
            source: None,
        }
    }

    pub fn with_source(error: impl StdError + Send + Sync + 'static) -> Self {
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
impl From<mrml_tensor::anyhow::Error> for Error {
    fn from(error: mrml_tensor::anyhow::Error) -> Self {
        Self::with_source(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::with_source(error)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    #[test]
    fn preserves_message_text() {
        assert_eq!(
            super::Error::message("model failed").to_string(),
            "model failed"
        );
    }
}
