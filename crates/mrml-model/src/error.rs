use std::fmt;

#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl Error {
    pub fn message(text: impl Into<String>) -> Self {
        Self {
            message: text.into(),
            source: None,
        }
    }

    pub fn with_source(error: impl std::error::Error + Send + Sync + 'static) -> Self {
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

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

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

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    #[test]
    fn preserves_message_text() {
        assert_eq!(
            super::Error::message("model failed").to_string(),
            "model failed"
        );
    }
}
