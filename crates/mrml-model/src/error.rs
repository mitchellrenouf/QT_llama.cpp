use core::error::Error as StdError;
use core::fmt::{self, Write};
use mrml_runtime::Text;

#[derive(Debug)]
pub struct Error {
    message: Text,
    source: Option<SourceError>,
}

#[derive(Debug)]
struct SourceError(Text);

impl Error {
    pub fn message(text: impl fmt::Display) -> Self {
        Self {
            message: format_text(format_args!("{text}")),
            source: None,
        }
    }

    pub fn with_source(error: impl StdError + Send + Sync + 'static) -> Self {
        let message = format_text(format_args!("{error}"));
        Self {
            message: message.clone(),
            source: Some(SourceError(message)),
        }
    }
}

fn format_text(arguments: fmt::Arguments<'_>) -> Text {
    let mut output = Text::new();
    output.write_fmt(arguments).expect("MRML allocation failed");
    output
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_ref().map(|source| source as _)
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl StdError for SourceError {}

#[cfg(feature = "runtime")]
impl From<mrml_tensor::anyhow::Error> for Error {
    fn from(error: mrml_tensor::anyhow::Error) -> Self {
        Self::with_source(error)
    }
}

#[cfg(feature = "runtime")]
impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::with_source(error)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use core::error::Error as _;
    use std::string::ToString;

    #[test]
    fn preserves_message_text_and_sources_without_boxing() {
        let plain = super::Error::message("model failed");
        assert_eq!(plain.to_string(), "model failed");
        assert!(plain.source().is_none());

        let sourced = super::Error::with_source(TestError);
        assert_eq!(sourced.to_string(), "source failed");
        assert_eq!(sourced.source().unwrap().to_string(), "source failed");
    }

    #[derive(Debug)]
    struct TestError;
    impl core::fmt::Display for TestError {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("source failed")
        }
    }
    impl core::error::Error for TestError {}
}
