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
    pub fn msg(message: impl fmt::Display) -> Self {
        Self {
            message: format_text(format_args!("{message}")),
            source: None,
        }
    }

    fn with_source(error: impl fmt::Display) -> Self {
        let source = format_text(format_args!("{error}"));
        Self {
            message: source.clone(),
            source: Some(SourceError(source)),
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

impl From<core::num::TryFromIntError> for Error {
    fn from(error: core::num::TryFromIntError) -> Self {
        Self::with_source(error)
    }
}
impl From<mrml_runtime::FileError> for Error {
    fn from(error: mrml_runtime::FileError) -> Self {
        Self::with_source(error)
    }
}

pub type Result<T, E = Error> = core::result::Result<T, E>;

pub fn formatted(arguments: fmt::Arguments<'_>) -> Error {
    Error {
        message: format_text(arguments),
        source: None,
    }
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
    use mrml_runtime::mrml_format as format;

    #[test]
    fn preserves_formatted_messages_and_sources() {
        let formatted = anyhow!("CUDA error {}", 7);
        assert_eq!(format!("{formatted}"), "CUDA error 7");
    }

    #[test]
    fn preserves_platform_file_error_sources() {
        let file_error = Error::from(mrml_runtime::FileError::OpenFailed);
        assert_eq!(format!("{file_error}"), "failed to open file");
        assert!(core::error::Error::source(&file_error).is_some());
    }
}
