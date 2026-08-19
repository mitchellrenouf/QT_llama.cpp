use core::fmt;
use core::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseChoiceError(&'static str);

impl fmt::Display for ParseChoiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl core::error::Error for ParseChoiceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    General,
    Coder,
    Automatic,
}

impl fmt::Display for AgentMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::General => "general",
            Self::Coder => "coder",
            Self::Automatic => "automatic",
        })
    }
}

impl FromStr for AgentMode {
    type Err = ParseChoiceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("general") {
            Ok(Self::General)
        } else if value.eq_ignore_ascii_case("coder") {
            Ok(Self::Coder)
        } else if value.eq_ignore_ascii_case("automatic") {
            Ok(Self::Automatic)
        } else {
            Err(ParseChoiceError("invalid agent mode"))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Cuda,
    Rocm,
    Vulkan,
    Sycl,
    Cpu,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Vulkan => "vulkan",
            Self::Sycl => "sycl",
            Self::Cpu => "cpu",
        })
    }
}

impl FromStr for BackendChoice {
    type Err = ParseChoiceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case("cuda") {
            Ok(Self::Cuda)
        } else if value.eq_ignore_ascii_case("rocm") {
            Ok(Self::Rocm)
        } else if value.eq_ignore_ascii_case("vulkan") {
            Ok(Self::Vulkan)
        } else if value.eq_ignore_ascii_case("sycl") {
            Ok(Self::Sycl)
        } else if value.eq_ignore_ascii_case("cpu") {
            Ok(Self::Cpu)
        } else {
            Err(ParseChoiceError("invalid backend"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portable_modes_parse_and_display() {
        assert_eq!("AuToMaTiC".parse(), Ok(AgentMode::Automatic));
        assert_eq!("CUDA".parse(), Ok(BackendChoice::Cuda));
        assert_eq!(
            "invalid".parse::<BackendChoice>(),
            Err(ParseChoiceError("invalid backend")),
        );
    }
}
