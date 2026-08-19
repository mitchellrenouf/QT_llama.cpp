use alloc::format;
use alloc::string::String;
use core::fmt;
use core::str::FromStr;

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
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "general" => Ok(Self::General),
            "coder" => Ok(Self::Coder),
            "automatic" => Ok(Self::Automatic),
            _ => Err(format!("invalid agent mode '{value}'")),
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
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "rocm" => Ok(Self::Rocm),
            "vulkan" => Ok(Self::Vulkan),
            "sycl" => Ok(Self::Sycl),
            "cpu" => Ok(Self::Cpu),
            _ => Err(format!("invalid backend '{value}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn portable_modes_parse_and_display() {
        assert_eq!("automatic".parse(), Ok(AgentMode::Automatic));
        assert_eq!(AgentMode::Coder.to_string(), "coder");
        assert_eq!("cuda".parse(), Ok(BackendChoice::Cuda));
        assert_eq!(BackendChoice::Cpu.to_string(), "cpu");
        assert!("invalid".parse::<BackendChoice>().is_err());
    }
}
