#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AnsiCode {
    Bold = 1,
    Dimmed = 2,
    Italic = 3,
    Red = 31,
    Green = 32,
    Yellow = 33,
    Magenta = 35,
    Cyan = 36,
    BrightBlack = 90,
    BrightGreen = 92,
    BrightYellow = 93,
    BrightCyan = 96,
    BrightWhite = 97,
}

impl AnsiCode {
    pub const fn value(self) -> u8 {
        self as u8
    }
}

#[cfg(feature = "alloc")]
mod allocated {
    extern crate alloc;

    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::fmt::{self, Display};
    #[cfg(feature = "std")]
    use std::io::IsTerminal;

    #[derive(Clone, Debug)]
    pub struct StyledContent {
        text: String,
        codes: Vec<u8>,
    }

    impl StyledContent {
        fn with_code(mut self, code: u8) -> Self {
            if !self.codes.contains(&code) {
                self.codes.push(code);
            }
            self
        }
    }

    impl Display for StyledContent {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            if colors_enabled() && !self.codes.is_empty() {
                write!(formatter, "\x1b[")?;
                for (index, code) in self.codes.iter().enumerate() {
                    if index != 0 {
                        formatter.write_str(";")?;
                    }
                    write!(formatter, "{code}")?;
                }
                write!(formatter, "m{}\x1b[0m", self.text)
            } else {
                formatter.write_str(&self.text)
            }
        }
    }

    fn colors_enabled() -> bool {
        #[cfg(feature = "std")]
        {
            std::env::var_os("NO_COLOR").is_none()
                && std::env::var("CLICOLOR").map_or(true, |value| value != "0")
                && std::io::stdout().is_terminal()
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    pub trait Colorize: Display {
        fn style(&self, code: u8) -> StyledContent {
            StyledContent {
                text: self.to_string(),
                codes: vec![code],
            }
        }

        fn bold(&self) -> StyledContent {
            self.style(1)
        }
        fn dimmed(&self) -> StyledContent {
            self.style(2)
        }
        fn italic(&self) -> StyledContent {
            self.style(3)
        }
        fn red(&self) -> StyledContent {
            self.style(31)
        }
        fn green(&self) -> StyledContent {
            self.style(32)
        }
        fn yellow(&self) -> StyledContent {
            self.style(33)
        }
        fn magenta(&self) -> StyledContent {
            self.style(35)
        }
        fn cyan(&self) -> StyledContent {
            self.style(36)
        }
        fn bright_black(&self) -> StyledContent {
            self.style(90)
        }
        fn bright_green(&self) -> StyledContent {
            self.style(92)
        }
        fn bright_yellow(&self) -> StyledContent {
            self.style(93)
        }
        fn bright_cyan(&self) -> StyledContent {
            self.style(96)
        }
        fn bright_white(&self) -> StyledContent {
            self.style(97)
        }
    }

    impl<T: Display> Colorize for T {}

    macro_rules! styled_methods {
    ($(($name:ident, $code:expr)),* $(,)?) => {
        impl StyledContent {
            $(pub fn $name(self) -> Self { self.with_code($code) })*
        }
    };
}

    styled_methods!(
        (bold, 1),
        (dimmed, 2),
        (italic, 3),
        (red, 31),
        (green, 32),
        (yellow, 33),
        (magenta, 35),
        (cyan, 36),
        (bright_black, 90),
        (bright_green, 92),
        (bright_yellow, 93),
        (bright_cyan, 96),
        (bright_white, 97),
    );

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn styling_preserves_plain_text_when_colors_are_disabled() {
            // SAFETY: this crate has a single unit test, so no sibling test thread
            // can read or mutate NO_COLOR concurrently.
            unsafe { std::env::set_var("NO_COLOR", "1") };
            assert_eq!("hello".green().bold().to_string(), "hello");
            // SAFETY: paired with the serialized mutation above.
            unsafe { std::env::remove_var("NO_COLOR") };
        }
    }
}

#[cfg(feature = "alloc")]
pub use allocated::*;

#[cfg(test)]
mod portable_tests {
    #[test]
    fn ansi_codes_have_protocol_values_without_allocation() {
        assert_eq!(super::AnsiCode::Bold.value(), 1);
        assert_eq!(super::AnsiCode::BrightWhite.value(), 97);
    }
}
