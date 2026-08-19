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

/// Allocation-free ANSI styling over a borrowed display value.
///
/// The caller decides whether ANSI output is enabled because terminal and
/// environment detection belong to a platform or application crate.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedStyle<'a, T: ?Sized> {
    value: &'a T,
    codes: [AnsiCode; 8],
    code_count: u8,
    ansi: bool,
}

impl<'a, T: core::fmt::Display + ?Sized> BorrowedStyle<'a, T> {
    pub const fn new(value: &'a T, ansi: bool) -> Self {
        Self {
            value,
            codes: [AnsiCode::Bold; 8],
            code_count: 0,
            ansi,
        }
    }

    pub const fn with(mut self, code: AnsiCode) -> Self {
        let mut index = 0;
        while index < self.code_count as usize {
            if self.codes[index] as u8 == code as u8 {
                return self;
            }
            index += 1;
        }
        if (self.code_count as usize) < self.codes.len() {
            self.codes[self.code_count as usize] = code;
            self.code_count += 1;
        }
        self
    }

    pub const fn ansi_enabled(mut self, enabled: bool) -> Self {
        self.ansi = enabled;
        self
    }
}

impl<T: core::fmt::Display + ?Sized> core::fmt::Display for BorrowedStyle<'_, T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if !self.ansi || self.code_count == 0 {
            return self.value.fmt(formatter);
        }
        formatter.write_str("\x1b[")?;
        for index in 0..self.code_count as usize {
            if index != 0 {
                formatter.write_str(";")?;
            }
            write!(formatter, "{}", self.codes[index].value())?;
        }
        write!(formatter, "m{}\x1b[0m", self.value)
    }
}

pub const fn style<T: core::fmt::Display + ?Sized>(
    value: &T,
    code: AnsiCode,
    ansi: bool,
) -> BorrowedStyle<'_, T> {
    BorrowedStyle::new(value, ansi).with(code)
}

mod allocated {
    use core::fmt::{self, Display, Write};
    use mrml_runtime::{Text, Vector};
    #[cfg(feature = "std")]
    use std::io::IsTerminal;

    #[derive(Clone, Debug)]
    pub struct StyledContent {
        text: Text,
        codes: Vector<u8>,
    }

    impl StyledContent {
        fn with_code(mut self, code: u8) -> Self {
            if !self.codes.contains(&code) {
                self.codes.try_push(code).expect("MRML allocation failed");
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
            let mut text = Text::new();
            write!(text, "{self}").expect("MRML allocation failed");
            let mut codes = Vector::new();
            codes.try_push(code).expect("MRML allocation failed");
            StyledContent { text, codes }
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
            #[cfg(feature = "std")]
            // SAFETY: this crate has a single unit test, so no sibling test thread
            // can read or mutate NO_COLOR concurrently.
            unsafe {
                std::env::set_var("NO_COLOR", "1")
            };
            let mut output = Text::new();
            write!(output, "{}", "hello".green().bold()).unwrap();
            assert_eq!(output, "hello");
            #[cfg(feature = "std")]
            // SAFETY: paired with the serialized mutation above.
            unsafe {
                std::env::remove_var("NO_COLOR")
            };
        }
    }
}

pub use allocated::*;

#[cfg(test)]
mod portable_tests {
    use core::fmt::Write;

    struct Buffer {
        bytes: [u8; 64],
        len: usize,
    }

    impl Write for Buffer {
        fn write_str(&mut self, value: &str) -> core::fmt::Result {
            let end = self.len + value.len();
            if end > self.bytes.len() {
                return Err(core::fmt::Error);
            }
            self.bytes[self.len..end].copy_from_slice(value.as_bytes());
            self.len = end;
            Ok(())
        }
    }

    #[test]
    fn ansi_codes_have_protocol_values_without_allocation() {
        assert_eq!(super::AnsiCode::Bold.value(), 1);
        assert_eq!(super::AnsiCode::BrightWhite.value(), 97);
    }

    #[test]
    fn borrowed_styles_format_into_caller_storage() {
        let mut output = Buffer {
            bytes: [0; 64],
            len: 0,
        };
        write!(
            output,
            "{}",
            super::style(&"hello", super::AnsiCode::Green, true)
                .with(super::AnsiCode::Bold)
                .with(super::AnsiCode::Green)
        )
        .unwrap();
        assert_eq!(&output.bytes[..output.len], b"\x1b[32;1mhello\x1b[0m");
    }
}
