use termimad::crossterm::style::Color::*;
use termimad::crossterm::style::Attribute;
use termimad::{MadSkin, StyledChar};

pub fn build_custom_skin() -> MadSkin {
    let mut skin = MadSkin::default();

    // Headers
    skin.headers[0].set_fg(Yellow);
    skin.headers[0].add_attr(Attribute::Bold);
    skin.headers[1].set_fg(Cyan);
    skin.headers[1].add_attr(Attribute::Bold);
    skin.headers[2].set_fg(Green);
    skin.headers[2].add_attr(Attribute::Bold);

    // Text formatting
    skin.bold.set_fg(White);
    skin.bold.add_attr(Attribute::Bold);
    skin.italic.set_fg(Magenta);

    // Code blocks & inline code
    skin.inline_code.set_fg(Cyan);
    skin.inline_code.set_bg(DarkGrey);

    skin.code_block.set_fg(White);
    skin.code_block.set_bg(DarkGrey);

    // Bullet list icons
    skin.bullet = StyledChar::from_fg_char(Yellow, '•');

    skin
}

pub fn print_rich_markdown(md_text: &str) {
    let skin = build_custom_skin();
    skin.print_text(md_text);
}

/// Safely truncate a UTF-8 string at or before `max_bytes` without splitting multi-byte character boundaries.
pub fn truncate_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skin_creation() {
        let skin = build_custom_skin();
        assert!(skin.headers[0].compound_style.has_attr(termimad::crossterm::style::Attribute::Bold));
    }

    #[test]
    fn test_truncate_utf8_ascii() {
        let s = "Hello, world!";
        assert_eq!(truncate_utf8(s, 5), "Hello");
        assert_eq!(truncate_utf8(s, 100), "Hello, world!");
        assert_eq!(truncate_utf8(s, 0), "");
    }

    #[test]
    fn test_truncate_utf8_multibyte() {
        // 'л' is 2 bytes: bytes 0..2
        let cyrillic = "л";
        assert_eq!(truncate_utf8(cyrillic, 1), "");
        assert_eq!(truncate_utf8(cyrillic, 2), "л");

        // String with mixed multi-byte characters
        let s = "Badger: Барсук 🦡";
        // Verify no panic for every possible slice index from 0 to s.len() + 10
        for i in 0..=s.len() + 10 {
            let truncated = truncate_utf8(s, i);
            assert!(truncated.len() <= i);
            assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
        }
    }
}

