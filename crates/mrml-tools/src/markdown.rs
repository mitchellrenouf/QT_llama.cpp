//! Low-latency terminal Markdown presentation.
use mrml_runtime::Text;
use mrml_runtime::mrml_print as print;

const RESET: &str = "\x1b[0m";

fn colors_enabled() -> bool {
    mrml_runtime::environment_variable("NO_COLOR").is_none()
        && mrml_runtime::environment_variable("CLICOLOR").is_none_or(|value| value != "0")
        && crate::platform::stdout_is_terminal()
}

fn push_styled(output: &mut Text, text: &str, code: &str, styled: bool) {
    if styled {
        output.push_str("\x1b[");
        output.push_str(code);
        output.push('m');
        output.push_str(text);
        output.push_str(RESET);
    } else {
        output.push_str(text);
    }
}

fn render_inline(text: &str, styled: bool, output: &mut Text) {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let (marker, code) = if bytes[index..].starts_with(b"**") {
            ("**", "1;97")
        } else if bytes[index] == b'`' {
            ("`", "36;100")
        } else if bytes[index] == b'*' {
            ("*", "35")
        } else {
            let character = text[index..].chars().next().unwrap();
            output.push(character);
            index += character.len_utf8();
            continue;
        };

        let content_start = index + marker.len();
        if let Some(relative_end) = text[content_start..].find(marker) {
            let content_end = content_start + relative_end;
            push_styled(output, &text[content_start..content_end], code, styled);
            index = content_end + marker.len();
        } else {
            output.push_str(marker);
            index = content_start;
        }
    }
}

fn render_markdown(md_text: &str, styled: bool) -> Text {
    let mut output = Text::with_capacity(md_text.len() + 64).expect("MRML allocation failed");
    let mut in_code_block = false;
    for line in md_text.split_inclusive('\n') {
        let (content, newline) = line
            .strip_suffix('\n')
            .map_or((line, ""), |line| (line, "\n"));
        if content.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            push_styled(&mut output, content, "97;100", styled);
            output.push_str(newline);
            continue;
        }

        let trimmed = content.trim_start();
        let leading = &content[..content.len() - trimmed.len()];
        let heading_level = trimmed.bytes().take_while(|byte| *byte == b'#').count();
        if (1..=6).contains(&heading_level) && trimmed.as_bytes().get(heading_level) == Some(&b' ')
        {
            let color = match heading_level {
                1 => "1;33",
                2 => "1;36",
                _ => "1;32",
            };
            output.push_str(leading);
            push_styled(&mut output, &trimmed[heading_level + 1..], color, styled);
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            output.push_str(leading);
            push_styled(&mut output, "•", "33", styled);
            output.push(' ');
            render_inline(item, styled, &mut output);
        } else {
            render_inline(content, styled, &mut output);
        }
        output.push_str(newline);
    }
    output
}

pub fn print_rich_markdown(md_text: &str) {
    print!("{}", render_markdown(md_text, colors_enabled()));
}

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
    fn renders_common_markdown_without_ansi_when_unstyled() {
        let rendered = render_markdown(
            "# Title\n- **bold** and `code`\n```rs\nlet x = 1;\n```\n",
            false,
        );
        assert_eq!(rendered, "Title\n• bold and code\nlet x = 1;\n");
    }

    #[test]
    fn styled_rendering_contains_ansi_and_preserves_text() {
        let rendered = render_markdown("## Heading\n*italic*", true);
        assert!(rendered.contains("\x1b[1;36mHeading\x1b[0m"));
        assert!(rendered.contains("\x1b[35mitalic\x1b[0m"));
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
        let s = "Badger: Барсук 🦡";
        for i in 0..=s.len() + 10 {
            let truncated = truncate_utf8(s, i);
            assert!(truncated.len() <= i);
            assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
        }
    }
}
