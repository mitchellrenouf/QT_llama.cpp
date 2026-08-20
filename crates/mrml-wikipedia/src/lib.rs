#![no_std]

use mrml_runtime::Text;
use mrml_zim::{DirectoryEntry, EntryLocation};

/// Returns true for content entries that can contribute natural-language text.
pub fn is_article(entry: &DirectoryEntry) -> bool {
    matches!(entry.location, EntryLocation::Blob { .. })
        && matches!(entry.namespace, b'A' | b'C')
        && !entry.path.starts_with("Special:")
        && !entry.path.starts_with("Category:")
        && !entry.path.starts_with("File:")
        && !entry.path.starts_with("Template:")
        && !entry.path.starts_with("User:")
}

/// Converts a Wikipedia HTML document into normalized training text.
///
/// Script, style, template, table, navigation, and citation markup is omitted.
/// Block elements become paragraph boundaries and entities are decoded without
/// allocating intermediary strings.
pub fn html_to_text(html: &str) -> Text {
    let bytes = html.as_bytes();
    let mut output = Text::with_capacity(html.len().min(64 * 1024)).expect("MRML allocation failed");
    let mut index = 0;
    let mut suppressed_depth = 0usize;
    let mut pending_space = false;
    let mut pending_paragraph = false;

    while index < bytes.len() {
        if bytes[index] == b'<' {
            let Some(relative_end) = html[index + 1..].find('>') else { break };
            let end = index + 1 + relative_end;
            let raw = html[index + 1..end].trim();
            let closing = raw.starts_with('/');
            let name_start = usize::from(closing);
            let name_end = raw[name_start..]
                .find(|character: char| character.is_ascii_whitespace() || character == '/')
                .map(|offset| name_start + offset)
                .unwrap_or(raw.len());
            let name = &raw[name_start..name_end];
            let suppressed = matches_ignore_ascii_case(name, &["script", "style", "table", "nav", "footer", "header", "aside", "math"]);
            if suppressed {
                if closing { suppressed_depth = suppressed_depth.saturating_sub(1); }
                else if !raw.ends_with('/') { suppressed_depth += 1; }
            } else if suppressed_depth == 0 {
                if matches_ignore_ascii_case(name, &["p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "hr", "blockquote", "pre"]) {
                    pending_paragraph = true;
                } else { pending_space = true; }
            }
            index = end + 1;
            continue;
        }
        if suppressed_depth != 0 { index += 1; continue; }
        if bytes[index] == b'&' {
            if let Some(relative_end) = html[index..].find(';').filter(|end| *end <= 12) {
                let entity = &html[index + 1..index + relative_end];
                if let Some(character) = decode_entity(entity) {
                    push_normalized(&mut output, character, &mut pending_space, &mut pending_paragraph);
                    index += relative_end + 1;
                    continue;
                }
            }
        }
        let character = html[index..].chars().next().unwrap();
        push_normalized(&mut output, character, &mut pending_space, &mut pending_paragraph);
        index += character.len_utf8();
    }
    while output.ends_with(' ') || output.ends_with('\n') { output.pop(); }
    output
}

fn push_normalized(output: &mut Text, character: char, pending_space: &mut bool, pending_paragraph: &mut bool) {
    if character.is_whitespace() { *pending_space = true; return; }
    if *pending_paragraph && !output.is_empty() {
        while output.ends_with(' ') { output.pop(); }
        if !output.ends_with("\n\n") { output.push_str(if output.ends_with('\n') { "\n" } else { "\n\n" }); }
    } else if *pending_space && !output.is_empty() && !output.ends_with('\n') && !output.ends_with(' ') {
        output.push(' ');
    }
    *pending_paragraph = false;
    *pending_space = false;
    output.push(character);
}

fn matches_ignore_ascii_case(value: &str, choices: &[&str]) -> bool {
    choices.iter().any(|choice| value.eq_ignore_ascii_case(choice))
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'), "lt" => Some('<'), "gt" => Some('>'), "quot" => Some('"'),
        "apos" | "#39" => Some('\''), "nbsp" => Some(' '), "ndash" => Some('–'),
        "mdash" => Some('—'), "hellip" => Some('…'),
        value if value.starts_with("#x") || value.starts_with("#X") => u32::from_str_radix(&value[2..], 16).ok().and_then(char::from_u32),
        value if value.starts_with('#') => value[1..].parse::<u32>().ok().and_then(char::from_u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_readable_wikipedia_text() {
        let html = "<html><style>bad{}</style><h1>Rust &amp; Safety</h1><p>Rust is a <b>systems</b> language.</p><table><tr><td>noise</td></tr></table><p>It prevents&nbsp;many bugs.</p><script>noise()</script></html>";
        assert_eq!(html_to_text(html), "Rust & Safety\n\nRust is a systems language.\n\nIt prevents many bugs.");
    }

    #[test]
    fn decodes_numeric_entities_and_normalizes_spacing() {
        assert_eq!(html_to_text("<p>A  &#x2014;  B &#65;</p>"), "A — B A");
    }

    #[test]
    fn filters_non_article_namespaces_and_paths() {
        let article = DirectoryEntry { mime_type: 0, namespace: b'C', revision: 0, path: "Rust".into(), title: "Rust".into(), location: EntryLocation::Blob { cluster: 0, blob: 0 } };
        assert!(is_article(&article));
        let mut category = article.clone();
        category.path = "Category:Languages".into();
        assert!(!is_article(&category));
        let mut redirect = article;
        redirect.location = EntryLocation::Redirect(1);
        assert!(!is_article(&redirect));
    }
}
