use alloc::collections::BTreeMap as HashMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Element {
    pub text: String,
    pub attributes: HashMap<String, String>,
}

fn decode_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn attributes(opening: &str) -> HashMap<String, String> {
    let mut result = HashMap::new();
    let bytes = opening.as_bytes();
    let mut index = opening.find(char::is_whitespace).unwrap_or(opening.len());
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index].is_ascii_whitespace() || bytes[index] == b'/') {
            index += 1;
        }
        let key_start = index;
        while index < bytes.len()
            && (bytes[index].is_ascii_alphanumeric() || matches!(bytes[index], b'-' | b'_' | b':'))
        {
            index += 1;
        }
        if key_start == index {
            index += 1;
            continue;
        }
        let key = opening[key_start..index].to_ascii_lowercase();
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'=' {
            result.insert(key, String::new());
            continue;
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let quote = bytes
            .get(index)
            .copied()
            .filter(|byte| matches!(byte, b'\'' | b'"'));
        if quote.is_some() {
            index += 1;
        }
        let value_start = index;
        while index < bytes.len()
            && quote.map_or(!bytes[index].is_ascii_whitespace(), |q| bytes[index] != q)
        {
            index += 1;
        }
        result.insert(key, decode_entities(&opening[value_start..index]));
        if quote.is_some() && index < bytes.len() {
            index += 1;
        }
    }
    result
}

fn normalize_text(text: &str) -> String {
    decode_entities(&visible_text(text))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn elements(html: &str, tag: &str, class_contains: Option<&str>) -> Vec<Element> {
    let lower = html.to_ascii_lowercase();
    let opening_prefix = format!("<{tag}");
    let closing = format!("</{tag}>");
    let mut result = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = lower[cursor..].find(&opening_prefix) {
        let start = cursor + relative;
        let boundary = lower.as_bytes().get(start + opening_prefix.len()).copied();
        if boundary.is_some_and(|byte| !byte.is_ascii_whitespace() && byte != b'>' && byte != b'/')
        {
            cursor = start + opening_prefix.len();
            continue;
        }
        let Some(open_end_rel) = lower[start..].find('>') else {
            break;
        };
        let open_end = start + open_end_rel;
        let attrs = attributes(&html[start + 1..open_end]);
        if class_contains.is_some_and(|class| {
            !attrs
                .get("class")
                .is_some_and(|value| value.split_whitespace().any(|item| item == class))
        }) {
            cursor = open_end + 1;
            continue;
        }
        if matches!(tag, "img" | "meta" | "link" | "input") {
            result.push(Element {
                text: String::new(),
                attributes: attrs,
            });
            cursor = open_end + 1;
            continue;
        }
        let content_start = open_end + 1;
        let Some(close_rel) = lower[content_start..].find(&closing) else {
            break;
        };
        let content_end = content_start + close_rel;
        result.push(Element {
            text: normalize_text(&html[content_start..content_end]),
            attributes: attrs,
        });
        cursor = content_end + closing.len();
    }
    result
}

pub fn visible_text(html: &str) -> String {
    const NOISE: &[&str] = &[
        "script", "style", "noscript", "svg", "iframe", "nav", "footer",
    ];
    const BLOCK: &[&str] = &[
        "h1", "h2", "h3", "h4", "h5", "p", "article", "section", "li", "a", "div", "br",
    ];
    let mut output = String::with_capacity(html.len() / 3);
    let mut noise_depth = 0usize;
    let mut cursor = 0;
    while cursor < html.len() {
        if html.as_bytes()[cursor] == b'<' {
            let Some(end_rel) = html[cursor..].find('>') else {
                break;
            };
            let end = cursor + end_rel;
            let raw = html[cursor + 1..end].trim();
            let closing = raw.starts_with('/');
            let name = raw
                .trim_start_matches('/')
                .split(|character: char| character.is_whitespace() || character == '/')
                .next()
                .unwrap_or("")
                .to_ascii_lowercase();
            if NOISE.contains(&name.as_str()) {
                if closing {
                    noise_depth = noise_depth.saturating_sub(1);
                } else {
                    noise_depth += 1;
                }
            } else if noise_depth == 0 && BLOCK.contains(&name.as_str()) && !output.ends_with('\n')
            {
                output.push('\n');
            }
            cursor = end + 1;
        } else {
            let character = html[cursor..].chars().next().unwrap();
            if noise_depth == 0 {
                output.push(character);
            }
            cursor += character.len_utf8();
        }
    }
    let mut lines = Vec::new();
    for line in decode_entities(&output).lines() {
        let cleaned = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.len() >= 3 && !lines.contains(&cleaned) {
            lines.push(cleaned);
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_visible_text_and_skips_noise() {
        let html = "<h1>Hello &amp; hi</h1><script>bad()</script><p>Useful words here.</p>";
        assert_eq!(visible_text(html), "Hello & hi\nUseful words here.");
    }

    #[test]
    fn extracts_classed_links_and_attributes() {
        let links = elements(
            "<a class='result__a other' href='https://example.com'>An <b>Example</b></a>",
            "a",
            Some("result__a"),
        );
        assert_eq!(links[0].text, "An Example");
        assert_eq!(links[0].attributes["href"], "https://example.com");
    }
}
