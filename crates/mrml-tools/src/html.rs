use mrml_runtime::{OrderedMap, Text, Vector};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Element {
    pub text: Text,
    pub attributes: OrderedMap<Text, Text>,
}

fn replace_all(text: &str, needle: &str, replacement: &str) -> Text {
    let mut output = Text::with_capacity(text.len()).expect("MRML allocation failed");
    let mut remainder = text;
    while let Some(index) = remainder.find(needle) {
        output.push_str(&remainder[..index]);
        output.push_str(replacement);
        remainder = &remainder[index + needle.len()..];
    }
    output.push_str(remainder);
    output
}

fn decode_entities(text: &str) -> Text {
    let mut output = Text::from(text);
    for (needle, replacement) in [
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", "\""),
        ("&#39;", "'"),
        ("&nbsp;", " "),
    ] {
        output = replace_all(&output, needle, replacement);
    }
    output
}

fn ascii_lowercase(text: &str) -> Text {
    let mut output = Text::with_capacity(text.len()).expect("MRML allocation failed");
    for character in text.chars() {
        output.push(character.to_ascii_lowercase());
    }
    output
}

fn attributes(opening: &str) -> OrderedMap<Text, Text> {
    let mut result = OrderedMap::new();
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
        let key = ascii_lowercase(&opening[key_start..index]);
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] != b'=' {
            result.insert(key, Text::new());
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

fn normalized_words(text: &str) -> Text {
    let mut output = Text::with_capacity(text.len()).expect("MRML allocation failed");
    for word in text.split_whitespace() {
        if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(word);
    }
    output
}

fn normalize_text(text: &str) -> Text {
    normalized_words(&decode_entities(&visible_text(text)))
}

pub fn elements(html: &str, tag: &str, class_contains: Option<&str>) -> Vector<Element> {
    let lower = ascii_lowercase(html);
    let mut opening_prefix = Text::from("<");
    opening_prefix.push_str(tag);
    let mut closing = Text::from("</");
    closing.push_str(tag);
    closing.push('>');
    let mut result = Vector::new();
    let mut cursor = 0;
    while let Some(relative) = lower[cursor..].find(opening_prefix.as_str()) {
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
                text: Text::new(),
                attributes: attrs,
            });
            cursor = open_end + 1;
            continue;
        }
        let content_start = open_end + 1;
        let Some(close_rel) = lower[content_start..].find(closing.as_str()) else {
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

pub fn visible_text(html: &str) -> Text {
    const NOISE: &[&str] = &[
        "script", "style", "noscript", "svg", "iframe", "nav", "footer",
    ];
    const BLOCK: &[&str] = &[
        "h1", "h2", "h3", "h4", "h5", "p", "article", "section", "li", "a", "div", "br",
    ];
    let mut output = Text::with_capacity(html.len() / 3).expect("MRML allocation failed");
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
            let name = ascii_lowercase(
                raw.trim_start_matches('/')
                    .split(|character: char| character.is_whitespace() || character == '/')
                    .next()
                    .unwrap_or(""),
            );
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
    let mut lines = Vector::new();
    for line in decode_entities(&output).lines() {
        let cleaned = normalized_words(line);
        if cleaned.len() >= 3 && !lines.contains(&cleaned) {
            lines.push(cleaned);
        }
    }
    let mut visible = Text::new();
    for (index, line) in lines.iter().enumerate() {
        if index != 0 {
            visible.push('\n');
        }
        visible.push_str(line);
    }
    visible
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
        assert_eq!(
            links[0].attributes.get("href").unwrap(),
            "https://example.com"
        );
    }
}
