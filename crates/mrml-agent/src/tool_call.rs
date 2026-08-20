use mrml_model::{FunctionCall, ToolCall};
use mrml_runtime::{Text, Vector, mrml_format as format};

type String = Text;
type Vec<T> = Vector<T>;

fn quote_relaxed_keys(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = String::with_capacity(input.len() + 8).expect("MRML allocation failed");
    let mut index = 0;
    let mut quote = None;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(delimiter) = quote {
            output.push(byte as char);
            if byte == delimiter && (index == 0 || bytes[index - 1] != b'\\') {
                quote = None;
            }
            index += 1;
            continue;
        }
        if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
            output.push(byte as char);
            index += 1;
            continue;
        }
        output.push(byte as char);
        index += 1;
        if byte != b'{' && byte != b',' {
            continue;
        }
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            output.push(bytes[index] as char);
            index += 1;
        }
        let start = index;
        if index < bytes.len() && (bytes[index].is_ascii_alphabetic() || bytes[index] == b'_') {
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let end = index;
            while index < bytes.len() && bytes[index].is_ascii_whitespace() {
                index += 1;
            }
            if index < bytes.len() && bytes[index] == b':' {
                output.push('"');
                output.push_str(&input[start..end]);
                output.push('"');
                output.push_str(&input[end..index]);
                output.push(':');
                index += 1;
            } else {
                output.push_str(&input[start..index]);
            }
        }
    }
    output
}

fn quote_relaxed_values(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut output = String::with_capacity(input.len() + 8).expect("MRML allocation failed");
    let mut index = 0;
    let mut quote = None;
    while index < bytes.len() {
        let byte = bytes[index];
        output.push(byte as char);
        if let Some(delimiter) = quote {
            if byte == delimiter && (index == 0 || bytes[index - 1] != b'\\') {
                quote = None;
            }
            index += 1;
            continue;
        }
        if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
            index += 1;
            continue;
        }
        index += 1;
        if byte != b':' {
            continue;
        }
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            output.push(bytes[index] as char);
            index += 1;
        }
        if index >= bytes.len() || !bytes[index].is_ascii_alphabetic() {
            continue;
        }
        let start = index;
        while index < bytes.len() && !matches!(bytes[index], b',' | b'}') {
            if matches!(bytes[index], b'"' | b'{' | b'[' | b']' | b':') {
                break;
            }
            index += 1;
        }
        if index < bytes.len() && matches!(bytes[index], b',' | b'}') {
            let value = input[start..index].trim();
            if matches!(value, "true" | "false" | "null") || value.parse::<f64>().is_ok() {
                output.push_str(value);
            } else {
                output.push('"');
                output.push_str(value);
                output.push('"');
            }
        }
    }
    output
}

fn split_kwargs(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote = None;
    for (index, byte) in input.bytes().enumerate() {
        if let Some(delimiter) = quote {
            if byte == delimiter && (index == 0 || input.as_bytes()[index - 1] != b'\\') {
                quote = None;
            }
        } else if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
        } else if byte == b',' {
            parts.push(input[start..index].trim());
            start = index + 1;
        }
    }
    parts.push(input[start..].trim());
    parts
}

pub fn normalize_relaxed_json(raw: &str) -> String {
    let mut s = Text::from(raw.trim())
        .replace("<|\"|>", "\"")
        .replace("<|\"|", "\"")
        .replace("|\">", "\"")
        .replace("<|'|>", "'")
        .replace("<|'", "'")
        .replace("|'>", "'")
        .replace("<|", "")
        .replace("|>", "");

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        return serde_json::stringify(&v);
    }

    // Replace unquoted key names: {query: "foo"} -> {"query": "foo"}
    s = quote_relaxed_keys(&s);

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        return serde_json::stringify(&v);
    }

    // Quote unquoted string values: {"text": Hello world} -> {"text": "Hello world"}
    let s2 = quote_relaxed_values(&s);

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s2) {
        return serde_json::stringify(&v);
    }

    s
}

pub fn parse_kwargs_to_json(args: &str) -> String {
    let mut map = serde_json::Map::new();
    for part in split_kwargs(args) {
        if let Some((key, raw_value)) = part.split_once('=') {
            let key = key.trim();
            if key.is_empty()
                || !key.bytes().enumerate().all(|(index, byte)| {
                    byte == b'_'
                        || if index == 0 {
                            byte.is_ascii_alphabetic()
                        } else {
                            byte.is_ascii_alphanumeric()
                        }
                })
            {
                continue;
            }
            let raw_val = raw_value.trim().trim_end_matches(')');
            if raw_val.len() >= 2
                && ((raw_val.starts_with('"') && raw_val.ends_with('"'))
                    || (raw_val.starts_with('\'') && raw_val.ends_with('\'')))
            {
                map.insert(
                    key.into(),
                    serde_json::Value::String(raw_val[1..raw_val.len() - 1].into()),
                );
            } else {
                if let Ok(n) = raw_val.parse::<i64>() {
                    map.insert(key.into(), serde_json::json!(n));
                } else if let Ok(b) = raw_val.parse::<bool>() {
                    map.insert(key.into(), serde_json::json!(b));
                } else {
                    map.insert(key.into(), serde_json::Value::String(raw_val.into()));
                }
            }
        }
    }
    serde_json::stringify(&serde_json::Value::Object(map))
}

pub fn parse_gemma_tool_call(raw: &str) -> Option<ToolCall> {
    let text = raw
        .trim()
        .trim_start_matches("<|call>")
        .trim_start_matches("<|tool_call>")
        .trim_end_matches("<call|>")
        .trim_end_matches("</call>")
        .trim_end_matches("<tool_call|>")
        .trim_end_matches("</tool_call>")
        .trim();

    if text.is_empty() {
        return None;
    }

    // Format 1: JSON with "name" and "arguments"
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(name) = val.get("name").and_then(|v| v.as_str()) {
            let args = val
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let args_str = if args.is_string() {
                args.as_str().unwrap().into()
            } else {
                serde_json::stringify(&args)
            };
            return Some(ToolCall {
                id: format!("call_{}", crate::platform::unix_timestamp_millis())
                    .as_str()
                    .into(),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: name.into(),
                    arguments: args_str.as_str().into(),
                },
            });
        }
    }

    // Format 2: call:function_name{...} or call:function_name(...) or function_name{...}
    let stripped_call = text.trim_start_matches("call:").trim();
    if let Some(brace_pos) = stripped_call.find('{') {
        let name = stripped_call[..brace_pos].trim();
        let args_part = &stripped_call[brace_pos..];
        if !name.is_empty() {
            let normalized_args = normalize_relaxed_json(args_part);
            return Some(ToolCall {
                id: format!("call_{}", crate::platform::unix_timestamp_millis())
                    .as_str()
                    .into(),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: name.into(),
                    arguments: normalized_args.as_str().into(),
                },
            });
        }
    } else if let Some(paren_pos) = stripped_call.find('(') {
        let name = stripped_call[..paren_pos].trim();
        let end_paren = stripped_call.rfind(')').unwrap_or(stripped_call.len());
        let args_part = &stripped_call[paren_pos + 1..end_paren];
        if !name.is_empty() {
            let normalized_args = parse_kwargs_to_json(args_part);
            return Some(ToolCall {
                id: format!("call_{}", crate::platform::unix_timestamp_millis())
                    .as_str()
                    .into(),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: name.into(),
                    arguments: normalized_args.as_str().into(),
                },
            });
        }
    }

    None
}
