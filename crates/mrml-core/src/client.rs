use anyhow::{Result, anyhow};
pub use mrml_model::{ChatMessage, FunctionCall, ModelEngine, ToolCall, format_gemma_chat};
use mrml_runtime::{Instant, Shared, Text, Vector, mrml_eprintln as eprintln, mrml_println as println};
use std::path::PathBuf;

type String = Text;
type Vec<T> = Vector<T>;

use crate::config::Config;
pub use mrml_tools::{FunctionDefinition, ToolDefinition};

pub(crate) fn thinking_enabled_for_mode(mode: crate::config::AgentMode) -> bool {
    matches!(mode, crate::config::AgentMode::Automatic)
}

#[allow(dead_code)]
pub type ToolFunction = FunctionDefinition;

#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: Text,
    pub messages: Vector<ChatMessage>,
    pub tools: Option<Vector<ToolDefinition>>,
    pub tool_choice: Option<Text>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<Text>,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: Text,
    pub choices: Vector<ChatCompletionChoice>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Reasoning(Text),
    Content(Text),
    ToolCallAssembled(ToolCall),
    ToolExecuted {
        name: Text,
        result: Text,
    },
    Metrics {
        token_count: usize,
        elapsed_secs: f64,
        tokens_per_sec: f64,
    },
    Finish(Text),
}

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
                args.as_str().unwrap().to_string()
            } else {
                args.to_string()
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

#[derive(Clone)]
pub struct MrmlClient {
    engine: Option<Shared<ModelEngine>>,
    system_prompt: Option<Text>,
    enable_thinking: bool,
}

impl MrmlClient {
    #[allow(dead_code)]
    pub fn new(_server_url: &str, _api_key: &str) -> Self {
        let model_path = find_model_file("gemma-4-26b-it-q4_0.gguf");
        let engine = if let Some(path) = model_path {
            let path_text = path.to_string_lossy();
            match ModelEngine::new(&path_text, -1, 8192, "auto", "auto", None) {
                Ok(eng) => Some(Shared::new(eng)),
                Err(e) => {
                    eprintln!("Notice: MRML engine init deferred: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            engine,
            system_prompt: None,
            enable_thinking: false,
        }
    }

    #[allow(dead_code)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt.as_str().into());
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        self.engine.is_some()
    }

    pub fn with_engine(
        engine: Shared<ModelEngine>,
        system_prompt: Option<Text>,
        enable_thinking: bool,
    ) -> Self {
        Self {
            engine: Some(engine),
            system_prompt,
            enable_thinking,
        }
    }

    pub fn set_thinking_enabled(&mut self, enabled: bool) {
        self.enable_thinking = enabled;
    }

    pub fn with_config(config: &Config) -> Self {
        let explicit_model = PathBuf::from(config.model.as_str());
        let model_path = if explicit_model
            .to_str()
            .is_some_and(crate::platform::path_is_file)
        {
            Some(explicit_model)
        } else if let Some(hf_spec_str) = config.hf.as_deref().filter(|s| !s.trim().is_empty()) {
            find_model_file(hf_spec_str).or_else(|| find_model_file(&config.model))
        } else {
            find_model_file(&config.model)
        };

        let engine = if let Some(path) = model_path {
            println!("Loading in-process GGUF model: {}", path.display());
            let path_text = path.to_string_lossy();
            let n_layers = config.n_gpu_layers.unwrap_or(-1);
            let backend_str = config.backend.to_string();
            match ModelEngine::new(
                &path_text,
                n_layers,
                config.ctx_size,
                &config.cache_type_k,
                &config.cache_type_v,
                Some(&backend_str),
            ) {
                Ok(eng) => Some(Shared::new(eng)),
                Err(e) => {
                    eprintln!("Notice: MRML engine init deferred: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            engine,
            system_prompt: config.system_prompt.as_deref().map(Text::from),
            enable_thinking: thinking_enabled_for_mode(config.mode),
        }
    }

    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    pub fn gpu_layer_residency(&self) -> Option<(usize, usize)> {
        self.engine.as_ref()?.gpu_layer_residency()
    }

    pub async fn health_check(&self) -> Result<String> {
        if self.engine.is_some() {
            Ok("Native MRML Engine Active".into())
        } else {
            Err(anyhow!(
                "No active native MRML engine loaded. (Place a .gguf model in .cache/gemma or pass --model <path>)"
            ))
        }
    }

    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let mut text_acc = Text::new();
        let mut thought_acc = Text::new();
        let mut assembled_tool_calls = Vec::new();

        let msg = self
            .stream_completion(request, |event| match event {
                StreamEvent::Content(c) => text_acc.push_str(&c),
                StreamEvent::Reasoning(r) => thought_acc.push_str(&r),
                StreamEvent::ToolCallAssembled(tc) => assembled_tool_calls.push(tc),
                StreamEvent::ToolExecuted { .. } => {}
                StreamEvent::Metrics { .. } => {}
                StreamEvent::Finish(_) => {}
            })
            .await?;

        Ok(ChatCompletionResponse {
            id: Text::from(
                format!("chatcmpl-{}", crate::platform::unix_timestamp_millis()).as_str(),
            ),
            choices: [ChatCompletionChoice {
                index: 0,
                message: msg,
                finish_reason: Some("stop".into()),
            }]
            .into_iter()
            .collect(),
        })
    }

    pub async fn send_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.chat_completion(request).await
    }

    pub async fn stream_completion<F>(
        &self,
        request: &ChatCompletionRequest,
        mut callback: F,
    ) -> Result<ChatMessage>
    where
        F: FnMut(StreamEvent),
    {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MRML engine not loaded"))?;

        let mut sys_prompt = self.system_prompt.clone().unwrap_or_default();
        let chat_template = engine.chat_template();
        if chat_template.is_none() {
            if let Some(tools) = &request.tools {
                for tool in tools {
                    sys_prompt.push_str(&format!(
                        "<|tool|>{}<tool|>\n",
                        mrml_model::format_tool_declaration_canonical(
                            &tool.function.name,
                            &tool.function.description,
                            &tool.function.parameters
                        )
                    ));
                }
            }
        }

        let template_tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| mrml_model::TemplateTool {
                    name: tool.function.name.as_str().into(),
                    description: tool.function.description.as_str().into(),
                    parameters: tool.function.parameters.clone(),
                })
                .collect::<Vec<_>>()
        });
        let prompt = if let Some(template) = chat_template {
            mrml_model::render_chat_template(
                &template,
                &request.messages,
                template_tools.as_deref(),
                Some(&sys_prompt),
                self.enable_thinking,
            )
            .map_err(anyhow::Error::with_source)?
        } else {
            format_gemma_chat(&request.messages, Some(&sys_prompt))
        };
        let max_tokens = request.max_tokens.unwrap_or(8192) as usize;
        let temp = request.temperature.unwrap_or(0.7);

        let (rx, _cancel) = engine.generate_stream(&prompt, max_tokens, temp);

        let mut first_token_time: Option<Instant> = None;
        let mut token_count = 0usize;

        let mut raw_acc = Text::new();
        let mut full_content = Text::new();
        let mut full_reasoning = Text::new();
        let mut in_thought =
            prompt.ends_with("<|channel>thought\n") || prompt.ends_with("<|channel>thought");
        let mut tool_calls = Vec::new();

        while let Ok(piece_res) = rx.recv() {
            let chunk = match piece_res {
                Ok(p) => p,
                Err(e) => return Err(anyhow::Error::with_source(e)),
            };
            let piece = chunk.text;

            let start = first_token_time.get_or_insert_with(Instant::now);
            token_count += chunk.token_count;
            let elapsed = start.elapsed().as_secs_f64();
            let tps = if token_count > 1 && elapsed > 0.0001 {
                (token_count - 1) as f64 / elapsed
            } else if elapsed > 0.0001 {
                token_count as f64 / elapsed
            } else {
                0.0
            };

            callback(StreamEvent::Metrics {
                token_count,
                elapsed_secs: elapsed,
                tokens_per_sec: tps,
            });

            raw_acc.push_str(&piece);

            loop {
                // 1. Check for tool calls: "<|call>" or "<|tool_call>"
                let tool_start_opt = raw_acc
                    .find("<|call>")
                    .or_else(|| raw_acc.find("<|tool_call>"));
                if let Some(tool_start) = tool_start_opt {
                    let end_opt = raw_acc[tool_start..]
                        .find("<call|>")
                        .or_else(|| raw_acc[tool_start..].find("</call>"))
                        .or_else(|| raw_acc[tool_start..].find("<tool_call|>"))
                        .or_else(|| raw_acc[tool_start..].find("</tool_call>"));

                    if let Some(rel_end) = end_opt {
                        let rest = &raw_acc[tool_start + rel_end..];
                        let tag_len = if rest.starts_with("<call|>") {
                            "<call|>".len()
                        } else if rest.starts_with("</call>") {
                            "</call>".len()
                        } else if rest.starts_with("<tool_call|>") {
                            "<tool_call|>".len()
                        } else {
                            "</tool_call>".len()
                        };
                        let end_pos = tool_start + rel_end + tag_len;
                        let before = &raw_acc[..tool_start];
                        if !before.is_empty() {
                            if in_thought {
                                let clean = before.trim().trim_start_matches("thought").trim();
                                if !clean.is_empty() && clean != "thought" {
                                    callback(StreamEvent::Reasoning(clean.into()));
                                    full_reasoning.push_str(clean);
                                }
                            } else {
                                callback(StreamEvent::Content(before.into()));
                                full_content.push_str(before);
                            }
                        }

                        let tool_raw = &raw_acc[tool_start..end_pos];
                        if let Some(tc) = parse_gemma_tool_call(tool_raw) {
                            callback(StreamEvent::ToolCallAssembled(tc.clone()));
                            tool_calls.push(tc);
                        }

                        raw_acc = raw_acc[end_pos..].into();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        in_thought = false;
                        continue;
                    } else {
                        // Incomplete tool call, wait for closing tag
                        let before = &raw_acc[..tool_start];
                        if !before.is_empty() {
                            if in_thought {
                                let clean = before.trim().trim_start_matches("thought").trim();
                                if !clean.is_empty() && clean != "thought" {
                                    callback(StreamEvent::Reasoning(clean.into()));
                                    full_reasoning.push_str(clean);
                                }
                            } else {
                                callback(StreamEvent::Content(before.into()));
                                full_content.push_str(before);
                            }
                            raw_acc = raw_acc[tool_start..].into();
                        }
                        break;
                    }
                }

                if !in_thought {
                    // Check for thought opening tag: "<|channel>" or "<thought>"
                    if let Some(pos) = raw_acc.find("<|channel>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.into()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<|channel>".len()..].into();
                        let trimmed = raw_acc.trim_start();
                        if trimmed.starts_with("thought") {
                            raw_acc = trimmed["thought".len()..]
                                .trim_start_matches(|c| c == '\n' || c == '\r' || c == ' ')
                                .into();
                        }
                        in_thought = true;
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("<thought>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.into()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<thought>".len()..].into();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        in_thought = true;
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("<channel|>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.into()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<channel|>".len()..].into();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("</channel>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.into()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "</channel>".len()..].into();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        continue;
                    }

                    // Check end of turn
                    if let Some(pos) = raw_acc.find("<end_of_turn>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.into()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<end_of_turn>".len()..].into();
                        continue;
                    }

                    // Prefix check for potential tags
                    let prefixes = [
                        "<",
                        "<|",
                        "<|c",
                        "<|ch",
                        "<|channel",
                        "<|t",
                        "<|tool",
                        "<|tool_call",
                        "<t",
                        "<th",
                        "<thought",
                        "<e",
                        "<end",
                        "<end_of_turn",
                    ];
                    if let Some(&prefix) = prefixes.iter().find(|&&p| raw_acc.ends_with(p)) {
                        let keep_len = prefix.len();
                        let emit_len = raw_acc.len() - keep_len;
                        if emit_len > 0 {
                            let to_emit: Text = raw_acc[..emit_len].into();
                            callback(StreamEvent::Content(to_emit.clone()));
                            full_content.push_str(&to_emit);
                            raw_acc = raw_acc[emit_len..].into();
                        }
                        break;
                    }

                    if !raw_acc.is_empty() {
                        let chunk = core::mem::take(&mut raw_acc);
                        callback(StreamEvent::Content(chunk.clone()));
                        full_content.push_str(&chunk);
                    }
                    break;
                } else {
                    // Inside thought: look for closing tags: "</channel>", "<channel|>", "</thought>", "<end_of_turn>"
                    let close_opt = raw_acc
                        .find("</channel>")
                        .map(|p| (p, "</channel>".len()))
                        .or_else(|| raw_acc.find("<channel|>").map(|p| (p, "<channel|>".len())))
                        .or_else(|| raw_acc.find("</thought>").map(|p| (p, "</thought>".len())))
                        .or_else(|| {
                            raw_acc
                                .find("<end_of_turn>")
                                .map(|p| (p, "<end_of_turn>".len()))
                        });

                    if let Some((pos, tag_len)) = close_opt {
                        let thought_part = &raw_acc[..pos];
                        let clean = thought_part.trim().trim_start_matches("thought").trim();
                        if !clean.is_empty() && clean != "thought" {
                            callback(StreamEvent::Reasoning(clean.into()));
                            full_reasoning.push_str(clean);
                        }
                        raw_acc = raw_acc[pos + tag_len..].into();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        in_thought = false;
                        continue;
                    }

                    // Prefix check for closing tags
                    let prefixes = [
                        "<",
                        "</",
                        "</c",
                        "</ch",
                        "</channel",
                        "<c",
                        "<ch",
                        "<channel",
                        "<channel|",
                        "</t",
                        "</th",
                        "</thought",
                        "<e",
                        "<end",
                        "<end_of_turn",
                        "<|t",
                        "<|tool",
                    ];
                    if let Some(&prefix) = prefixes.iter().find(|&&p| raw_acc.ends_with(p)) {
                        let keep_len = prefix.len();
                        let emit_len = raw_acc.len() - keep_len;
                        if emit_len > 0 {
                            let to_emit: Text = raw_acc[..emit_len].into();
                            let clean = to_emit.trim();
                            if !clean.is_empty() && clean != "thought" {
                                callback(StreamEvent::Reasoning(to_emit.clone()));
                                full_reasoning.push_str(&to_emit);
                            }
                            raw_acc = raw_acc[emit_len..].into();
                        }
                        break;
                    }

                    if !raw_acc.is_empty() {
                        let chunk = core::mem::take(&mut raw_acc);
                        let clean = chunk.trim();
                        if !clean.is_empty() && clean != "thought" {
                            callback(StreamEvent::Reasoning(chunk.clone()));
                            full_reasoning.push_str(&chunk);
                        }
                    }
                    break;
                }
            }

            // Tool Call detection in markdown codeblocks
            if full_content.contains("```tool_call") && full_content.ends_with("```") {
                for block in full_content.split("```tool_call").skip(1) {
                    if let Some(json_match) = block.split("```").next().map(str::trim) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_match) {
                            let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args = val
                                .get("arguments")
                                .cloned()
                                .unwrap_or(serde_json::json!({}));
                            let args_str = if args.is_string() {
                                args.as_str().unwrap().to_string()
                            } else {
                                args.to_string()
                            };
                            let tc = ToolCall {
                                id: format!("call_{}", crate::platform::unix_timestamp_millis())
                                    .as_str()
                                    .into(),
                                tool_type: "function".into(),
                                function: FunctionCall {
                                    name: name.into(),
                                    arguments: args_str.as_str().into(),
                                },
                            };
                            callback(StreamEvent::ToolCallAssembled(tc.clone()));
                            tool_calls.push(tc);
                        }
                    }
                }
            }
        }

        // Flush remaining buffer at EOF
        if !raw_acc.is_empty() {
            if raw_acc.contains("<|tool_call>") || raw_acc.contains("<|call>") {
                if let Some(tc) = parse_gemma_tool_call(&raw_acc) {
                    callback(StreamEvent::ToolCallAssembled(tc.clone()));
                    tool_calls.push(tc);
                }
            } else {
                let clean_tail = raw_acc
                    .replace("<|channel>", "")
                    .replace("</channel>", "")
                    .replace("<channel|>", "")
                    .replace("<thought>", "")
                    .replace("</thought>", "")
                    .replace("<end_of_turn>", "")
                    .replace("<start_of_turn>", "")
                    .replace("<|im_end|>", "");
                if in_thought {
                    let clean = clean_tail.trim().trim_start_matches("thought").trim();
                    if !clean.is_empty() && clean != "thought" {
                        callback(StreamEvent::Reasoning(clean.into()));
                        full_reasoning.push_str(clean);
                    }
                } else {
                    callback(StreamEvent::Content(clean_tail.clone()));
                    full_content.push_str(&clean_tail);
                }
            }
        }

        let total_elapsed = first_token_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let final_tps = if token_count > 1 && total_elapsed > 0.0001 {
            (token_count - 1) as f64 / total_elapsed
        } else if total_elapsed > 0.0001 {
            token_count as f64 / total_elapsed
        } else {
            0.0
        };
        callback(StreamEvent::Metrics {
            token_count,
            elapsed_secs: total_elapsed,
            tokens_per_sec: final_tps,
        });

        callback(StreamEvent::Finish("stop".into()));

        let clean_full_reasoning =
            Text::from(full_reasoning.trim().trim_start_matches("thought").trim());
        let reasoning_opt = if clean_full_reasoning.is_empty()
            || clean_full_reasoning == "thought"
            || clean_full_reasoning == "</channel>"
            || clean_full_reasoning == "<channel|>"
        {
            None
        } else {
            Some(clean_full_reasoning)
        };

        let content_opt = if full_content.is_empty() {
            None
        } else {
            Some(full_content)
        };

        let tool_calls_opt = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        let mut msg = ChatMessage::assistant(
            content_opt.map(|content| content.as_str().into()),
            tool_calls_opt.map(|calls| calls.into_iter().collect()),
        );
        msg.reasoning_content = reasoning_opt.map(|reasoning| reasoning.as_str().into());
        Ok(msg)
    }
}

pub fn get_model_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(p) = mrml_runtime::environment_variable("HF_HUB_CACHE") {
        roots.push(PathBuf::from(p.as_str()));
    }
    if let Some(p) = mrml_runtime::environment_variable("HF_HOME") {
        roots.push(PathBuf::from(p.as_str()).join("hub"));
    }
    if let Some(p) = mrml_runtime::environment_variable("MRML_CACHE") {
        roots.push(PathBuf::from(p.as_str()));
    }

    #[cfg(windows)]
    {
        if let Some(local_appdata) = mrml_runtime::environment_variable("LOCALAPPDATA") {
            roots.push(
                PathBuf::from(local_appdata.as_str())
                    .join("huggingface")
                    .join("hub"),
            );
        }
        if let Some(userprofile) = mrml_runtime::environment_variable("USERPROFILE") {
            roots.push(
                PathBuf::from(userprofile.as_str())
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            );
        }
    }

    if let Some(home) = crate::platform::home_dir() {
        let home = PathBuf::from(home.as_str());
        roots.push(home.join(".cache").join("huggingface").join("hub"));
        roots.push(home.join(".cache").join("gemma").join("models"));
    }

    let mut unique_roots = Vec::new();
    for r in roots {
        if r.to_str().is_some_and(mrml_runtime::path_is_directory)
            && !unique_roots.contains(&r)
        {
            unique_roots.push(r);
        }
    }
    unique_roots
}

pub fn find_model_file(model_arg: &str) -> Option<PathBuf> {
    let p = PathBuf::from(model_arg);
    if p.to_str().is_some_and(crate::platform::path_is_file) {
        return Some(p);
    }

    let cache_roots = get_model_cache_roots();

    if let Ok(spec) = crate::hf::HfModelSpec::parse(model_arg) {
        let repo_slug = format!("models--{}--{}", spec.user, spec.model);
        let target_quant = spec.quant.to_lowercase();

        // 1. Search for matching repo slug in Hugging Face cache directories
        for root in &cache_roots {
            let repo_dir = root.join(&repo_slug);
            if repo_dir
                .to_str()
                .is_some_and(mrml_runtime::path_is_directory)
            {
                let mut best_match = None;
                for path in crate::fs_walk::paths(repo_dir.to_str().unwrap_or("")) {
                    let name = path
                        .rsplit(['/', '\\'])
                        .next()
                        .unwrap_or(&path)
                        .to_lowercase();
                    if name.ends_with(".gguf")
                        && !name.ends_with(".part")
                        && !name.contains("mmproj")
                        && !name.contains("mtp")
                    {
                        if name.contains(&target_quant) {
                            return Some(PathBuf::from(path.as_str()));
                        }
                        if best_match.is_none() {
                            best_match = Some(PathBuf::from(path.as_str()));
                        }
                    }
                }
                if let Some(m) = best_match {
                    return Some(m);
                }
            }

            // Legacy folder name check (e.g. user_model)
            let legacy_dir = root.join(format!("{}_{}", spec.user, spec.model));
            if legacy_dir
                .to_str()
                .is_some_and(mrml_runtime::path_is_directory)
            {
                for path in crate::fs_walk::paths(legacy_dir.to_str().unwrap_or("")) {
                    let name = path
                        .rsplit(['/', '\\'])
                        .next()
                        .unwrap_or(&path)
                        .to_lowercase();
                    if name.ends_with(".gguf")
                        && !name.ends_with(".part")
                        && !name.contains("mmproj")
                        && !name.contains("mtp")
                    {
                        if name.contains(&target_quant) {
                            return Some(PathBuf::from(path.as_str()));
                        }
                    }
                }
            }
        }
    }

    // 2. Scan whole cache roots for matching model file
    for root in &cache_roots {
        for path in crate::fs_walk::paths(root.to_str().unwrap_or("")) {
            let name = path
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(&path)
                .to_lowercase();
            if name.ends_with(".gguf")
                && !name.ends_with(".part")
                && !name.contains("mmproj")
                && !name.contains("mtp")
            {
                if name.contains("gemma-4") || name.contains("gemma") {
                    return Some(PathBuf::from(path.as_str()));
                }
            }
        }
    }

    let candidates = [
        PathBuf::from("models").join(model_arg),
        PathBuf::from(model_arg).with_extension("gguf"),
        PathBuf::from(
            mrml_runtime::join_path(
                &mrml_runtime::join_path(
                    &crate::platform::home_dir().unwrap_or_default(),
                    ".cache/gemma",
                ),
                model_arg,
            )
            .as_str(),
        ),
        PathBuf::from(
            mrml_runtime::join_path(
                &crate::platform::home_dir().unwrap_or_default(),
                ".cache/gemma/gemma-4-26b-it-q4_0.gguf",
            )
            .as_str(),
        ),
        PathBuf::from("/models/gemma-4-26b-it-q4_0.gguf"),
    ];

    for c in candidates {
        if c.to_str().is_some_and(crate::platform::path_is_file) {
            return Some(c);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_is_reserved_for_automatic_mode() {
        assert!(!thinking_enabled_for_mode(
            crate::config::AgentMode::General
        ));
        assert!(!thinking_enabled_for_mode(crate::config::AgentMode::Coder));
        assert!(thinking_enabled_for_mode(
            crate::config::AgentMode::Automatic
        ));
    }

    #[test]
    fn test_explicit_model_path_wins_over_hf_default() {
        let path = std::env::temp_dir().join(format!("mrml-explicit-{}.gguf", std::process::id()));
        std::fs::write(&path, b"test").unwrap();
        assert_eq!(find_model_file(path.to_str().unwrap()), Some(path.clone()));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_parse_gemma_tool_call_screenshot() {
        let raw = r#"<|tool_call>call:web_search{query: "image of a bat"}<tool_call|>"#;
        let tc = parse_gemma_tool_call(raw).expect("should parse tool call");
        assert_eq!(tc.function.name, "web_search");
        assert!(tc.function.arguments.contains("image of a bat"));
    }

    #[test]
    fn test_parse_gemma_tool_call_json() {
        let raw = r#"<|tool_call>{"name": "fetch_url", "arguments": {"url": "https://example.com"}}<tool_call|>"#;
        let tc = parse_gemma_tool_call(raw).expect("should parse tool call");
        assert_eq!(tc.function.name, "fetch_url");
        assert!(tc.function.arguments.contains("https://example.com"));
    }

    #[test]
    fn test_parse_gemma_tool_call_kwargs() {
        let raw = r#"<|tool_call>call:bash_execute(command="ls -la")</tool_call>"#;
        let tc = parse_gemma_tool_call(raw).expect("should parse tool call");
        assert_eq!(tc.function.name, "bash_execute");
        assert!(tc.function.arguments.contains("ls -la"));
    }

    #[test]
    fn test_parse_gemma_tool_call_with_unquoted_command() {
        let raw = r#"<|tool_call>call:run_command{command_line: Get-Date}<tool_call|>"#;
        let tc = parse_gemma_tool_call(raw).expect("should parse relaxed tool call");
        assert_eq!(tc.function.name, "run_command");
        let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
        assert_eq!(args["command_line"], "Get-Date");
    }
}
