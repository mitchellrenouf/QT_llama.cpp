use anyhow::{anyhow, Result};
pub use mrml_model::{format_gemma_chat, ChatMessage, FunctionCall, ModelEngine, ToolCall};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::Config;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[allow(dead_code)]
pub type ToolFunction = FunctionDefinition;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Reasoning(String),
    Content(String),
    ToolCallAssembled(ToolCall),
    ToolExecuted { name: String, result: String },
    Metrics { token_count: usize, elapsed_secs: f64, tokens_per_sec: f64 },
    Finish(String),
}

pub fn normalize_relaxed_json(raw: &str) -> String {
    let mut s = raw
        .trim()
        .replace("<|\"|>", "\"")
        .replace("<|\"|", "\"")
        .replace("|\">", "\"")
        .replace("<|'|>", "'")
        .replace("<|'", "'")
        .replace("|'>", "'")
        .replace("<|", "")
        .replace("|>", "");

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        return v.to_string();
    }

    // Replace unquoted key names: {query: "foo"} -> {"query": "foo"}
    let re_keys = regex::Regex::new(r#"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:"#).unwrap();
    s = re_keys.replace_all(&s, r#"$1"$2":"#).to_string();

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
        return v.to_string();
    }

    // Quote unquoted string values: {"text": Hello world} -> {"text": "Hello world"}
    let re_vals = regex::Regex::new(r#":\s*([a-zA-Z][^"{}\[\]:,]+?)(\s*[},])"#).unwrap();
    let s2 = re_vals
        .replace_all(&s, |caps: &regex::Captures| {
            let val = caps.get(1).unwrap().as_str().trim();
            if val == "true" || val == "false" || val == "null" || val.parse::<f64>().is_ok() {
                format!(": {}{}", val, caps.get(2).unwrap().as_str())
            } else {
                format!(": \"{}\"{}", val, caps.get(2).unwrap().as_str())
            }
        })
        .to_string();

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s2) {
        return v.to_string();
    }

    s
}

pub fn parse_kwargs_to_json(args: &str) -> String {
    let mut map = serde_json::Map::new();
    let re = regex::Regex::new(r#"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^,)]+))"#).unwrap();
    for cap in re.captures_iter(args) {
        let key = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let val_str = cap.get(2).or_else(|| cap.get(3)).map(|m| m.as_str());
        if let Some(s) = val_str {
            map.insert(key.to_string(), serde_json::Value::String(s.to_string()));
        } else if let Some(raw_val) = cap.get(4).map(|m| m.as_str().trim()) {
            if let Ok(n) = raw_val.parse::<i64>() {
                map.insert(key.to_string(), serde_json::json!(n));
            } else if let Ok(b) = raw_val.parse::<bool>() {
                map.insert(key.to_string(), serde_json::json!(b));
            } else {
                map.insert(key.to_string(), serde_json::Value::String(raw_val.to_string()));
            }
        }
    }
    serde_json::Value::Object(map).to_string()
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
            let args = val.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            let args_str = if args.is_string() {
                args.as_str().unwrap().to_string()
            } else {
                args.to_string()
            };
            return Some(ToolCall {
                id: format!("call_{}", chrono::Utc::now().timestamp_millis()),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: args_str,
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
                id: format!("call_{}", chrono::Utc::now().timestamp_millis()),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: normalized_args,
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
                id: format!("call_{}", chrono::Utc::now().timestamp_millis()),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: normalized_args,
                },
            });
        }
    }

    None
}

#[derive(Clone)]
pub struct MrmlClient {
    engine: Option<Arc<ModelEngine>>,
    system_prompt: Option<String>,
}

impl MrmlClient {
    #[allow(dead_code)]
    pub fn new(_server_url: &str, _api_key: &str) -> Self {
        let model_path = find_model_file("gemma-4-26b-it-q4_0.gguf");
        let engine = if let Some(path) = model_path {
            match ModelEngine::new(&path, -1, 8192, "auto", "auto", None) {
                Ok(eng) => Some(Arc::new(eng)),
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
        }
    }

    #[allow(dead_code)]
    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = Some(prompt);
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        self.engine.is_some()
    }

    pub fn with_engine(engine: Arc<ModelEngine>, system_prompt: Option<String>) -> Self {
        Self {
            engine: Some(engine),
            system_prompt,
        }
    }

    pub fn with_config(config: &Config) -> Self {
        let explicit_model = PathBuf::from(&config.model);
        let model_path = if explicit_model.is_file() {
            Some(explicit_model)
        } else if let Some(hf_spec_str) = config.hf.as_deref().filter(|s| !s.trim().is_empty()) {
            find_model_file(hf_spec_str).or_else(|| find_model_file(&config.model))
        } else {
            find_model_file(&config.model)
        };

        let engine = if let Some(path) = model_path {
            println!("Loading in-process GGUF model: {}", path.display());
            let n_layers = config.n_gpu_layers.unwrap_or(-1);
            let backend_str = config.backend.to_string();
            match ModelEngine::new(
                &path,
                n_layers,
                config.ctx_size,
                &config.cache_type_k,
                &config.cache_type_v,
                Some(&backend_str),
            ) {
                Ok(eng) => Some(Arc::new(eng)),
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
            system_prompt: config.system_prompt.clone(),
        }
    }

    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    pub async fn health_check(&self) -> Result<String> {
        if self.engine.is_some() {
            Ok("Native MRML Engine Active".to_string())
        } else {
            Err(anyhow!("No active native MRML engine loaded. (Place a .gguf model in .cache/gemma or pass --model <path>)"))
        }
    }

    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let mut text_acc = String::new();
        let mut thought_acc = String::new();
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
            id: format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: msg,
                finish_reason: Some("stop".to_string()),
            }],
        })
    }

    pub async fn send_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
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

        let prompt = if let Some(template) = chat_template {
            mrml_model::render_chat_template(
                &template,
                &request.messages,
                request.tools.as_deref(),
                Some(&sys_prompt),
                true,
            )?
        } else {
            format_gemma_chat(&request.messages, Some(&sys_prompt))
        };
        let max_tokens = request.max_tokens.unwrap_or(8192) as usize;
        let temp = request.temperature.unwrap_or(0.7);

        let (mut rx, _cancel) = engine.generate_stream(&prompt, max_tokens, temp);

        let mut first_token_time: Option<std::time::Instant> = None;
        let mut token_count = 0usize;

        let mut raw_acc = String::new();
        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut in_thought = prompt.ends_with("<|channel>thought\n") || prompt.ends_with("<|channel>thought");
        let mut tool_calls = Vec::new();

        while let Some(piece_res) = rx.recv().await {
            let chunk = match piece_res {
                Ok(p) => p,
                Err(e) => return Err(e),
            };
            let piece = chunk.text;

            let start = first_token_time.get_or_insert_with(std::time::Instant::now);
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
                let tool_start_opt = raw_acc.find("<|call>").or_else(|| raw_acc.find("<|tool_call>"));
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
                                    callback(StreamEvent::Reasoning(clean.to_string()));
                                    full_reasoning.push_str(clean);
                                }
                            } else {
                                callback(StreamEvent::Content(before.to_string()));
                                full_content.push_str(before);
                            }
                        }

                        let tool_raw = &raw_acc[tool_start..end_pos];
                        if let Some(tc) = parse_gemma_tool_call(tool_raw) {
                            callback(StreamEvent::ToolCallAssembled(tc.clone()));
                            tool_calls.push(tc);
                        }

                        raw_acc = raw_acc[end_pos..].to_string();
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
                                    callback(StreamEvent::Reasoning(clean.to_string()));
                                    full_reasoning.push_str(clean);
                                }
                            } else {
                                callback(StreamEvent::Content(before.to_string()));
                                full_content.push_str(before);
                            }
                            raw_acc = raw_acc[tool_start..].to_string();
                        }
                        break;
                    }
                }

                if !in_thought {
                    // Check for thought opening tag: "<|channel>" or "<thought>"
                    if let Some(pos) = raw_acc.find("<|channel>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.to_string()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<|channel>".len()..].to_string();
                        let trimmed = raw_acc.trim_start();
                        if trimmed.starts_with("thought") {
                            raw_acc = trimmed["thought".len()..]
                                .trim_start_matches(|c| c == '\n' || c == '\r' || c == ' ')
                                .to_string();
                        }
                        in_thought = true;
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("<thought>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.to_string()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<thought>".len()..].to_string();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        in_thought = true;
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("<channel|>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.to_string()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<channel|>".len()..].to_string();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        continue;
                    }
                    if let Some(pos) = raw_acc.find("</channel>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.to_string()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "</channel>".len()..].to_string();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        continue;
                    }

                    // Check end of turn
                    if let Some(pos) = raw_acc.find("<end_of_turn>") {
                        let before = &raw_acc[..pos];
                        if !before.is_empty() {
                            callback(StreamEvent::Content(before.to_string()));
                            full_content.push_str(before);
                        }
                        raw_acc = raw_acc[pos + "<end_of_turn>".len()..].to_string();
                        continue;
                    }

                    // Prefix check for potential tags
                    let prefixes = [
                        "<", "<|", "<|c", "<|ch", "<|channel", "<|t", "<|tool", "<|tool_call",
                        "<t", "<th", "<thought", "<e", "<end", "<end_of_turn",
                    ];
                    if let Some(&prefix) = prefixes.iter().find(|&&p| raw_acc.ends_with(p)) {
                        let keep_len = prefix.len();
                        let emit_len = raw_acc.len() - keep_len;
                        if emit_len > 0 {
                            let to_emit = raw_acc[..emit_len].to_string();
                            callback(StreamEvent::Content(to_emit.clone()));
                            full_content.push_str(&to_emit);
                            raw_acc = raw_acc[emit_len..].to_string();
                        }
                        break;
                    }

                    if !raw_acc.is_empty() {
                        let chunk = std::mem::take(&mut raw_acc);
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
                        .or_else(|| raw_acc.find("<end_of_turn>").map(|p| (p, "<end_of_turn>".len())));

                    if let Some((pos, tag_len)) = close_opt {
                        let thought_part = &raw_acc[..pos];
                        let clean = thought_part.trim().trim_start_matches("thought").trim();
                        if !clean.is_empty() && clean != "thought" {
                            callback(StreamEvent::Reasoning(clean.to_string()));
                            full_reasoning.push_str(clean);
                        }
                        raw_acc = raw_acc[pos + tag_len..].to_string();
                        if raw_acc.starts_with('\n') {
                            raw_acc.remove(0);
                        }
                        in_thought = false;
                        continue;
                    }

                    // Prefix check for closing tags
                    let prefixes = [
                        "<", "</", "</c", "</ch", "</channel", "<c", "<ch", "<channel",
                        "<channel|", "</t", "</th", "</thought", "<e", "<end", "<end_of_turn",
                        "<|t", "<|tool",
                    ];
                    if let Some(&prefix) = prefixes.iter().find(|&&p| raw_acc.ends_with(p)) {
                        let keep_len = prefix.len();
                        let emit_len = raw_acc.len() - keep_len;
                        if emit_len > 0 {
                            let to_emit = raw_acc[..emit_len].to_string();
                            let clean = to_emit.trim();
                            if !clean.is_empty() && clean != "thought" {
                                callback(StreamEvent::Reasoning(to_emit.clone()));
                                full_reasoning.push_str(&to_emit);
                            }
                            raw_acc = raw_acc[emit_len..].to_string();
                        }
                        break;
                    }

                    if !raw_acc.is_empty() {
                        let chunk = std::mem::take(&mut raw_acc);
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
                let re = regex::Regex::new(r"```tool_call\s*(\{[\s\S]*?\})\s*```").unwrap();
                for cap in re.captures_iter(&full_content) {
                    if let Some(json_match) = cap.get(1) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_match.as_str()) {
                            let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let args = val.get("arguments").cloned().unwrap_or(serde_json::json!({}));
                            let args_str = if args.is_string() {
                                args.as_str().unwrap().to_string()
                            } else {
                                args.to_string()
                            };
                            let tc = ToolCall {
                                id: format!("call_{}", chrono::Utc::now().timestamp_millis()),
                                tool_type: "function".to_string(),
                                function: FunctionCall {
                                    name: name.to_string(),
                                    arguments: args_str,
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
                        callback(StreamEvent::Reasoning(clean.to_string()));
                        full_reasoning.push_str(clean);
                    }
                } else {
                    callback(StreamEvent::Content(clean_tail.clone()));
                    full_content.push_str(&clean_tail);
                }
            }
        }

        let total_elapsed = first_token_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
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

        callback(StreamEvent::Finish("stop".to_string()));

        let clean_full_reasoning = full_reasoning
            .trim()
            .trim_start_matches("thought")
            .trim()
            .to_string();
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

        let mut msg = ChatMessage::assistant(content_opt, tool_calls_opt);
        msg.reasoning_content = reasoning_opt;
        Ok(msg)
    }
}

pub fn get_model_cache_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(p).join("hub"));
    }
    if let Ok(p) = std::env::var("MRML_CACHE") {
        roots.push(PathBuf::from(p));
    }

    #[cfg(windows)]
    {
        if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
            roots.push(PathBuf::from(&local_appdata).join("huggingface").join("hub"));
        }
        if let Ok(userprofile) = std::env::var("USERPROFILE") {
            roots.push(PathBuf::from(&userprofile).join(".cache").join("huggingface").join("hub"));
        }
    }

    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".cache").join("huggingface").join("hub"));
        roots.push(home.join(".cache").join("gemma").join("models"));
    }

    let mut unique_roots = Vec::new();
    for r in roots {
        if r.is_dir() && !unique_roots.contains(&r) {
            unique_roots.push(r);
        }
    }
    unique_roots
}

pub fn find_model_file(model_arg: &str) -> Option<PathBuf> {
    let p = PathBuf::from(model_arg);
    if p.is_file() {
        return Some(p);
    }

    let cache_roots = get_model_cache_roots();

    if let Ok(spec) = crate::hf::HfModelSpec::parse(model_arg) {
        let repo_slug = format!("models--{}--{}", spec.user, spec.model);
        let target_quant = spec.quant.to_lowercase();

        // 1. Search for matching repo slug in Hugging Face cache directories
        for root in &cache_roots {
            let repo_dir = root.join(&repo_slug);
            if repo_dir.is_dir() {
                let mut best_match = None;
                for entry in walkdir::WalkDir::new(&repo_dir).into_iter().flatten() {
                    let path = entry.into_path();
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if name.ends_with(".gguf") && !name.ends_with(".part") && !name.contains("mmproj") && !name.contains("mtp") {
                        if name.contains(&target_quant) {
                            return Some(path);
                        }
                        if best_match.is_none() {
                            best_match = Some(path);
                        }
                    }
                }
                if let Some(m) = best_match {
                    return Some(m);
                }
            }

            // Legacy folder name check (e.g. user_model)
            let legacy_dir = root.join(format!("{}_{}", spec.user, spec.model));
            if legacy_dir.is_dir() {
                for entry in walkdir::WalkDir::new(&legacy_dir).into_iter().flatten() {
                    let path = entry.into_path();
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if name.ends_with(".gguf") && !name.ends_with(".part") && !name.contains("mmproj") && !name.contains("mtp") {
                        if name.contains(&target_quant) {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }

    // 2. Scan whole cache roots for matching model file
    for root in &cache_roots {
        for entry in walkdir::WalkDir::new(root).into_iter().flatten() {
            let path = entry.into_path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            if name.ends_with(".gguf") && !name.ends_with(".part") && !name.contains("mmproj") && !name.contains("mtp") {
                if name.contains("gemma-4") || name.contains("gemma") {
                    return Some(path);
                }
            }
        }
    }

    let candidates = [
        PathBuf::from("models").join(model_arg),
        PathBuf::from(model_arg).with_extension("gguf"),
        dirs::home_dir().unwrap_or_default().join(".cache/gemma").join(model_arg),
        dirs::home_dir().unwrap_or_default().join(".cache/gemma/gemma-4-26b-it-q4_0.gguf"),
        PathBuf::from("/models/gemma-4-26b-it-q4_0.gguf"),
    ];

    for c in candidates {
        if c.is_file() {
            return Some(c);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
