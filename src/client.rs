use crate::config::Config;
use anyhow::{anyhow, Result};
pub use llama_cpp_binding::{
    format_gemma_chat, ChatMessage, FunctionCall, LlamaEngine, ToolCall,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

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
    Finish(String),
}

#[derive(Clone)]
pub struct LlamaClient {
    engine: Option<Arc<LlamaEngine>>,
    system_prompt: Option<String>,
}

impl LlamaClient {
    #[allow(dead_code)]
    pub fn new(_server_url: &str, _api_key: &str) -> Self {
        let model_path = find_model_file("gemma-4-26b-it-q4_0.gguf");
        let engine = if let Some(path) = model_path {
            match LlamaEngine::new(&path, 99, 8192) {
                Ok(eng) => Some(Arc::new(eng)),
                Err(e) => {
                    eprintln!("Notice: llama.cpp engine init deferred: {}", e);
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

    pub fn with_config(config: &Config) -> Self {
        let model_path = find_model_file(&config.model);
        let engine = if let Some(path) = model_path {
            match LlamaEngine::new(&path, 99, config.max_context_tokens as u32) {
                Ok(eng) => Some(Arc::new(eng)),
                Err(e) => {
                    eprintln!("Notice: llama.cpp engine init deferred: {}", e);
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

    pub async fn health_check(&self) -> Result<String> {
        if self.engine.is_some() {
            Ok("In-Process llama.cpp Engine Active".to_string())
        } else {
            Err(anyhow!("No active in-process llama.cpp engine loaded. (Place a .gguf model in .cache/gemma or pass --model <path>)"))
        }
    }

    pub async fn send_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let msg = self.stream_completion(request, |_| {}).await?;
        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", chrono::Utc::now().timestamp_millis()),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: msg,
                finish_reason: Some("stop".to_string()),
            }],
        })
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
            .ok_or_else(|| anyhow!("In-process llama.cpp engine is not loaded. Please specify a valid GGUF model path."))?;

        let mut sys_prompt = self.system_prompt.clone().unwrap_or_default();
        if let Some(tools) = &request.tools {
            sys_prompt.push_str("\n\n# AVAILABLE TOOLS:\n");
            for t in tools {
                sys_prompt.push_str(&format!(
                    "- `{}`: {}\n  Parameters: {}\n",
                    t.function.name, t.function.description, t.function.parameters
                ));
            }
            sys_prompt.push_str("\nTo call a tool, output a fenced codeblock with ```tool_call containing {\"name\": \"...\", \"arguments\": {...}}.\n");
        }

        let prompt = format_gemma_chat(&request.messages, Some(&sys_prompt));
        let max_tokens = request.max_tokens.unwrap_or(8192) as usize;
        let temp = request.temperature.unwrap_or(0.7);

        let (mut rx, _cancel) = engine.generate_stream(&prompt, max_tokens, temp);

        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut in_thought = false;
        let mut buffer = String::new();

        while let Some(piece_res) = rx.recv().await {
            let piece = piece_res?;
            buffer.push_str(&piece);

            if !in_thought {
                if let Some(pos) = buffer.find("<thought>") {
                    let before = &buffer[..pos];
                    if !before.is_empty() {
                        full_content.push_str(before);
                        callback(StreamEvent::Content(before.to_string()));
                    }
                    in_thought = true;
                    buffer = buffer[pos + 9..].to_string();
                } else if !buffer.starts_with('<') || buffer.len() > 10 {
                    full_content.push_str(&buffer);
                    callback(StreamEvent::Content(std::mem::take(&mut buffer)));
                }
            }

            if in_thought {
                if let Some(pos) = buffer.find("</thought>") {
                    let thought_text = &buffer[..pos];
                    if !thought_text.is_empty() {
                        full_reasoning.push_str(thought_text);
                        callback(StreamEvent::Reasoning(thought_text.to_string()));
                    }
                    in_thought = false;
                    buffer = buffer[pos + 10..].to_string();
                } else if !buffer.ends_with('<') || buffer.len() > 12 {
                    full_reasoning.push_str(&buffer);
                    callback(StreamEvent::Reasoning(std::mem::take(&mut buffer)));
                }
            }
        }

        if !buffer.is_empty() {
            if in_thought {
                full_reasoning.push_str(&buffer);
                callback(StreamEvent::Reasoning(buffer));
            } else {
                full_content.push_str(&buffer);
                callback(StreamEvent::Content(buffer));
            }
        }

        // Parse tool calls from content if any
        let mut tool_calls = Vec::new();
        if full_content.contains("```tool_call") {
            let re = regex::Regex::new(r"```tool_call\s*\n([\s\S]*?)\n```").unwrap();
            for cap in re.captures_iter(&full_content) {
                if let Some(m) = cap.get(1) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(m.as_str().trim()) {
                        if let Some(name) = json.get("name").and_then(|n| n.as_str()) {
                            let args = json.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
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

        callback(StreamEvent::Finish("stop".to_string()));

        let reasoning_opt = if full_reasoning.is_empty() {
            None
        } else {
            Some(full_reasoning)
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

pub fn find_model_file(model_arg: &str) -> Option<PathBuf> {
    let p = PathBuf::from(model_arg);
    if p.is_file() {
        return Some(p);
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
