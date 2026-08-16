use anyhow::{anyhow, Result};
pub use llama_cpp_binding::{format_gemma_chat, ChatMessage, FunctionCall, LlamaEngine, ToolCall};
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

    pub fn with_engine(engine: Arc<LlamaEngine>, system_prompt: Option<String>) -> Self {
        Self {
            engine: Some(engine),
            system_prompt,
        }
    }

    pub fn with_config(config: &Config) -> Self {
        let model_path = if let Some(hf_spec_str) = &config.hf {
            find_model_file(hf_spec_str).or_else(|| find_model_file(&config.model))
        } else {
            find_model_file(&config.model)
        };

        let engine = if let Some(path) = model_path {
            println!("Loading in-process GGUF model: {}", path.display());
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
        let mut tool_calls = Vec::new();

        while let Some(piece_res) = rx.recv().await {
            let piece = match piece_res {
                Ok(p) => p,
                Err(e) => return Err(e),
            };

            let combined = if in_thought {
                format!("{}{}", full_reasoning, piece)
            } else {
                format!("{}{}", full_content, piece)
            };

            let mut piece_clean = piece
                .replace("<|channel>thought", "")
                .replace("<|channel>", "")
                .replace("<channel|>", "")
                .replace("<tool_call|>", "")
                .replace("<thought>", "")
                .replace("</thought>", "");

            if full_content.is_empty() && (piece_clean.trim() == "thought" || piece_clean.trim() == "thought\n") {
                piece_clean = String::new();
            }

            if (combined.contains("<thought>") || combined.contains("<|channel>thought")) && !in_thought {
                in_thought = true;
            }

            if (combined.contains("</thought>") || combined.contains("<channel|>")) && in_thought {
                in_thought = false;
            }

            if in_thought {
                if !piece_clean.is_empty() {
                    callback(StreamEvent::Reasoning(piece_clean.clone()));
                    full_reasoning.push_str(&piece_clean);
                }
            } else if !piece_clean.is_empty() {
                callback(StreamEvent::Content(piece_clean.clone()));
                full_content.push_str(&piece_clean);

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

    if let Ok(spec) = crate::hf::HfModelSpec::parse(model_arg) {
        // 1. Check HF Hub cache (~/.cache/huggingface/hub/models--{user}--{model}/)
        let hf_hub_dir = dirs::home_dir().unwrap_or_default()
            .join(".cache/huggingface/hub")
            .join(format!("models--{}--{}", spec.user, spec.model));
        if hf_hub_dir.is_dir() {
            let target_quant = spec.quant.to_lowercase();
            for entry in walkdir::WalkDir::new(&hf_hub_dir).into_iter().flatten() {
                let path = entry.into_path();
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if name.ends_with(".gguf") && !name.contains("mmproj") && !name.contains("mtp") {
                    if name.contains(&target_quant) {
                        return Some(path);
                    }
                }
            }
            // fallback to any .gguf in this hub repo
            for entry in walkdir::WalkDir::new(&hf_hub_dir).into_iter().flatten() {
                let path = entry.into_path();
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if name.ends_with(".gguf") && !name.contains("mmproj") && !name.contains("mtp") {
                    return Some(path);
                }
            }
        }

        // 2. Check local gemma cache dir
        let model_dir = spec.get_model_dir();
        if model_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&model_dir) {
                let target_quant = spec.quant.to_lowercase();
                let mut best_match = None;
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if name.ends_with(".gguf") && !name.ends_with(".part") && !name.contains("mmproj") && !name.contains("mtp") {
                        if name.contains(&target_quant) {
                            return Some(path);
                        }
                        best_match = Some(path);
                    }
                }
                if let Some(m) = best_match {
                    return Some(m);
                }
            }
        }
    }

    // 3. Scan whole HF hub cache for matching quant or gemma-4
    let hf_hub_base = dirs::home_dir().unwrap_or_default().join(".cache/huggingface/hub");
    if hf_hub_base.is_dir() {
        for entry in walkdir::WalkDir::new(&hf_hub_base).into_iter().flatten() {
            let path = entry.into_path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
            if name.ends_with(".gguf") && !name.contains("mmproj") && !name.contains("mtp") {
                if name.contains("gemma-4") || name.contains("gemma") {
                    return Some(path);
                }
            }
        }
    }

    // 4. Scan ~/.cache/gemma/models
    let cache_dir = dirs::home_dir().unwrap_or_default().join(".cache/gemma/models");
    if cache_dir.is_dir() {
        for entry in walkdir::WalkDir::new(&cache_dir).into_iter().flatten() {
            if entry.file_type().is_file() {
                let path = entry.into_path();
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                if name.ends_with(".gguf") && !name.ends_with(".part") && !name.contains("mmproj") && !name.contains("mtp") {
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
