use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrlDetail {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlDetail },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.map(MessageContent::Text),
            name: None,
            tool_call_id: None,
            tool_calls,
            reasoning_content: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.into())),
            name: Some(name.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            reasoning_content: None,
        }
    }

    pub fn get_text_content(&self) -> Option<String> {
        match &self.content {
            Some(MessageContent::Text(t)) => Some(t.clone()),
            Some(MessageContent::Parts(parts)) => {
                let mut text = String::new();
                for part in parts {
                    if let ContentPart::Text { text: t } = part {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(t);
                    }
                }
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
            None => None,
        }
    }
}

pub fn format_argument_canonical(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("<|\"|>{}<|\"|>", s),
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_argument_canonical).collect();
            format!("[{}]", items.join(","))
        }
        serde_json::Value::Object(map) => {
            let mut entries = Vec::new();
            for (k, v) in map {
                entries.push(format!("{}:{}", k, format_argument_canonical(v)));
            }
            format!("{{{}}}", entries.join(","))
        }
    }
}

pub fn format_tool_declaration_canonical(name: &str, description: &str, params: &serde_json::Value) -> String {
    let mut s = format!("declaration:{}{{description:<|\"|>{}<|\"|>", name, description);
    if let Some(props) = params.get("properties").and_then(|p| p.as_object()) {
        let mut prop_entries = Vec::new();
        for (prop_name, prop_val) in props {
            let desc = prop_val.get("description").and_then(|d| d.as_str()).unwrap_or("");
            let p_type = prop_val.get("type").and_then(|t| t.as_str()).unwrap_or("string").to_uppercase();
            prop_entries.push(format!("{}:{{description:<|\"|>{}<|\"|>,type:<|\"|>{}<|\"|>}}", prop_name, desc, p_type));
        }
        s.push_str(&format!(",parameters:{{properties:{{{}}},type:<|\"|>OBJECT<|\"|>}}", prop_entries.join(",")));
    }
    s.push('}');
    s
}

pub fn format_gemma_chat(messages: &[ChatMessage], system_prompt: Option<&str>) -> String {
    format_gemma_canonical_chat(messages, system_prompt, true)
}

pub fn format_gemma_canonical_chat(messages: &[ChatMessage], system_prompt: Option<&str>, enable_thinking: bool) -> String {
    let mut formatted = String::new();

    let mut system_text = String::new();
    if let Some(sys) = system_prompt {
        system_text.push_str(sys.trim());
    }

    // 1. System turn with thinking token if enabled
    if !system_text.is_empty() || enable_thinking {
        formatted.push_str("<|turn>system\n");
        if enable_thinking {
            formatted.push_str("<|think|>\n");
        }
        if !system_text.is_empty() {
            formatted.push_str(&system_text);
            formatted.push('\n');
        }
        formatted.push_str("<turn|>\n");
    }

    // 2. Turns
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                // Already handled in first turn
            }
            "user" => {
                formatted.push_str("<|turn>user\n");
                if let Some(text) = msg.get_text_content() {
                    formatted.push_str(text.trim());
                    formatted.push('\n');
                }
                formatted.push_str("<turn|>\n");
            }
            "assistant" => {
                formatted.push_str("<|turn>model\n");
                if let Some(thought) = &msg.reasoning_content {
                    let clean_thought = thought.trim();
                    if !clean_thought.is_empty() {
                        formatted.push_str("<|channel>thought\n");
                        formatted.push_str(clean_thought);
                        formatted.push_str("\n<channel|>\n");
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let args_val = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                            .unwrap_or(serde_json::json!({}));
                        formatted.push_str(&format!(
                            "<|tool_call>call:{}{}<tool_call|>",
                            tc.function.name,
                            format_argument_canonical(&args_val)
                        ));
                    }
                }
                if let Some(text) = msg.get_text_content() {
                    let clean_text = text.trim();
                    if !clean_text.is_empty() {
                        formatted.push_str(clean_text);
                        formatted.push('\n');
                    }
                }
                formatted.push_str("<turn|>\n");
            }
            "tool" => {
                formatted.push_str("<|turn>model\n");
                let name = msg.name.as_deref().unwrap_or("unknown");
                let content = msg.get_text_content().unwrap_or_default();
                formatted.push_str(&format!(
                    "<|tool_response>response:{}{{result:<|\"|>{}<|\"|>}}<tool_response|>\n",
                    name,
                    content.replace("<|\"|>", "")
                ));
                formatted.push_str("<turn|>\n");
            }
            _ => {}
        }
    }

    // 3. Prompt for model generation with thought channel
    formatted.push_str("<|turn>model\n");
    if enable_thinking {
        formatted.push_str("<|channel>thought\n");
    }
    formatted
}
