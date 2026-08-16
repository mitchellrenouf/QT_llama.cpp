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

pub fn format_gemma_chat(messages: &[ChatMessage], system_prompt: Option<&str>) -> String {
    let mut formatted = String::new();

    let mut system_text = String::new();
    if let Some(sys) = system_prompt {
        system_text.push_str(sys);
    }

    let mut first_user = true;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(text) = msg.get_text_content() {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(&text);
                }
            }
            "user" => {
                formatted.push_str("<start_of_turn>user\n");
                if first_user && !system_text.is_empty() {
                    formatted.push_str(&system_text);
                    formatted.push_str("\n\n");
                    first_user = false;
                }
                if let Some(text) = msg.get_text_content() {
                    formatted.push_str(&text);
                }
                formatted.push_str("<end_of_turn>\n");
            }
            "assistant" => {
                formatted.push_str("<start_of_turn>model\n");
                if let Some(thought) = &msg.reasoning_content {
                    formatted.push_str("<thought>\n");
                    formatted.push_str(thought);
                    formatted.push_str("\n</thought>\n");
                }
                if let Some(text) = msg.get_text_content() {
                    formatted.push_str(&text);
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        formatted.push_str(&format!("\n```tool_call\n{}\n```\n", serde_json::json!({
                            "name": tc.function.name,
                            "arguments": serde_json::from_str::<serde_json::Value>(&tc.function.arguments).unwrap_or(serde_json::Value::Null)
                        })));
                    }
                }
                formatted.push_str("<end_of_turn>\n");
            }
            "tool" => {
                formatted.push_str("<start_of_turn>tool\n");
                if let Some(name) = &msg.name {
                    formatted.push_str(&format!("[Tool: {}]\n", name));
                }
                if let Some(text) = msg.get_text_content() {
                    formatted.push_str(&text);
                }
                formatted.push_str("<end_of_turn>\n");
            }
            _ => {}
        }
    }

    // Append model prompt for next turn
    formatted.push_str("<start_of_turn>model\n");
    formatted
}
