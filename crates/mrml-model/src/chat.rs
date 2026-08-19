use crate::error::{Error as ModelError, Result};
use alloc::borrow::ToOwned;
use alloc::format;
use alloc::string::{String, ToString};
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub tool_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone)]
pub struct ImageUrlDetail {
    pub url: String,
}

#[derive(Debug, Clone)]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlDetail },
}

#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<MessageContent>,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
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

    pub fn tool(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
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
                if text.is_empty() { None } else { Some(text) }
            }
            None => None,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let content = match &self.content {
            Some(MessageContent::Text(text)) => text.as_str().into(),
            Some(MessageContent::Parts(parts)) => serde_json::Value::Array(
                parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => serde_json::object([
                            ("type", "text".into()),
                            ("text", text.as_str().into()),
                        ]),
                        ContentPart::ImageUrl { image_url } => serde_json::object([
                            ("type", "image_url".into()),
                            (
                                "image_url",
                                serde_json::object([("url", image_url.url.as_str().into())]),
                            ),
                        ]),
                    })
                    .collect(),
            ),
            None => serde_json::Value::Null,
        };
        let calls = self
            .tool_calls
            .as_ref()
            .map(|calls| {
                serde_json::Value::Array(
                    calls
                        .iter()
                        .map(|call| {
                            serde_json::object([
                                ("id", call.id.as_str().into()),
                                ("type", call.tool_type.as_str().into()),
                                (
                                    "function",
                                    serde_json::object([
                                        ("name", call.function.name.as_str().into()),
                                        ("arguments", call.function.arguments.as_str().into()),
                                    ]),
                                ),
                            ])
                        })
                        .collect(),
                )
            })
            .unwrap_or(serde_json::Value::Null);
        let fields = [
            ("role", self.role.as_str().into()),
            ("content", content),
            ("name", self.name.as_deref().into()),
            ("tool_call_id", self.tool_call_id.as_deref().into()),
            ("tool_calls", calls),
            (
                "reasoning_content",
                self.reasoning_content.as_deref().into(),
            ),
        ];
        serde_json::Value::Object(
            fields
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        let role = value
            .get("role")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ModelError::message("chat message role must be a string"))?
            .to_owned();
        let content = match value.get("content") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(text)) => Some(MessageContent::Text(text.to_string())),
            Some(serde_json::Value::Array(parts)) => Some(MessageContent::Parts(
                parts
                    .iter()
                    .map(
                        |part| match part.get("type").and_then(serde_json::Value::as_str) {
                            Some("text") => Ok(ContentPart::Text {
                                text: part
                                    .get("text")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("")
                                    .to_owned(),
                            }),
                            Some("image_url") => Ok(ContentPart::ImageUrl {
                                image_url: ImageUrlDetail {
                                    url: part
                                        .get("image_url")
                                        .and_then(|image| image.get("url"))
                                        .and_then(serde_json::Value::as_str)
                                        .unwrap_or("")
                                        .to_owned(),
                                },
                            }),
                            _ => Err(ModelError::message("unknown chat content part")),
                        },
                    )
                    .collect::<Result<Vec<_>>>()?,
            )),
            _ => return Err(ModelError::message("chat message content has invalid type")),
        };
        let tool_calls = value
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        let function = call
                            .get("function")
                            .ok_or_else(|| ModelError::message("tool call is missing function"))?;
                        Ok(ToolCall {
                            id: call
                                .get("id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("")
                                .to_owned(),
                            tool_type: call
                                .get("type")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("function")
                                .to_owned(),
                            function: FunctionCall {
                                name: function
                                    .get("name")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("")
                                    .to_owned(),
                                arguments: function
                                    .get("arguments")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("{}")
                                    .to_owned(),
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self {
            role,
            content,
            name: value
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            tool_call_id: value
                .get("tool_call_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            tool_calls,
            reasoning_content: value
                .get("reasoning_content")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
        })
    }
}

pub fn format_argument_canonical(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
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

pub fn format_tool_declaration_canonical(
    name: &str,
    description: &str,
    params: &serde_json::Value,
) -> String {
    let mut s = format!(
        "declaration:{}{{description:<|\"|>{}<|\"|>",
        name, description
    );
    if let Some(props) = params.get("properties").and_then(|p| p.as_object()) {
        let mut prop_entries = Vec::new();
        for (prop_name, prop_val) in props {
            let desc = prop_val
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let p_type = prop_val
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("string")
                .to_uppercase();
            prop_entries.push(format!(
                "{}:{{description:<|\"|>{}<|\"|>,type:<|\"|>{}<|\"|>}}",
                prop_name, desc, p_type
            ));
        }
        s.push_str(&format!(
            ",parameters:{{properties:{{{}}},type:<|\"|>OBJECT<|\"|>}}",
            prop_entries.join(",")
        ));
    }
    s.push('}');
    s
}

pub fn format_gemma_chat(messages: &[ChatMessage], system_prompt: Option<&str>) -> String {
    format_gemma_canonical_chat(messages, system_prompt, true)
}

/// Render the tokenizer-provided GGUF chat template with llama.cpp-compatible
/// inputs. Tool call arguments are converted from the OpenAI wire-format JSON
/// string into the mapping expected by tokenizer templates.
#[derive(Debug, Clone)]
pub struct TemplateTool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

pub fn render_chat_template(
    template_source: &str,
    messages: &[ChatMessage],
    tools: Option<&[TemplateTool]>,
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> Result<String> {
    if !template_source.contains("Google Gemma 4 Canonical Chat Template") {
        return Err(ModelError::message(
            "the embedded GGUF chat template is not a supported Gemma 4 canonical template",
        ));
    }
    render_gemma4_template(
        messages,
        tools.unwrap_or(&[]),
        system_prompt,
        enable_thinking,
    )
}

fn render_gemma4_template(
    messages: &[ChatMessage],
    tools: &[TemplateTool],
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> Result<String> {
    let mut output = String::from("<bos>");
    let mut first = 0;
    let leading_system = messages
        .first()
        .filter(|message| message.role == "system" || message.role == "developer");
    let system = system_prompt
        .filter(|text| !text.trim().is_empty())
        .map(str::to_owned)
        .or_else(|| leading_system.and_then(ChatMessage::get_text_content));
    if system_prompt.is_none() && leading_system.is_some() {
        first = 1;
    }
    if enable_thinking || !tools.is_empty() || system.is_some() {
        output.push_str("<|turn>system\n");
        if enable_thinking {
            output.push_str("<|think|>\n");
        }
        if let Some(system) = &system {
            output.push_str(system.trim());
        }
        for tool in tools {
            output.push_str("<|tool>");
            output.push_str(&format_tool_value(tool)?);
            output.push_str("<tool|>");
        }
        output.push_str("<turn|>\n");
    }

    let mut index = first;
    while index < messages.len() {
        let message = &messages[index];
        if message.role == "tool" {
            index += 1;
            continue;
        }
        let role = if message.role == "assistant" {
            "model"
        } else {
            &message.role
        };
        output.push_str("<|turn>");
        output.push_str(role);
        output.push('\n');
        if message.role == "assistant" {
            if let Some(reasoning) = message
                .reasoning_content
                .as_deref()
                .filter(|text| !text.is_empty())
            {
                output.push_str("<|channel>thought\n");
                output.push_str(reasoning);
                output.push_str("\n<channel|>");
            }
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    let arguments: serde_json::Value =
                        serde_json::from_str(&call.function.arguments).map_err(|error| {
                            ModelError::message(format!(
                                "invalid tool-call arguments passed to chat template: {error}"
                            ))
                        })?;
                    let arguments = arguments.as_object().ok_or_else(|| {
                        ModelError::message("tool-call arguments must be a JSON object")
                    })?;
                    output.push_str("<|tool_call>call:");
                    output.push_str(&call.function.name);
                    output.push('{');
                    for (position, (key, value)) in arguments.iter().enumerate() {
                        if position != 0 {
                            output.push(',');
                        }
                        output.push_str(key);
                        output.push(':');
                        output.push_str(&format_argument_canonical(value));
                    }
                    output.push_str("}<tool_call|>");
                }
                let mut follow = index + 1;
                while follow < messages.len() && messages[follow].role == "tool" {
                    let response = &messages[follow];
                    let name = calls
                        .iter()
                        .find(|call| response.tool_call_id.as_deref() == Some(call.id.as_str()))
                        .map(|call| call.function.name.as_str())
                        .or(response.name.as_deref())
                        .unwrap_or("unknown");
                    output.push_str("<|tool_response>response:");
                    output.push_str(name);
                    output.push_str("{value:");
                    output.push_str(&format_argument_canonical(&serde_json::Value::String(
                        response
                            .get_text_content()
                            .unwrap_or_default()
                            .as_str()
                            .into(),
                    )));
                    output.push_str("}<tool_response|>");
                    follow += 1;
                }
                index = follow.saturating_sub(1);
            }
        }
        if let Some(content) = message.get_text_content() {
            output.push_str(content.trim());
        }
        output.push_str("<turn|>\n");
        index += 1;
    }
    output.push_str("<|turn>model\n");
    if !enable_thinking {
        output.push_str("<|channel>thought\n<channel|>");
    }
    Ok(output)
}

fn format_tool_value(tool: &TemplateTool) -> Result<String> {
    Ok(format_tool_declaration_canonical(
        &tool.name,
        &tool.description,
        &tool.parameters,
    ))
}

pub fn format_gemma_canonical_chat(
    messages: &[ChatMessage],
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> String {
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
                        let args_val =
                            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
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

#[cfg(test)]
mod template_tests {
    use super::*;

    const GEMMA_TEMPLATE: &str = "Template: Google Gemma 4 Canonical Chat Template";

    #[test]
    fn chat_message_json_round_trip_preserves_tool_calls() {
        let mut message = ChatMessage::assistant(
            Some("done".into()),
            Some(vec![ToolCall {
                id: "call-9".into(),
                tool_type: "function".into(),
                function: FunctionCall {
                    name: "clock".into(),
                    arguments: r#"{"timezone":"UTC"}"#.into(),
                },
            }]),
        );
        message.reasoning_content = Some("checked the clock".into());

        let restored = ChatMessage::from_json(&message.to_json()).unwrap();
        assert_eq!(restored.role, "assistant");
        assert_eq!(restored.get_text_content().as_deref(), Some("done"));
        assert_eq!(
            restored.reasoning_content.as_deref(),
            Some("checked the clock")
        );
        let call = &restored.tool_calls.unwrap()[0];
        assert_eq!(call.id, "call-9");
        assert_eq!(call.function.name, "clock");
        assert_eq!(call.function.arguments, r#"{"timezone":"UTC"}"#);
    }

    #[test]
    fn renders_messages_tools_and_generation_flags() {
        let tools = [TemplateTool {
            name: "clock".into(),
            description: "".into(),
            parameters: serde_json::Value::Null,
        }];
        let rendered = render_chat_template(
            GEMMA_TEMPLATE,
            &[ChatMessage::user("what time is it?")],
            Some(&tools),
            Some("system rules"),
            true,
        )
        .unwrap();
        assert_eq!(
            rendered,
            "<bos><|turn>system\n<|think|>\nsystem rules<|tool>declaration:clock{description:<|\"|><|\"|>}<tool|><turn|>\n<|turn>user\nwhat time is it?<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn deserializes_tool_call_arguments_for_template_mappings() {
        let call = ToolCall {
            id: "call-1".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: "run_command".into(),
                arguments: r#"{"command_line":"Get-Date"}"#.into(),
            },
        };
        let rendered = render_chat_template(
            GEMMA_TEMPLATE,
            &[ChatMessage::assistant(None, Some(vec![call]))],
            None,
            None,
            false,
        )
        .unwrap();
        assert!(rendered.contains(
            "<|tool_call>call:run_command{command_line:<|\"|>Get-Date<|\"|>}<tool_call|>"
        ));
    }

    #[test]
    fn pairs_openai_tool_responses_with_function_names() {
        let call = ToolCall {
            id: "call-7".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: "clock".into(),
                arguments: "{}".into(),
            },
        };
        let rendered = render_chat_template(
            GEMMA_TEMPLATE,
            &[
                ChatMessage::assistant(None, Some(vec![call])),
                ChatMessage::tool("call-7", "ignored", "10:59"),
            ],
            None,
            None,
            true,
        )
        .unwrap();
        assert!(rendered.contains(
            "<|tool_call>call:clock{}<tool_call|><|tool_response>response:clock{value:<|\"|>10:59<|\"|>}<tool_response|>"
        ));
    }

    #[test]
    fn rejects_unknown_template_dialects() {
        let error = render_chat_template(
            "{{ messages }}",
            &[ChatMessage::user("hello")],
            None,
            None,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not a supported Gemma 4"));
    }
}
