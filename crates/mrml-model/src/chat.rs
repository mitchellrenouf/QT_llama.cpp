use crate::error::{Error as ModelError, Result};
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
pub fn render_chat_template<T: Serialize>(
    template_source: &str,
    messages: &[ChatMessage],
    tools: Option<&[T]>,
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> Result<String> {
    if !template_source.contains("Google Gemma 4 Canonical Chat Template") {
        return Err(ModelError::message(
            "the embedded GGUF chat template is not a supported Gemma 4 canonical template",
        ));
    }
    let tools = tools
        .map(serde_json::to_value)
        .transpose()?
        .unwrap_or_else(|| serde_json::json!([]));
    render_gemma4_template(messages, tools.as_array().map(Vec::as_slice).unwrap_or(&[]), system_prompt, enable_thinking)
}

fn render_gemma4_template(
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    system_prompt: Option<&str>,
    enable_thinking: bool,
) -> Result<String> {
    let mut output = String::from("<bos>");
    let mut first = 0;
    let leading_system = messages.first().filter(|message| {
        message.role == "system" || message.role == "developer"
    });
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
        let role = if message.role == "assistant" { "model" } else { &message.role };
        output.push_str("<|turn>");
        output.push_str(role);
        output.push('\n');
        if message.role == "assistant" {
            if let Some(reasoning) = message.reasoning_content.as_deref().filter(|text| !text.is_empty()) {
                output.push_str("<|channel>thought\n");
                output.push_str(reasoning);
                output.push_str("\n<channel|>");
            }
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    let arguments: serde_json::Value = serde_json::from_str(&call.function.arguments)
                        .map_err(|error| ModelError::message(format!("invalid tool-call arguments passed to chat template: {error}")))?;
                    let arguments = arguments.as_object().ok_or_else(|| {
                        ModelError::message("tool-call arguments must be a JSON object")
                    })?;
                    output.push_str("<|tool_call>call:");
                    output.push_str(&call.function.name);
                    output.push('{');
                    for (position, (key, value)) in arguments.iter().enumerate() {
                        if position != 0 { output.push(','); }
                        output.push_str(key);
                        output.push(':');
                        output.push_str(&format_argument_canonical(value));
                    }
                    output.push_str("}<tool_call|>");
                }
                let mut follow = index + 1;
                while follow < messages.len() && messages[follow].role == "tool" {
                    let response = &messages[follow];
                    let name = calls.iter()
                        .find(|call| response.tool_call_id.as_deref() == Some(call.id.as_str()))
                        .map(|call| call.function.name.as_str())
                        .or(response.name.as_deref())
                        .unwrap_or("unknown");
                    output.push_str("<|tool_response>response:");
                    output.push_str(name);
                    output.push_str("{value:");
                    output.push_str(&format_argument_canonical(&serde_json::Value::String(
                        response.get_text_content().unwrap_or_default(),
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

fn format_tool_value(tool: &serde_json::Value) -> Result<String> {
    let function = tool.get("function").ok_or_else(|| ModelError::message("tool is missing function"))?;
    Ok(format_tool_declaration_canonical(
        function.get("name").and_then(|value| value.as_str()).unwrap_or("unknown"),
        function.get("description").and_then(|value| value.as_str()).unwrap_or(""),
        function.get("parameters").unwrap_or(&serde_json::Value::Null),
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

    #[derive(Serialize)]
    struct TestTool {
        function: serde_json::Value,
    }

    #[test]
    fn renders_messages_tools_and_generation_flags() {
        let tools = [TestTool {
            function: serde_json::json!({"name": "clock", "description": "", "parameters": null}),
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
        let rendered = render_chat_template::<serde_json::Value>(
            GEMMA_TEMPLATE,
            &[ChatMessage::assistant(None, Some(vec![call]))],
            None,
            None,
            false,
        )
        .unwrap();
        assert!(rendered.contains("<|tool_call>call:run_command{command_line:<|\"|>Get-Date<|\"|>}<tool_call|>"));
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
        let rendered = render_chat_template::<serde_json::Value>(
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
        let error = render_chat_template::<serde_json::Value>(
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
