use anyhow::Result;
use mrml_core::client::{ChatCompletionRequest, ChatMessage, MrmlClient, StreamEvent};
use mrml_json::{Value, object};
use mrml_runtime::{Shared, Text, Vector};
use mrml_terminal_style::Colorize;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

#[derive(Debug)]
pub struct OpenAiChatRequest {
    pub model: Option<Text>,
    pub messages: Vector<OpenAiMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct OpenAiMessage {
    pub role: Text,
    pub content: Text,
}

impl OpenAiChatRequest {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let source = core::str::from_utf8(bytes).map_err(|error| error.to_string())?;
        let value = mrml_json::parse(source).map_err(|error| error.to_string())?;
        let messages = value
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| "messages must be an array".to_owned())?
            .iter()
            .map(|message| {
                Ok(OpenAiMessage {
                    role: message
                        .get("role")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "message role must be a string".to_owned())?
                        .into(),
                    content: message
                        .get("content")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "message content must be a string".to_owned())?
                        .into(),
                })
            })
            .collect::<Result<Vector<_>, String>>()?;
        Ok(Self {
            model: value.get("model").and_then(Value::as_str).map(Text::from),
            messages,
            stream: value.get("stream").and_then(Value::as_bool),
            temperature: value
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            max_tokens: value
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map(|value| value as u32),
        })
    }
}

pub struct ApiServer {
    client: Shared<MrmlClient>,
    port: u16,
}

impl ApiServer {
    pub fn new(client: Shared<MrmlClient>, port: u16) -> Self {
        Self { client, port }
    }

    pub fn run(&self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr)?;
        println!(
            "{}",
            format!(
                "🌐 OpenAI-Compatible API Server listening on http://127.0.0.1:{}",
                self.port
            )
            .bright_green()
            .bold()
        );
        println!("   - Endpoints: /v1/models, /v1/chat/completions (supports SSE stream: true)");

        loop {
            let (socket, _) = match listener.accept() {
                Ok(conn) => conn,
                Err(e) => {
                    eprintln!("Socket accept error: {}", e);
                    continue;
                }
            };

            let client = self.client.clone();
            if mrml_runtime::spawn_detached(move || {
                if let Err(e) = handle_connection(socket, client) {
                    eprintln!("HTTP Handler Error: {}", e);
                }
            })
            .is_err()
            {
                eprintln!("HTTP Handler Error: failed to start MRML connection thread");
            }
        }
    }
}

fn handle_connection(mut socket: TcpStream, client: Shared<MrmlClient>) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut total_read = 0;

    // Read HTTP headers
    let header_end_pos = loop {
        let n = socket.read(&mut buf[total_read..])?;
        if n == 0 {
            return Ok(());
        }
        total_read += n;
        if let Some(pos) = find_subsequence(&buf[..total_read], b"\r\n\r\n") {
            break pos;
        }
        if total_read >= buf.len() {
            buf.resize(buf.len() * 2, 0);
        }
    };

    let header_str = String::from_utf8_lossy(&buf[..header_end_pos]);
    let mut lines = header_str.lines();
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    // Handle CORS preflight
    if method == "OPTIONS" {
        let resp = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n\r\n";
        socket.write_all(resp.as_bytes())?;
        return Ok(());
    }

    if method == "GET" && (path == "/v1/models" || path == "/models") {
        let now = (mrml_tools::platform::unix_timestamp_millis() / 1000) as u64;
        let body = mrml_json::stringify(&object([
            ("object", "list".into()),
            (
                "data",
                mrml_json::array([object([
                    ("id", "gemma-4-26B-A4B-it".into()),
                    ("object", "model".into()),
                    ("created", now.into()),
                    ("owned_by", "mrml".into()),
                ])]),
            ),
        ]));
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(resp.as_bytes())?;
        return Ok(());
    }

    if method == "POST" && (path == "/v1/chat/completions" || path == "/chat/completions") {
        let mut content_length = 0;
        for line in lines {
            if line.to_lowercase().starts_with("content-length:") {
                if let Some(val) = line.split(':').nth(1) {
                    content_length = val.trim().parse().unwrap_or(0);
                }
            }
        }

        let body_start = header_end_pos + 4;
        while total_read < body_start + content_length {
            let n = socket.read(&mut buf[total_read..])?;
            if n == 0 {
                break;
            }
            total_read += n;
        }

        let body_bytes = &buf[body_start..body_start + content_length];
        let chat_req = match OpenAiChatRequest::parse(body_bytes) {
            Ok(r) => r,
            Err(e) => {
                let err_body = mrml_json::stringify(&object([(
                    "error",
                    Value::text(format!("Invalid JSON: {e}")),
                )]));
                let resp = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    err_body.len(),
                    err_body
                );
                socket.write_all(resp.as_bytes())?;
                return Ok(());
            }
        };

        let stream_mode = chat_req.stream.unwrap_or(false);
        let messages: Vector<ChatMessage> = chat_req
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role.as_str().into(),
                content: Some(mrml_model::MessageContent::Text(m.content.as_str().into())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            })
            .collect();

        let req = ChatCompletionRequest {
            model: chat_req
                .model
                .unwrap_or_else(|| "gemma-4-26B-A4B-it".into()),
            messages,
            tools: None,
            stream: Some(stream_mode),
            temperature: chat_req.temperature,
            max_tokens: chat_req.max_tokens,
            tool_choice: None,
        };

        if stream_mode {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            socket.write_all(header.as_bytes())?;

            let now = (mrml_tools::platform::unix_timestamp_millis() / 1000) as u64;

            let mut disconnected = false;
            let stream_result = mrml_tools::block_on(client.stream_completion(&req, |event| {
                let piece = match event {
                    StreamEvent::Content(content) | StreamEvent::Reasoning(content) => content,
                    _ => return,
                };
                if !disconnected {
                    let line = format!("data: {}\n\n", sse_chunk(now, Some(piece), None));
                    disconnected = socket.write_all(line.as_bytes()).is_err();
                }
            }));
            stream_result?;

            let done_obj = sse_chunk(now, None, Some("stop"));
            if !disconnected {
                let _ =
                    socket.write_all(format!("data: {}\n\ndata: [DONE]\n\n", done_obj).as_bytes());
            }
            return Ok(());
        } else {
            let res = mrml_tools::block_on(client.chat_completion(&req))?;
            let choices = res
                .choices
                .into_iter()
                .map(|choice| {
                    object([
                        ("index", choice.index.into()),
                        ("message", chat_message_value(choice.message)),
                        ("finish_reason", Value::optional_text(choice.finish_reason)),
                    ])
                })
                .collect();
            let body = mrml_json::stringify(&object([
                ("id", Value::text(res.id)),
                ("choices", Value::Array(choices)),
            ]));
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes())?;
            return Ok(());
        }
    }

    let not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    socket.write_all(not_found.as_bytes())?;
    Ok(())
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn sse_chunk(created: u64, content: Option<Text>, finish_reason: Option<&str>) -> Text {
    let delta = content
        .map(|content| object([("content", Value::text(content))]))
        .unwrap_or_else(|| Value::Object(Default::default()));
    mrml_json::stringify(&object([
        ("id", "chatcmpl-mrml".into()),
        ("object", "chat.completion.chunk".into()),
        ("created", created.into()),
        ("model", "gemma-4-26B-A4B-it".into()),
        (
            "choices",
            mrml_json::array([object([
                ("index", 0usize.into()),
                ("delta", delta),
                ("finish_reason", Value::optional_text(finish_reason)),
            ])]),
        ),
    ]))
}

fn chat_message_value(message: ChatMessage) -> Value {
    let content = match message.content {
        Some(mrml_model::MessageContent::Text(text)) => Value::text(text),
        Some(mrml_model::MessageContent::Parts(parts)) => Value::Array(
            parts
                .into_iter()
                .map(|part| match part {
                    mrml_model::ContentPart::Text { text } => {
                        object([("type", "text".into()), ("text", Value::text(text))])
                    }
                    mrml_model::ContentPart::ImageUrl { image_url } => object([
                        ("type", "image_url".into()),
                        ("image_url", object([("url", Value::text(image_url.url))])),
                    ]),
                })
                .collect(),
        ),
        None => Value::Null,
    };
    let mut fields = [
        ("role", Value::text(message.role)),
        ("content", content),
        ("name", Value::optional_text(message.name)),
        ("tool_call_id", Value::optional_text(message.tool_call_id)),
        (
            "reasoning_content",
            Value::optional_text(message.reasoning_content),
        ),
        ("tool_calls", Value::Null),
    ];
    if let Some(calls) = message.tool_calls {
        fields[5].1 = Value::Array(
            calls
                .into_iter()
                .map(|call| {
                    object([
                        ("id", Value::text(call.id)),
                        ("type", Value::text(call.tool_type)),
                        (
                            "function",
                            object([
                                ("name", Value::text(call.function.name)),
                                ("arguments", Value::text(call.function.arguments)),
                            ]),
                        ),
                    ])
                })
                .collect(),
        );
    }
    Value::Object(
        fields
            .into_iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

#[cfg(test)]
mod json_tests {
    use super::*;

    #[test]
    fn parses_openai_request_and_rejects_invalid_messages() {
        let request = OpenAiChatRequest::parse(br#"{"model":"gemma","messages":[{"role":"user","content":"hello"}],"stream":true,"temperature":0.25,"max_tokens":32}"#).unwrap();
        assert_eq!(request.model.as_deref(), Some("gemma"));
        assert_eq!(request.messages[0].content, "hello");
        assert_eq!(request.stream, Some(true));
        assert!(OpenAiChatRequest::parse(br#"{"messages":[{"role":"user"}]}"#).is_err());
    }

    #[test]
    fn emits_valid_sse_chunks_with_escaped_content() {
        let chunk = sse_chunk(7, Some("hello\n\"world\"".into()), None);
        let value = mrml_json::parse(&chunk).unwrap();
        assert_eq!(value.get("created").and_then(Value::as_u64), Some(7));
        assert!(chunk.contains(r#"hello\n\"world\""#));
    }

    #[test]
    fn serves_models_over_standard_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = Shared::new(MrmlClient::new("", ""));
        let server = std::thread::spawn(move || {
            let (socket, _) = listener.accept().unwrap();
            handle_connection(socket, client).unwrap();
        });

        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        server.join().unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("gemma-4-26B-A4B-it"));
    }
}
