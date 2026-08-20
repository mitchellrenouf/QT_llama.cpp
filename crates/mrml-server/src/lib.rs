#![no_std]

use mrml_agent::client::{ChatCompletionRequest, ChatMessage, MrmlClient, StreamEvent};
use mrml_error::Result;
use mrml_json::{Value, object};
use mrml_runtime::{
    Shared, TcpListener, TcpStream, Text, Text as String, Vector, mrml_eprintln as eprintln,
    mrml_format as format, mrml_println as println,
};
use mrml_terminal_style::Colorize;
use mrml_tls::{TlsServerConfig, TlsServerStream};

enum Connection {
    Plain(TcpStream),
    Tls(TlsServerStream),
}
impl Connection {
    fn read(&mut self, output: &mut [u8]) -> Result<usize> {
        Ok(match self {
            Self::Plain(s) => s.read(output)?,
            Self::Tls(s) => s
                .read(output)
                .map_err(|e| mrml_error::message(format!("{e}")))?,
        })
    }
    fn write_all(&mut self, input: &[u8]) -> Result<()> {
        match self {
            Self::Plain(s) => s.write_all(input)?,
            Self::Tls(s) => s
                .write_all(input)
                .map_err(|e| mrml_error::message(format!("{e}")))?,
        }
        Ok(())
    }
}

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
        let source = core::str::from_utf8(bytes).map_err(|error| format!("{}", error))?;
        let value = mrml_json::parse(source).map_err(|error| format!("{}", error))?;
        let messages = value
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| Text::from("messages must be an array"))?
            .iter()
            .map(|message| {
                Ok(OpenAiMessage {
                    role: message
                        .get("role")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Text::from("message role must be a string"))?
                        .into(),
                    content: message
                        .get("content")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Text::from("message content must be a string"))?
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
    tls: Option<Shared<TlsServerConfig>>,
    bearer_token: Option<Shared<Text>>,
}

impl ApiServer {
    pub fn new(client: Shared<MrmlClient>, port: u16) -> Self {
        Self {
            client,
            port,
            tls: None,
            bearer_token: None,
        }
    }

    pub fn with_bearer_token(mut self, token: Text) -> Result<Self> {
        if token.len() < 32 || !token.is_ascii() || token.bytes().any(|b| b.is_ascii_whitespace()) {
            return Err(mrml_error::message(
                "MRML_API_TOKEN must be at least 32 non-whitespace ASCII bytes",
            ));
        }
        self.bearer_token = Some(Shared::new(token));
        Ok(self)
    }

    pub fn with_tls_pem(mut self, certificate_pem: &[u8], private_key_pem: &[u8]) -> Result<Self> {
        self.tls = Some(Shared::new(
            TlsServerConfig::from_pem(certificate_pem, private_key_pem)
                .map_err(|e| mrml_error::message(format!("{e}")))?,
        ));
        Ok(self)
    }

    pub fn run(&self) -> Result<()> {
        let token = self
            .bearer_token
            .clone()
            .ok_or_else(|| mrml_error::message("bearer authentication is required"))?;
        let listener = TcpListener::bind([127, 0, 0, 1], self.port)?;
        println!(
            "{}",
            format!(
                "🌐 OpenAI-Compatible API Server listening on {}://127.0.0.1:{}",
                if self.tls.is_some() { "https" } else { "http" },
                self.port
            )
            .bright_green()
            .bold()
        );
        println!("   - Endpoints: /v1/models, /v1/chat/completions (supports SSE stream: true)");

        loop {
            let socket = match listener.accept() {
                Ok(socket) => socket,
                Err(e) => {
                    eprintln!("Socket accept error: {}", e);
                    continue;
                }
            };

            let client = self.client.clone();
            let tls = self.tls.clone();
            let token = token.clone();
            if mrml_runtime::spawn_detached(move || {
                let connection = match tls {
                    Some(config) => match TlsServerStream::accept(socket, &config) {
                        Ok(stream) => Connection::Tls(stream),
                        Err(error) => {
                            eprintln!("TLS handshake error: {}", error);
                            return;
                        }
                    },
                    None => Connection::Plain(socket),
                };
                if let Err(e) = handle_connection(connection, client, &token) {
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

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= left.get(index).copied().unwrap_or(0) as usize
            ^ right.get(index).copied().unwrap_or(0) as usize;
    }
    difference == 0
}

fn authorized(headers: &str, expected: &str) -> bool {
    let mut authorization = None;
    for line in headers.lines().skip(1) {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("authorization")
        {
            if authorization.is_some() {
                return false;
            }
            authorization = Some(value.trim());
        }
    }
    authorization
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| constant_time_equal(token.as_bytes(), expected.as_bytes()))
}

fn handle_connection(
    mut socket: Connection,
    client: Shared<MrmlClient>,
    bearer_token: &str,
) -> Result<()> {
    const MAX_HEADER_BYTES: usize = 64 * 1024;
    const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
    let mut buf = Vector::new();
    buf.resize(8192, 0);
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
        if total_read >= MAX_HEADER_BYTES {
            return Err(mrml_error::message("HTTP request headers exceed 64 KiB"));
        }
        if total_read >= buf.len() {
            buf.resize((buf.len() * 2).min(MAX_HEADER_BYTES), 0);
        }
    };

    let header_str = core::str::from_utf8(&buf[..header_end_pos]).map_err(mrml_error::message)?;
    if !authorized(header_str, bearer_token) {
        socket.write_all(b"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer\r\nCache-Control: no-store\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
        return Ok(());
    }
    let mut lines = header_str.lines();
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    if method == "OPTIONS" {
        let resp = "HTTP/1.1 405 Method Not Allowed\r\nAllow: GET, POST\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
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
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(resp.as_bytes())?;
        return Ok(());
    }

    if method == "POST" && (path == "/v1/chat/completions" || path == "/chat/completions") {
        let mut content_length = None;
        for line in lines {
            if let Some((name, val)) = line.split_once(':') {
                if name.eq_ignore_ascii_case("content-length") {
                    let parsed = val
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| mrml_error::message("invalid HTTP Content-Length"))?;
                    if content_length.is_some_and(|prior| prior != parsed) {
                        return Err(mrml_error::message(
                            "conflicting HTTP Content-Length headers",
                        ));
                    }
                    content_length = Some(parsed);
                } else if name.eq_ignore_ascii_case("transfer-encoding") {
                    return Err(mrml_error::message(
                        "HTTP request transfer encoding is unsupported",
                    ));
                }
            }
        }
        let content_length = content_length.unwrap_or(0);
        if content_length > MAX_BODY_BYTES {
            return Err(mrml_error::message("HTTP request body exceeds 16 MiB"));
        }

        let body_start = header_end_pos + 4;
        let body_end = body_start
            .checked_add(content_length)
            .ok_or_else(|| mrml_error::message("HTTP request size overflow"))?;
        if buf.len() < body_end {
            buf.resize(body_end, 0);
        }
        while total_read < body_end {
            let n = socket.read(&mut buf[total_read..])?;
            if n == 0 {
                return Err(mrml_error::message("truncated HTTP request body"));
            }
            total_read += n;
        }

        let body_bytes = &buf[body_start..body_end];
        let chat_req = match OpenAiChatRequest::parse(body_bytes) {
            Ok(r) => r,
            Err(e) => {
                let err_body = mrml_json::stringify(&object([(
                    "error",
                    Value::text(format!("Invalid JSON: {e}")),
                )]));
                let resp = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
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
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Content-Type-Options: nosniff\r\n\r\n";
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
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nContent-Length: {}\r\n\r\n{}",
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
    fn serves_models_over_native_tcp() {
        let listener = TcpListener::bind([127, 0, 0, 1], 0).unwrap();
        let port = listener.local_port().unwrap();
        let client = Shared::new(MrmlClient::new("", ""));
        assert!(
            mrml_runtime::spawn_detached(move || {
                let socket = listener.accept().unwrap();
                handle_connection(
                    Connection::Plain(socket),
                    client,
                    "01234567890123456789012345678901",
                )
                .unwrap();
            })
            .is_ok()
        );

        let mut socket = TcpStream::connect([127, 0, 0, 1], port).unwrap();
        socket
            .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer 01234567890123456789012345678901\r\n\r\n")
            .unwrap();
        let mut response = Vector::new();
        let mut buffer = [0u8; 1024];
        loop {
            let read = socket.read(&mut buffer).unwrap();
            if read == 0 {
                break;
            }
            response.try_extend_from_slice(&buffer[..read]).unwrap();
        }
        let response = core::str::from_utf8(&response).unwrap();

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("gemma-4-26B-A4B-it"));
    }

    #[test]
    fn bearer_auth_is_exact_and_rejects_duplicates() {
        let token = "01234567890123456789012345678901";
        assert!(authorized(
            "GET / HTTP/1.1\r\nAuthorization: Bearer 01234567890123456789012345678901\r\n",
            token
        ));
        assert!(!authorized(
            "GET / HTTP/1.1\r\nAuthorization: Bearer wrong\r\n",
            token
        ));
        assert!(!authorized(
            "GET / HTTP/1.1\r\nAuthorization: Bearer 01234567890123456789012345678901\r\nAuthorization: Bearer 01234567890123456789012345678901\r\n",
            token
        ));
    }
}
