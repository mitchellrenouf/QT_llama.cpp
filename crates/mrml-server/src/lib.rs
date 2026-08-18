use mrml_core::client::{ChatCompletionRequest, ChatMessage, MrmlClient, StreamEvent};
use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

#[derive(Debug, Deserialize)]
pub struct OpenAiChatRequest {
    pub model: Option<String>,
    pub messages: Vec<OpenAiMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModelList {
    pub object: String,
    pub data: Vec<OpenAiModelItem>,
}

#[derive(Debug, Serialize)]
pub struct OpenAiModelItem {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

pub struct ApiServer {
    client: Arc<MrmlClient>,
    port: u16,
}

impl ApiServer {
    pub fn new(client: Arc<MrmlClient>, port: u16) -> Self {
        Self { client, port }
    }

    pub async fn run(&self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        println!("{}", format!("🌐 OpenAI-Compatible API Server listening on http://127.0.0.1:{}", self.port).bright_green().bold());
        println!("   - Endpoints: /v1/models, /v1/chat/completions (supports SSE stream: true)");

        loop {
            let (socket, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(e) => {
                    eprintln!("Socket accept error: {}", e);
                    continue;
                }
            };

            let client = self.client.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(socket, client).await {
                    eprintln!("HTTP Handler Error: {}", e);
                }
            });
        }
    }
}

async fn handle_connection(mut socket: TcpStream, client: Arc<MrmlClient>) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut total_read = 0;

    // Read HTTP headers
    let header_end_pos = loop {
        let n = socket.read(&mut buf[total_read..]).await?;
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
        socket.write_all(resp.as_bytes()).await?;
        return Ok(());
    }

    if method == "GET" && (path == "/v1/models" || path == "/models") {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let models = OpenAiModelList {
            object: "list".to_string(),
            data: vec![OpenAiModelItem {
                id: "gemma-4-26B-A4B-it".to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "mrml".to_string(),
            }],
        };
        let body = serde_json::to_string(&models)?;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(resp.as_bytes()).await?;
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
            let n = socket.read(&mut buf[total_read..]).await?;
            if n == 0 {
                break;
            }
            total_read += n;
        }

        let body_bytes = &buf[body_start..body_start + content_length];
        let chat_req: OpenAiChatRequest = match serde_json::from_slice(body_bytes) {
            Ok(r) => r,
            Err(e) => {
                let err_body = format!("{{\"error\":\"Invalid JSON: {}\"}}", e);
                let resp = format!(
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    err_body.len(),
                    err_body
                );
                socket.write_all(resp.as_bytes()).await?;
                return Ok(());
            }
        };

        let stream_mode = chat_req.stream.unwrap_or(false);
        let messages: Vec<ChatMessage> = chat_req.messages.into_iter().map(|m| ChatMessage {
            role: m.role,
            content: Some(mrml_model::MessageContent::Text(m.content)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        }).collect();

        let req = ChatCompletionRequest {
            model: chat_req.model.unwrap_or_else(|| "gemma-4-26B-A4B-it".to_string()),
            messages,
            tools: None,
            stream: Some(stream_mode),
            temperature: chat_req.temperature,
            max_tokens: chat_req.max_tokens,
            tool_choice: None,
        };

        if stream_mode {
            let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            socket.write_all(header.as_bytes()).await?;

            let (tx, mut rx) = mpsc::channel::<String>(128);
            let client_clone = client.clone();
            let req_clone = req.clone();

            tokio::spawn(async move {
                let _ = client_clone.stream_completion(&req_clone, |event| {
                    match event {
                        StreamEvent::Content(c) => {
                            let _ = tx.blocking_send(c);
                        }
                        StreamEvent::Reasoning(r) => {
                            let _ = tx.blocking_send(r);
                        }
                        _ => {}
                    }
                }).await;
            });

            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

            while let Some(piece) = rx.recv().await {
                let sse_obj = serde_json::json!({
                    "id": "chatcmpl-mrml",
                    "object": "chat.completion.chunk",
                    "created": now,
                    "model": "gemma-4-26B-A4B-it",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": piece
                        },
                        "finish_reason": null
                    }]
                });
                let sse_line = format!("data: {}\n\n", sse_obj);
                if socket.write_all(sse_line.as_bytes()).await.is_err() {
                    break;
                }
            }

            let done_obj = serde_json::json!({
                "id": "chatcmpl-mrml",
                "object": "chat.completion.chunk",
                "created": now,
                "model": "gemma-4-26B-A4B-it",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });
            let _ = socket.write_all(format!("data: {}\n\ndata: [DONE]\n\n", done_obj).as_bytes()).await;
            return Ok(());
        } else {
            let res = client.chat_completion(&req).await?;
            let _now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            let body = serde_json::to_string(&res)?;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await?;
            return Ok(());
        }
    }

    let not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    socket.write_all(not_found.as_bytes()).await?;
    Ok(())
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}
