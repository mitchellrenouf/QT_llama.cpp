#[allow(dead_code)]
pub mod bridge;
mod app;

#[allow(unused_imports, dead_code)]
pub use bridge::*;

use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;

use crate::agent::GemmaAgent;
use crate::client::StreamEvent;
use crate::config::Config;

pub fn is_display_available() -> bool {
    if cfg!(windows) {
        true
    } else {
        std::env::var("WAYLAND_DISPLAY").is_ok() || std::env::var("DISPLAY").is_ok()
    }
}

pub async fn launch_qt_gui(config: &Config) -> Result<()> {
    if !is_display_available() {
        return Err(anyhow!("No graphical display server found (WAYLAND_DISPLAY or DISPLAY is unset)."));
    }

    // Bind local WebSocket server on ephemeral port
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let port_file = std::env::temp_dir().join("gemma_agent_port");
    let _ = std::fs::write(&port_file, port.to_string());

    let agent = Arc::new(Mutex::new(GemmaAgent::new(config.clone())));

    // Spawn WebSocket connection handler
    let agent_srv = agent.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let agent_conn = agent_srv.clone();
            tokio::spawn(async move {
                if let Ok(ws_stream) = tokio_tungstenite::accept_async(stream).await {
                    handle_ws_session(ws_stream, agent_conn).await;
                }
            });
        }
    });

    println!("🎨 Launching RustLlama native Qt6 interface [WS IPC port: {}]...", port);
    app::run(port)
}

async fn handle_ws_session<S>(ws_stream: tokio_tungstenite::WebSocketStream<S>, agent: Arc<Mutex<GemmaAgent>>)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (mut ws_tx, mut ws_rx) = ws_stream.split();
    let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();

    // Forward outbound queue to websocket
    tokio::spawn(async move {
        while let Some(msg_val) = out_rx.recv().await {
            let json_str = serde_json::to_string(&msg_val).unwrap_or_default();
            if ws_tx.send(Message::Text(json_str.into())).await.is_err() {
                break;
            }
        }
    });

    // Send initial status to the native GUI.
    {
        let agent_guard = agent.lock().await;
        let init_evt = serde_json::json!({
            "type": "init_state",
            "model_loaded": agent_guard.has_model_loaded(),
            "model_name": agent_guard.get_config().model,
            "mode": agent_guard.get_mode().to_string(),
            "backend": agent_guard.get_config().backend.to_string(),
            "speech_enabled": agent_guard.is_speech_enabled(),
            "tokens": agent_guard.estimate_tokens(),
        });
        let _ = out_tx.send(init_evt);
    }

    // Process inbound messages from the native GUI.
    while let Some(msg_res) = ws_rx.next().await {
        let msg = match msg_res {
            Ok(Message::Text(txt)) => txt,
            Ok(Message::Close(_)) => break,
            _ => continue,
        };

        let cmd: serde_json::Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let cmd_type = cmd.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match cmd_type {
            "send_message" => {
                if let Some(user_text) = cmd.get("message").and_then(|v| v.as_str()) {
                    let agent_clone = agent.clone();
                    let out_tx_clone = out_tx.clone();
                    let user_text_owned = user_text.to_string();

                    tokio::spawn(async move {
                        let mut agent_lock = agent_clone.lock().await;
                        let out_tx_stream = out_tx_clone.clone();

                        let res = agent_lock
                            .process_user_request_stream(&user_text_owned, move |event| {
                                match event {
                                    StreamEvent::Reasoning(thought) => {
                                        let _ = out_tx_stream.send(serde_json::json!({
                                            "type": "stream_thought",
                                            "thought": thought,
                                        }));
                                    }
                                    StreamEvent::Content(token) => {
                                        let _ = out_tx_stream.send(serde_json::json!({
                                            "type": "stream_token",
                                            "token": token,
                                        }));
                                    }
                                    StreamEvent::ToolCallAssembled(tc) => {
                                        let _ = out_tx_stream.send(serde_json::json!({
                                            "type": "tool_started",
                                            "name": tc.function.name,
                                            "args": tc.function.arguments,
                                        }));
                                    }
                                    StreamEvent::ToolExecuted { name, result } => {
                                        let _ = out_tx_stream.send(serde_json::json!({
                                            "type": "tool_finished",
                                            "name": name,
                                            "result": result,
                                        }));
                                    }
                                    StreamEvent::Metrics { token_count, elapsed_secs, tokens_per_sec } => {
                                        let _ = out_tx_stream.send(serde_json::json!({
                                            "type": "metrics",
                                            "token_count": token_count,
                                            "elapsed_secs": elapsed_secs,
                                            "tokens_per_sec": tokens_per_sec,
                                        }));
                                    }
                                    StreamEvent::Finish(_) => {}
                                }
                            })
                            .await;

                        match res {
                            Ok((content, thought)) => {
                                let total_tokens = agent_lock.estimate_tokens();
                                let _ = out_tx_clone.send(serde_json::json!({
                                    "type": "turn_done",
                                    "content": content,
                                    "thought": thought,
                                    "tokens": total_tokens,
                                }));
                            }
                            Err(e) => {
                                let _ = out_tx_clone.send(serde_json::json!({
                                    "type": "error",
                                    "message": e.to_string(),
                                }));
                            }
                        }
                    });
                }
            }
            "load_hf_model" => {
                if let Some(spec_str) = cmd.get("spec").and_then(|v| v.as_str()) {
                    let agent_clone = agent.clone();
                    let out_tx_clone = out_tx.clone();
                    let spec_str_owned = spec_str.to_string();

                    tokio::spawn(async move {
                        match crate::hf::HfModelSpec::parse(&spec_str_owned) {
                            Ok(spec) => {
                                let out_tx_prog = out_tx_clone.clone();
                                let fetch_res = crate::hf::resolve_or_fetch_hf_model(&spec, move |msg, p, file_idx, total_files| {
                                    let _ = out_tx_prog.send(serde_json::json!({
                                        "type": "download_progress",
                                        "message": msg,
                                        "progress": p,
                                        "file_idx": file_idx,
                                        "total_files": total_files,
                                    }));
                                }).await;

                                match fetch_res {
                                    Ok(model_files) => {
                                        let mut agent_lock = agent_clone.lock().await;
                                        if let Err(e) = agent_lock.reload_model(&model_files.primary_entry_file) {
                                            let _ = out_tx_clone.send(serde_json::json!({
                                                "type": "error",
                                                "message": format!("Failed to load GGUF model: {}", e),
                                            }));
                                        } else {
                                            let _ = out_tx_clone.send(serde_json::json!({
                                                "type": "model_loaded",
                                                "model_name": spec.repo_id,
                                            }));
                                        }
                                    }
                                    Err(e) => {
                                        let _ = out_tx_clone.send(serde_json::json!({
                                            "type": "error",
                                            "message": format!("Download error: {}", e),
                                        }));
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = out_tx_clone.send(serde_json::json!({
                                    "type": "error",
                                    "message": format!("Invalid model specification: {}", e),
                                }));
                            }
                        }
                    });
                }
            }
            "switch_mode" => {
                if let Some(mode_str) = cmd.get("mode").and_then(|v| v.as_str()) {
                    let mut agent_lock = agent.lock().await;
                    match mode_str {
                        "code" | "coder" => agent_lock.set_mode(crate::config::AgentMode::Coder),
                        "auto" | "automatic" => agent_lock.set_mode(crate::config::AgentMode::Automatic),
                        _ => agent_lock.set_mode(crate::config::AgentMode::General),
                    }
                    let _ = out_tx.send(serde_json::json!({
                        "type": "mode_changed",
                        "mode": agent_lock.get_mode().to_string(),
                    }));
                }
            }
            "switch_backend" => {
                if let Some(backend_str) = cmd.get("backend").and_then(|v| v.as_str()) {
                    let agent_clone = agent.clone();
                    let out_tx_clone = out_tx.clone();
                    let backend_choice = match backend_str.to_lowercase().as_str() {
                        "cuda" => crate::config::BackendChoice::Cuda,
                        "vulkan" => crate::config::BackendChoice::Vulkan,
                        "sycl" => crate::config::BackendChoice::Sycl,
                        "cpu" => crate::config::BackendChoice::Cpu,
                        _ => crate::config::BackendChoice::Auto,
                    };
                    tokio::spawn(async move {
                        let mut agent_lock = agent_clone.lock().await;
                        if let Err(e) = agent_lock.switch_backend(backend_choice) {
                            let _ = out_tx_clone.send(serde_json::json!({
                                "type": "error",
                                "message": format!("Failed to switch backend: {}", e),
                            }));
                        } else {
                            let _ = out_tx_clone.send(serde_json::json!({
                                "type": "backend_changed",
                                "backend": agent_lock.get_config().backend.to_string(),
                            }));
                        }
                    });
                }
            }
            "clear_history" => {
                let mut agent_lock = agent.lock().await;
                agent_lock.reset_context();
                let _ = out_tx.send(serde_json::json!({
                    "type": "history_cleared",
                    "tokens": agent_lock.estimate_tokens(),
                }));
            }
            "toggle_speech" => {
                let mut agent_lock = agent.lock().await;
                let enabled = agent_lock.toggle_speech();
                let _ = out_tx.send(serde_json::json!({
                    "type": "speech_toggled",
                    "enabled": enabled,
                }));
            }
            _ => {}
        }
    }
}
