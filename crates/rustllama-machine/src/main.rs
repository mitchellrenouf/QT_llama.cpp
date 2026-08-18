use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rustllama_core::client::StreamEvent;
use rustllama_core::{Config, GemmaAgent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const RECORD_PREFIX: &str = "RUSTLLAMA_MACHINE_JSON=";

/// Stable, non-interactive RustLlama interface for ChatGPT and test harnesses.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    #[command(flatten)]
    config: Config,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run one prompt and emit a single JSON result record.
    Chat {
        /// Prompt text. Use --stdin to read it from standard input instead.
        #[arg(long, conflicts_with = "stdin")]
        prompt: Option<String>,

        /// Read the complete prompt from standard input.
        #[arg(long)]
        stdin: bool,
    },

    /// Load the configured inference engine and report its health.
    Health,

    /// Process newline-delimited JSON requests while preserving conversation state.
    Session,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum SessionRequest {
    Chat { id: Option<Value>, prompt: String },
    Health { id: Option<Value> },
    Reset { id: Option<Value> },
    Exit { id: Option<Value> },
}

#[derive(Debug, Serialize)]
struct ToolEvent {
    name: String,
    result: String,
}

#[derive(Debug, Default, Serialize)]
struct TurnMetrics {
    token_count: Option<usize>,
    generation_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct TurnCapture {
    tools: Vec<ToolEvent>,
    finish_reason: Option<String>,
    token_count: Option<usize>,
    generation_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
}

fn emit(value: Value) -> Result<()> {
    println!("{RECORD_PREFIX}{}", serde_json::to_string(&value)?);
    Ok(())
}

async fn run_chat(agent: &mut GemmaAgent, prompt: &str, id: Option<Value>) -> Result<Value> {
    let capture = Arc::new(Mutex::new(TurnCapture::default()));
    let event_capture = Arc::clone(&capture);
    let started = Instant::now();
    let (content, reasoning) = agent
        .process_user_request_stream(prompt, move |event| {
            let mut state = event_capture.lock().expect("turn capture mutex poisoned");
            match event {
                StreamEvent::ToolExecuted { name, result } => {
                    state.tools.push(ToolEvent { name, result });
                }
                StreamEvent::Metrics {
                    token_count,
                    elapsed_secs,
                    tokens_per_sec,
                } => {
                    state.token_count = Some(token_count);
                    state.generation_seconds = Some(elapsed_secs);
                    state.tokens_per_second = Some(tokens_per_sec);
                }
                StreamEvent::Finish(reason) => state.finish_reason = Some(reason),
                StreamEvent::Reasoning(_)
                | StreamEvent::Content(_)
                | StreamEvent::ToolCallAssembled(_) => {}
            }
        })
        .await?;
    let mut state = capture.lock().expect("turn capture mutex poisoned");
    let metrics = TurnMetrics {
        token_count: state.token_count,
        generation_seconds: state.generation_seconds,
        tokens_per_second: state.tokens_per_second,
        wall_seconds: started.elapsed().as_secs_f64(),
    };

    Ok(json!({
        "schema_version": 1,
        "type": "chat_result",
        "id": id,
        "ok": true,
        "content": content,
        "reasoning": reasoning,
        "tool_events": std::mem::take(&mut state.tools),
        "finish_reason": state.finish_reason,
        "metrics": metrics,
    }))
}

async fn health(agent: &GemmaAgent, id: Option<Value>) -> Value {
    match agent.health_check().await {
        Ok(message) => json!({
            "schema_version": 1, "type": "health_result", "id": id,
            "ok": true, "message": message,
        }),
        Err(error) => json!({
            "schema_version": 1, "type": "health_result", "id": id,
            "ok": false, "error": error.to_string(),
        }),
    }
}

async fn run_session(mut agent: GemmaAgent) -> Result<()> {
    emit(json!({
        "schema_version": 1, "type": "ready", "ok": true,
        "protocol": "rustllama-machine-jsonl-v1",
    }))?;

    for line in io::stdin().lock().lines() {
        let line = line.context("failed reading session input")?;
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<SessionRequest>(&line) {
            Ok(request) => request,
            Err(error) => {
                emit(json!({
                    "schema_version": 1, "type": "error", "ok": false,
                    "error": "invalid_request", "message": error.to_string(),
                }))?;
                continue;
            }
        };

        let response = match request {
            SessionRequest::Chat { id, prompt } => run_chat(&mut agent, &prompt, id).await,
            SessionRequest::Health { id } => Ok(health(&agent, id).await),
            SessionRequest::Reset { id } => {
                agent.reset_context();
                Ok(json!({
                    "schema_version": 1, "type": "reset_result", "id": id, "ok": true,
                }))
            }
            SessionRequest::Exit { id } => {
                emit(json!({
                    "schema_version": 1, "type": "exit_result", "id": id, "ok": true,
                }))?;
                break;
            }
        };

        match response {
            Ok(value) => emit(value)?,
            Err(error) => emit(json!({
                "schema_version": 1, "type": "error", "ok": false,
                "error": "operation_failed", "message": error.to_string(),
            }))?,
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = Args::parse();
    args.config.cli = true;
    args.config.prompt = None;
    let mut agent = GemmaAgent::new(args.config);
    agent.init_mcp_servers().await?;

    match args.command {
        Command::Chat { prompt, stdin } => {
            let prompt = if stdin {
                io::read_to_string(io::stdin()).context("failed reading prompt from stdin")?
            } else {
                prompt.context("chat requires --prompt or --stdin")?
            };
            emit(run_chat(&mut agent, &prompt, None).await?)
        }
        Command::Health => emit(health(&agent, None).await),
        Command::Session => run_session(agent).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_session_chat_request() {
        let request: SessionRequest =
            serde_json::from_str(r#"{"op":"chat","id":7,"prompt":"hi"}"#).unwrap();
        assert!(matches!(request, SessionRequest::Chat { prompt, .. } if prompt == "hi"));
    }

    #[test]
    fn rejects_unknown_session_operation() {
        assert!(serde_json::from_str::<SessionRequest>(r#"{"op":"explode"}"#).is_err());
    }
}
