use anyhow::{Context, Result};
use mrml_core::client::StreamEvent;
use mrml_core::{Config, MrmlAgent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const RECORD_PREFIX: &str = "MRML_MACHINE_JSON=";

/// Stable, non-interactive MRML interface for ChatGPT and test harnesses.
#[derive(Debug)]
struct Args {
    config: Config,
    require_full_gpu: bool,
    gpu_load_retries: usize,
    command: Command,
}

impl Args {
    fn parse() -> Self {
        let arguments = std::env::args().collect::<Vec<_>>();
        if arguments.iter().any(|argument| argument == "--help" || argument == "-h") {
            println!("Usage: mrml-machine [OPTIONS] <chat|health|session>\n\n{}", Config::help());
            std::process::exit(0);
        }
        if arguments.iter().any(|argument| argument == "--version" || argument == "-V") {
            println!("{}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        match Self::try_parse_from(arguments) {
            Ok(args) => args,
            Err(error) => {
                eprintln!("error: {error}\n\nUsage: mrml-machine [OPTIONS] <chat|health|session>");
                std::process::exit(2);
            }
        }
    }

    fn try_parse_from<I, S>(arguments: I) -> std::result::Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let all = arguments.into_iter().map(Into::into).collect::<Vec<String>>();
        let program = all.first().cloned().unwrap_or_else(|| "mrml-machine".into());
        let command_index = all.iter().position(|arg| matches!(arg.as_str(), "chat" | "health" | "session"))
            .ok_or_else(|| "a chat, health, or session command is required".to_owned())?;
        let mut common = vec![program];
        let mut require_full_gpu = std::env::var("MRML_REQUIRE_FULL_GPU")
            .ok().map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")).unwrap_or(false);
        let mut gpu_load_retries = std::env::var("MRML_GPU_LOAD_RETRIES").ok().and_then(|value| value.parse().ok()).unwrap_or(0);
        let mut index = 1;
        while index < command_index {
            match all[index].as_str() {
                "--require-full-gpu" => require_full_gpu = true,
                "--gpu-load-retries" => {
                    index += 1;
                    gpu_load_retries = all.get(index).ok_or_else(|| "--gpu-load-retries requires a value".to_owned())?
                        .parse().map_err(|_| "invalid --gpu-load-retries value".to_owned())?;
                }
                value if value.starts_with("--gpu-load-retries=") => {
                    gpu_load_retries = value[19..].parse().map_err(|_| "invalid --gpu-load-retries value".to_owned())?;
                }
                _ => common.push(all[index].clone()),
            }
            index += 1;
        }
        let config = Config::try_parse_from(common)?;
        let tail = &all[command_index + 1..];
        let command = match all[command_index].as_str() {
            "health" if tail.is_empty() => Command::Health,
            "session" if tail.is_empty() => Command::Session,
            "chat" => {
                let mut prompt = None;
                let mut stdin = false;
                let mut index = 0;
                while index < tail.len() {
                    match tail[index].as_str() {
                        "--stdin" => stdin = true,
                        "--prompt" => {
                            index += 1;
                            prompt = Some(tail.get(index).ok_or_else(|| "--prompt requires a value".to_owned())?.clone());
                        }
                        value if value.starts_with("--prompt=") => prompt = Some(value[9..].to_owned()),
                        value => return Err(format!("unknown chat argument '{value}'")),
                    }
                    index += 1;
                }
                if stdin && prompt.is_some() { return Err("--stdin conflicts with --prompt".to_owned()); }
                Command::Chat { prompt, stdin }
            }
            command => return Err(format!("unexpected arguments after {command}")),
        };
        Ok(Self { config, require_full_gpu, gpu_load_retries, command })
    }
}

fn load_agent_with_residency_policy(args: &Args) -> Result<MrmlAgent> {
    let strict = args.require_full_gpu || args.gpu_load_retries > 0;
    for attempt in 0..=args.gpu_load_retries {
        let agent = MrmlAgent::new(args.config.clone());
        match agent.gpu_layer_residency() {
            Some((resident, total)) if resident == total => return Ok(agent),
            Some(_) if !strict => return Ok(agent),
            Some((resident, total)) => {
                let reason = format!("only {resident}/{total} transformer layers are fully GPU-resident");
                if attempt == args.gpu_load_retries {
                    anyhow::bail!("GPU residency requirement failed after {} load attempt(s): {reason}. Close other GPU applications or reduce context/cache memory, then retry", attempt + 1);
                }
                eprintln!("[mrml-machine] Load attempt {} rejected: {reason}; releasing the model and retrying", attempt + 1);
            }
            None if !strict => return Ok(agent),
            None => {
                if attempt == args.gpu_load_retries {
                    anyhow::bail!("GPU residency requirement failed after {} load attempt(s): no CUDA model engine was loaded. Verify --features cuda, --backend cuda, and the model path", attempt + 1);
                }
                eprintln!("[mrml-machine] Load attempt {} did not create a CUDA model engine; retrying", attempt + 1);
            }
        }
        drop(agent);
        #[cfg(feature = "cuda")]
        mrml_core::clear_cuda_allocation_pool();
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    unreachable!()
}

#[derive(Debug)]
enum Command {
    /// Run one prompt and emit a single JSON result record.
    Chat {
        /// Prompt text. Use --stdin to read it from standard input instead.
        prompt: Option<String>,

        /// Read the complete prompt from standard input.
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

async fn run_chat(agent: &mut MrmlAgent, prompt: &str, id: Option<Value>) -> Result<Value> {
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

async fn health(agent: &MrmlAgent, id: Option<Value>) -> Value {
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

async fn run_session(mut agent: MrmlAgent) -> Result<()> {
    emit(json!({
        "schema_version": 1, "type": "ready", "ok": true,
        "protocol": "mrml-machine-jsonl-v1",
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
    args.config.prompt = None;
    let mut agent = load_agent_with_residency_policy(&args)?;
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

    #[test]
    fn parses_strict_gpu_residency_options() {
        let args=Args::try_parse_from(["mrml-machine","--require-full-gpu","--gpu-load-retries","3","health"]).unwrap();
        assert!(args.require_full_gpu);
        assert_eq!(args.gpu_load_retries,3);
    }
}
