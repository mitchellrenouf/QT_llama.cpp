use anyhow::{Context, Result};
use core::fmt::Write as _;
use mrml_core::client::StreamEvent;
use mrml_core::{Config, MrmlAgent};
use mrml_json::{Value, object};
use mrml_runtime::{Shared, SpinMutex, Text, Vector, mrml_eprintln as eprintln, mrml_println as println};

const RECORD_PREFIX: &str = "MRML_MACHINE_JSON=";

macro_rules! text_format {
    ($($argument:tt)*) => {{
        let mut output = Text::new();
        write!(output, $($argument)*).expect("MRML text allocation failed");
        output
    }};
}

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
        let arguments = mrml_runtime::command_arguments();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: mrml-machine [OPTIONS] <chat|health|session>\n\n{}",
                Config::help()
            );
            mrml_core::tools::platform::exit_process(0);
        }
        if arguments
            .iter()
            .any(|argument| argument == "--version" || argument == "-V")
        {
            println!("{}", env!("CARGO_PKG_VERSION"));
            mrml_core::tools::platform::exit_process(0);
        }
        match Self::try_parse_from(arguments) {
            Ok(args) => args,
            Err(error) => {
                eprintln!("error: {error}\n\nUsage: mrml-machine [OPTIONS] <chat|health|session>");
                mrml_core::tools::platform::exit_process(2);
            }
        }
    }

    fn try_parse_from<I, S>(arguments: I) -> std::result::Result<Self, Text>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let all = arguments
            .into_iter()
            .map(|argument| Text::from(argument.as_ref()))
            .collect::<Vector<Text>>();
        let program = all
            .first()
            .cloned()
            .unwrap_or_else(|| "mrml-machine".into());
        let command_index = all
            .iter()
            .position(|arg| matches!(arg.as_str(), "chat" | "health" | "session"))
            .ok_or_else(|| Text::from("a chat, health, or session command is required"))?;
        let mut common = Vector::from([program]);
        let mut require_full_gpu = mrml_runtime::environment_variable("MRML_REQUIRE_FULL_GPU")
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        let mut gpu_load_retries = mrml_runtime::environment_variable("MRML_GPU_LOAD_RETRIES")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let mut index = 1;
        while index < command_index {
            match all[index].as_str() {
                "--require-full-gpu" => require_full_gpu = true,
                "--gpu-load-retries" => {
                    index += 1;
                    gpu_load_retries = all
                        .get(index)
                        .ok_or_else(|| Text::from("--gpu-load-retries requires a value"))?
                        .parse()
                        .map_err(|_| Text::from("invalid --gpu-load-retries value"))?;
                }
                value if value.starts_with("--gpu-load-retries=") => {
                    gpu_load_retries = value[19..]
                        .parse()
                        .map_err(|_| Text::from("invalid --gpu-load-retries value"))?;
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
                            prompt = Some(
                                tail.get(index)
                                    .ok_or_else(|| Text::from("--prompt requires a value"))?
                                    .as_str()
                                    .into(),
                            );
                        }
                        value if value.starts_with("--prompt=") => prompt = Some(value[9..].into()),
                        value => return Err(text_format!("unknown chat argument '{value}'")),
                    }
                    index += 1;
                }
                if stdin && prompt.is_some() {
                    return Err("--stdin conflicts with --prompt".into());
                }
                Command::Chat { prompt, stdin }
            }
            command => return Err(text_format!("unexpected arguments after {command}")),
        };
        Ok(Self {
            config,
            require_full_gpu,
            gpu_load_retries,
            command,
        })
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
                let reason =
                    format!("only {resident}/{total} transformer layers are fully GPU-resident");
                if attempt == args.gpu_load_retries {
                    anyhow::bail!(
                        "GPU residency requirement failed after {} load attempt(s): {reason}. Close other GPU applications or reduce context/cache memory, then retry",
                        attempt + 1
                    );
                }
                eprintln!(
                    "[mrml-machine] Load attempt {} rejected: {reason}; releasing the model and retrying",
                    attempt + 1
                );
            }
            None if !strict => return Ok(agent),
            None => {
                if attempt == args.gpu_load_retries {
                    anyhow::bail!(
                        "GPU residency requirement failed after {} load attempt(s): no CUDA model engine was loaded. Verify --features cuda, --backend cuda, and the model path",
                        attempt + 1
                    );
                }
                eprintln!(
                    "[mrml-machine] Load attempt {} did not create a CUDA model engine; retrying",
                    attempt + 1
                );
            }
        }
        drop(agent);
        #[cfg(feature = "cuda")]
        mrml_core::clear_cuda_allocation_pool();
        mrml_core::tools::platform::sleep_millis(500);
    }
    unreachable!()
}

#[derive(Debug)]
enum Command {
    /// Run one prompt and emit a single JSON result record.
    Chat {
        /// Prompt text. Use --stdin to read it from standard input instead.
        prompt: Option<Text>,

        /// Read the complete prompt from standard input.
        stdin: bool,
    },

    /// Load the configured inference engine and report its health.
    Health,

    /// Process newline-delimited JSON requests while preserving conversation state.
    Session,
}

#[derive(Debug)]
enum SessionRequest {
    Chat { id: Option<Value>, prompt: Text },
    Health { id: Option<Value> },
    Reset { id: Option<Value> },
    Exit { id: Option<Value> },
}

impl SessionRequest {
    fn parse(source: &str) -> std::result::Result<Self, mrml_json::Error> {
        let value = mrml_json::parse(source)?;
        let operation = value.get("op").and_then(Value::as_str).unwrap_or("");
        let id = value.get("id").cloned().filter(|value| !value.is_null());
        Ok(match operation {
            "chat" => Self::Chat {
                id,
                prompt: value
                    .get("prompt")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        mrml_json::Error::message("chat request requires a string prompt")
                    })?
                    .into(),
            },
            "health" => Self::Health { id },
            "reset" => Self::Reset { id },
            "exit" => Self::Exit { id },
            _ => {
                return Err(mrml_json::Error::message(format!(
                    "unknown session operation '{operation}'"
                )));
            }
        })
    }
}

#[derive(Debug)]
struct ToolEvent {
    name: Text,
    result: Text,
}

#[derive(Debug, Default)]
struct TurnMetrics {
    token_count: Option<usize>,
    generation_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct TurnCapture {
    tools: Vector<ToolEvent>,
    finish_reason: Option<Text>,
    token_count: Option<usize>,
    generation_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
}

fn emit(value: Value) -> Result<()> {
    println!("{RECORD_PREFIX}{}", mrml_json::stringify(&value));
    Ok(())
}

async fn run_chat(agent: &mut MrmlAgent, prompt: &str, id: Option<Value>) -> Result<Value> {
    let capture = Shared::new(SpinMutex::new(TurnCapture::default()));
    let event_capture = capture.clone();
    let started = mrml_core::tools::platform::monotonic_timestamp_nanos();
    let (content, reasoning) = agent
        .process_user_request_stream(prompt, move |event| {
            let mut state = event_capture.lock();
            match event {
                StreamEvent::ToolExecuted { name, result } => {
                    state.tools.push(ToolEvent {
                        name: name.as_str().into(),
                        result: result.as_str().into(),
                    });
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
                StreamEvent::Finish(reason) => state.finish_reason = Some(reason.as_str().into()),
                StreamEvent::Reasoning(_)
                | StreamEvent::Content(_)
                | StreamEvent::ToolCallAssembled(_) => {}
            }
        })
        .await?;
    let mut state = capture.lock();
    let metrics = TurnMetrics {
        token_count: state.token_count,
        generation_seconds: state.generation_seconds,
        tokens_per_second: state.tokens_per_second,
        wall_seconds: mrml_core::tools::platform::monotonic_timestamp_nanos()
            .saturating_sub(started) as f64
            / 1_000_000_000.0,
    };

    let tools = core::mem::take(&mut state.tools)
        .into_iter()
        .map(|tool| {
            object([
                ("name", Value::text(tool.name)),
                ("result", Value::text(tool.result)),
            ])
        })
        .collect();
    Ok(object([
        ("schema_version", 1usize.into()),
        ("type", "chat_result".into()),
        ("id", id.into()),
        ("ok", true.into()),
        ("content", Value::text(content)),
        ("reasoning", Value::text(reasoning)),
        ("tool_events", Value::Array(tools)),
        (
            "finish_reason",
            Value::optional_text(state.finish_reason.clone()),
        ),
        (
            "metrics",
            object([
                ("token_count", metrics.token_count.into()),
                ("generation_seconds", metrics.generation_seconds.into()),
                ("tokens_per_second", metrics.tokens_per_second.into()),
                ("wall_seconds", metrics.wall_seconds.into()),
            ]),
        ),
    ]))
}

async fn health(agent: &MrmlAgent, id: Option<Value>) -> Value {
    match agent.health_check().await {
        Ok(message) => object([
            ("schema_version", 1usize.into()),
            ("type", "health_result".into()),
            ("id", id.into()),
            ("ok", true.into()),
            ("message", Value::text(message)),
        ]),
        Err(error) => object([
            ("schema_version", 1usize.into()),
            ("type", "health_result".into()),
            ("id", id.into()),
            ("ok", false.into()),
            ("error", Value::text(error.to_string())),
        ]),
    }
}

async fn run_session(mut agent: MrmlAgent) -> Result<()> {
    emit(object([
        ("schema_version", 1usize.into()),
        ("type", "ready".into()),
        ("ok", true.into()),
        ("protocol", "mrml-machine-jsonl-v1".into()),
    ]))?;

    while let Some(line) = mrml_runtime::read_stdin_line()
        .map_err(|_| anyhow::anyhow!("failed reading session input"))?
    {
        if line.trim().is_empty() {
            continue;
        }
        let request = match SessionRequest::parse(&line) {
            Ok(request) => request,
            Err(error) => {
                emit(object([
                    ("schema_version", 1usize.into()),
                    ("type", "error".into()),
                    ("ok", false.into()),
                    ("error", "invalid_request".into()),
                    ("message", Value::text(error.to_string())),
                ]))?;
                continue;
            }
        };

        let response = match request {
            SessionRequest::Chat { id, prompt } => run_chat(&mut agent, &prompt, id).await,
            SessionRequest::Health { id } => Ok(health(&agent, id).await),
            SessionRequest::Reset { id } => {
                agent.reset_context();
                Ok(object([
                    ("schema_version", 1usize.into()),
                    ("type", "reset_result".into()),
                    ("id", id.into()),
                    ("ok", true.into()),
                ]))
            }
            SessionRequest::Exit { id } => {
                emit(object([
                    ("schema_version", 1usize.into()),
                    ("type", "exit_result".into()),
                    ("id", id.into()),
                    ("ok", true.into()),
                ]))?;
                break;
            }
        };

        match response {
            Ok(value) => emit(value)?,
            Err(error) => emit(object([
                ("schema_version", 1usize.into()),
                ("type", "error".into()),
                ("ok", false.into()),
                ("error", "operation_failed".into()),
                ("message", Value::text(error.to_string())),
            ]))?,
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    mrml_tools::block_on(async_main())
}

async fn async_main() -> Result<()> {
    let mut args = Args::parse();
    args.config.prompt = None;
    let mut agent = load_agent_with_residency_policy(&args)?;
    agent.init_mcp_servers().await?;

    match args.command {
        Command::Chat { prompt, stdin } => {
            let prompt = if stdin {
                mrml_runtime::read_stdin_to_end()
                    .map_err(|_| anyhow::anyhow!("failed reading prompt from stdin"))?
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
        let request = SessionRequest::parse(r#"{"op":"chat","id":7,"prompt":"hi"}"#).unwrap();
        assert!(matches!(request, SessionRequest::Chat { prompt, .. } if prompt == "hi"));
    }

    #[test]
    fn rejects_unknown_session_operation() {
        assert!(SessionRequest::parse(r#"{"op":"explode"}"#).is_err());
        assert!(SessionRequest::parse(r#"{"op":"chat"}"#).is_err());
    }

    #[test]
    fn local_json_preserves_machine_protocol_values_and_escaping() {
        let value = object([
            ("schema_version", 1usize.into()),
            ("type", "chat_result".into()),
            ("content", "line one\n\"line two\"".into()),
            ("ok", true.into()),
        ]);
        let encoded = mrml_json::stringify(&value);
        assert_eq!(mrml_json::parse(&encoded).unwrap(), value);
        assert!(encoded.contains(r#"line one\n\"line two\""#));
    }

    #[test]
    fn parses_strict_gpu_residency_options() {
        let args = Args::try_parse_from([
            "mrml-machine",
            "--require-full-gpu",
            "--gpu-load-retries",
            "3",
            "health",
        ])
        .unwrap();
        assert!(args.require_full_gpu);
        assert_eq!(args.gpu_load_retries, 3);
    }
}
