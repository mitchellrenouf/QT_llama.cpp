pub use crate::modes::{AgentMode, BackendChoice};
use core::fmt::Write as _;
use core::str::FromStr;
use mrml_runtime::{Text, Vector};
use std::path::{Path, PathBuf};

macro_rules! text_format {
    ($($argument:tt)*) => {{
        let mut output = Text::new();
        write!(output, $($argument)*).expect("MRML text allocation failed");
        output
    }};
}

#[derive(Debug, Clone)]
pub struct Config {
    pub server_url: Text,
    pub api_key: Text,
    pub model: Text,
    pub hf: Option<Text>,
    pub mode: AgentMode,
    pub workspace_root: PathBuf,
    pub temperature: f32,
    pub max_tokens: u32,
    pub ctx_size: u32,
    pub cache_type_k: Text,
    pub cache_type_v: Text,
    pub max_context_tokens: usize,
    pub auto_approve: bool,
    pub system_prompt: Option<Text>,
    pub prompt: Option<Text>,
    pub n_gpu_layers: Option<i32>,
    pub backend: BackendChoice,
    pub browser_exe: Option<Text>,
    pub browser_profile: Option<Text>,
    pub serve: bool,
    pub port: u16,
    pub mcp_servers: Vector<Text>,
}

impl Default for Config {
    fn default() -> Self {
        fn env(name: &str, fallback: &str) -> Text {
            mrml_runtime::environment_variable(name).unwrap_or_else(|| fallback.into())
        }
        Self {
            server_url: env("MRML_SERVER_URL", "http://localhost:8080/v1"),
            api_key: env("MRML_API_KEY", "mitchell"),
            model: env("MRML_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0"),
            hf: Some(env("HF_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0")),
            mode: AgentMode::General,
            workspace_root: PathBuf::from(env("WORKSPACE_ROOT", ".").as_str()),
            temperature: 0.7,
            max_tokens: 8192,
            ctx_size: mrml_runtime::environment_variable("MRML_CTX_SIZE")
                .and_then(|v| v.parse().ok())
                .unwrap_or(8192),
            cache_type_k: env("MRML_CACHE_TYPE_K", "auto"),
            cache_type_v: env("MRML_CACHE_TYPE_V", "auto"),
            max_context_tokens: 256_000,
            auto_approve: true,
            system_prompt: None,
            prompt: None,
            n_gpu_layers: mrml_runtime::environment_variable("MRML_GPU_LAYERS")
                .and_then(|v| v.parse().ok()),
            backend: mrml_runtime::environment_variable("MRML_BACKEND")
                .and_then(|v| v.parse().ok())
                .unwrap_or(BackendChoice::Auto),
            browser_exe: mrml_runtime::environment_variable("BROWSER_EXE"),
            browser_profile: mrml_runtime::environment_variable("BROWSER_PROFILE"),
            serve: false,
            port: 8080,
            mcp_servers: Vector::new(),
        }
    }
}

impl Config {
    pub fn parse() -> Self {
        let arguments = mrml_runtime::command_arguments();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!("{}", Self::help());
            crate::platform::exit_process(0);
        }
        if arguments
            .iter()
            .any(|argument| argument == "--version" || argument == "-V")
        {
            println!("{}", env!("CARGO_PKG_VERSION"));
            crate::platform::exit_process(0);
        }
        match Self::try_parse_from(arguments) {
            Ok(config) => config,
            Err(error) => {
                eprintln!("error: {error}\n\n{}", Self::help());
                crate::platform::exit_process(2);
            }
        }
    }

    pub fn try_parse_from<I, S>(arguments: I) -> Result<Self, Text>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut args = arguments
            .into_iter()
            .map(|argument| Text::from(argument.as_ref()));
        let _program = args.next();
        let mut config = Self::default();
        let mut args = args.peekable();
        while let Some(raw) = args.next() {
            if raw == "--help" || raw == "-h" {
                return Err(Self::help().into());
            }
            if raw == "--version" || raw == "-V" {
                return Err(env!("CARGO_PKG_VERSION").into());
            }
            let (name, inline) = raw
                .split_once('=')
                .map_or((raw.as_str(), None), |(name, value)| {
                    (name, Some(value.into()))
                });
            let mut value = || {
                inline
                    .clone()
                    .or_else(|| args.next())
                    .ok_or_else(|| text_format!("{name} requires a value"))
            };
            match name {
                "--server-url" => config.server_url = value()?.as_str().into(),
                "--api-key" => config.api_key = value()?.as_str().into(),
                "--model" => config.model = value()?.as_str().into(),
                "--hf" => config.hf = Some(value()?.as_str().into()),
                "--mode" => config.mode = parse_value("--mode", &value()?)?,
                "--workspace-root" => config.workspace_root = PathBuf::from(value()?.as_str()),
                "--temperature" => config.temperature = parse_value(name, &value()?)?,
                "--max-tokens" => config.max_tokens = parse_value(name, &value()?)?,
                "--ctx-size" => config.ctx_size = parse_value(name, &value()?)?,
                "--cache-type-k" => {
                    config.cache_type_k = parse_cache_type(name, value()?)?.as_str().into()
                }
                "--cache-type-v" => {
                    config.cache_type_v = parse_cache_type(name, value()?)?.as_str().into()
                }
                "--max-context-tokens" => config.max_context_tokens = parse_value(name, &value()?)?,
                "--auto-approve" => {
                    config.auto_approve = if let Some(value) = inline.as_deref() {
                        parse_bool(value)?
                    } else if let Some(candidate) =
                        args.peek().filter(|value| parse_bool(value).is_ok())
                    {
                        let parsed = parse_bool(candidate)?;
                        args.next();
                        parsed
                    } else {
                        true
                    };
                }
                "--no-auto-approve" => config.auto_approve = false,
                "--system-prompt" => config.system_prompt = Some(value()?.as_str().into()),
                "--prompt" | "-p" => config.prompt = Some(value()?.as_str().into()),
                "--gpu-layers" => config.n_gpu_layers = Some(parse_value(name, &value()?)?),
                "--backend" => config.backend = parse_value("--backend", &value()?)?,
                "--browser-exe" => config.browser_exe = Some(value()?.as_str().into()),
                "--browser-profile" => config.browser_profile = Some(value()?.as_str().into()),
                "--serve" => config.serve = true,
                "--port" => config.port = parse_value(name, &value()?)?,
                "--mcp-server" => config.mcp_servers.push(value()?.as_str().into()),
                _ => return Err(text_format!("unknown argument '{raw}'")),
            }
        }
        Ok(config)
    }

    pub const fn help() -> &'static str {
        "MRML local GGUF inference and multimodal agent interface\n\nOptions:\n  --model <PATH|SPEC>       Model path or repository specifier\n  --hf <SPEC>               Hugging Face model specifier\n  --mode <MODE>             general, coder, or automatic\n  --workspace-root <PATH>   Workspace used by tools\n  --temperature <FLOAT>     Generation temperature\n  --max-tokens <COUNT>      Maximum generated tokens\n  --ctx-size <COUNT>        KV-cache context size\n  --cache-type-k <TYPE>     Key cache type\n  --cache-type-v <TYPE>     Value cache type\n  --backend <BACKEND>       auto, cuda, rocm, vulkan, sycl, or cpu\n  --gpu-layers <COUNT>      GPU layer count\n  -p, --prompt <TEXT>       One-shot prompt\n  --port <PORT>             HTTP server port\n  --mcp-server <COMMAND>    Add an MCP server\n  -h, --help                Print help\n  -V, --version             Print version"
    }
}

fn parse_value<T: FromStr>(name: &str, value: &str) -> Result<T, Text> {
    value
        .parse()
        .map_err(|_| text_format!("invalid value '{value}' for {name}"))
}

fn parse_bool(value: &str) -> Result<bool, Text> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(text_format!("invalid boolean '{value}'")),
    }
}

fn parse_cache_type(name: &str, value: Text) -> Result<Text, Text> {
    const TYPES: &[&str] = &[
        "auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
    ];
    if TYPES.contains(&value.as_str()) {
        Ok(value)
    } else {
        Err(text_format!("invalid cache type '{value}' for {name}"))
    }
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parses_values_flags_aliases_and_repeated_options() {
        let config = Config::try_parse_from([
            "mrml",
            "--model",
            "model.gguf",
            "-p",
            "hello",
            "--ctx-size=4096",
            "--backend",
            "cuda",
            "--no-auto-approve",
            "--mcp-server",
            "one",
            "--mcp-server=two",
        ])
        .unwrap();
        assert_eq!(config.model, "model.gguf");
        assert_eq!(config.prompt.as_deref(), Some("hello"));
        assert_eq!(config.ctx_size, 4096);
        assert_eq!(config.backend, BackendChoice::Cuda);
        assert!(!config.auto_approve);
        assert_eq!(
            config
                .mcp_servers
                .iter()
                .map(Text::as_str)
                .collect::<std::vec::Vec<_>>(),
            ["one", "two"]
        );
    }

    #[test]
    fn rejects_unknown_options_and_invalid_cache_types() {
        assert!(Config::try_parse_from(["mrml", "--unknown"]).is_err());
        assert!(Config::try_parse_from(["mrml", "--cache-type-k", "bad"]).is_err());
    }
}

pub fn detect_os_name() -> Text {
    if cfg!(windows) {
        return "Windows".into();
    }
    if cfg!(target_os = "macos") {
        return "macOS".into();
    }
    if Path::new("/.flatpak-info").exists() {
        return "Flatpak Sandbox (Freedesktop SDK 26.08)".into();
    }
    if let Ok(content) = mrml_runtime::read_file_text("/etc/os-release") {
        for line in content.lines() {
            if let Some(pretty) = line.strip_prefix("PRETTY_NAME=") {
                return pretty.trim_matches('"').into();
            }
            if let Some(name) = line.strip_prefix("NAME=") {
                return name.trim_matches('"').into();
            }
        }
    }
    "Linux".into()
}

impl Config {
    pub fn get_system_prompt(&self, mode: AgentMode, rules_text: &str) -> Text {
        let abs_workspace = self
            .workspace_root
            .to_str()
            .and_then(|path| mrml_runtime::canonical_path(path).ok())
            .unwrap_or_else(|| self.workspace_root.to_string_lossy().as_ref().into());

        let current_date = crate::platform::local_date_string();
        let os_name = detect_os_name();

        let rules_section = if rules_text.trim().is_empty() {
            Text::new()
        } else {
            format!("\nPROJECT CUSTOM RULES:\n{}\n", rules_text)
                .as_str()
                .into()
        };

        match mode {
            AgentMode::General => format!(
                r#"You are Gemma, a highly capable, versatile AI Assistant with Multimodal Vision, Audio/Speech, Video, Desktop Control, and Web Automation capabilities, powered by Gemma 4 26B, running directly on {os_name}.
You act like Gemini in a browser—friendly, knowledgeable, concise, and capable of operating the user's computer via speech synthesis, audio recording, video capture, desktop screenshots, app opening, browser controls, and web tools.

CURRENT SYSTEM DATE: {current_date}
Workspace Root: "{abs_workspace}"
{rules_section}
DYNAMIC WEB FETCHING & SPEECH GUIDELINES:
- For the current local time or any other live system state, call `run_command` first. On Windows use `Get-Date`; on Linux or macOS use `date`. Never claim that you lack access when an available tool can answer the request.
- When a user asks you to "go to/on [website] and tell me/read/say the news/content":
  1. First execute `web_fetch` or `browser_open` on the specific URL (e.g., `https://apnews.com`).
  2. Extract the clean headlines and news text from the web tool output.
- ABSOLUTE SCREENSHOT RULE: Never call `browser_screenshot` or `take_screenshot` during web browsing, research, or e-commerce workflows. Screenshots of web pages are strictly prohibited. Rely exclusively on `web_fetch`, `browser_get_content`, `browser_open`, `browser_click_element`, and `browser_type`.
- MANDATORY E-COMMERCE WORKFLOW (Amazon, eBay, Walmart, Shopping Carts):
  1. Open search results using `browser_open` (e.g., `https://www.amazon.ca/s?k=mouse`).
  2. Extract product titles and prices using `browser_get_content`.
  3. Identify the target item (e.g. cheapest item).
  4. NAVIGATE TO PRODUCT PAGE: Click the item's title link using `browser_click_element` or `browser_open` its product URL. If `browser_click_element` fails in headless mode, call `browser_open(url=..., headless=false)` to open in GUI mode for UI Automation, or click "Add to Cart" directly.
  5. Once on the product detail page, call `browser_click_element(target_name="Add to Cart")` to successfully add the item to the cart.

Available Tools:
- `speak_text`, `record_audio`, `capture_webcam`, `record_screen_video`, `take_screenshot`, `open_app`, `browser_open`, `browser_get_content`, `browser_screenshot`, `browser_click_element`, `browser_click`, `browser_type`, `web_search`, `web_fetch`, `view_file`, `write_file`, `replace_file_content`, `list_dir`, `grep_search`, `run_command`, `git_checkpoint`, `git_rollback`, `git_diff`."#
            )
            .as_str()
            .into(),
            AgentMode::Coder => format!(
                r#"You are Gemma Vibe-Coder, an elite autonomous AI coding assistant with Multimodal Vision, Audio/Speech, Video, and Desktop Control capabilities, powered by Gemma 4 26B.
You work directly inside the user's workspace on {os_name}.

CURRENT SYSTEM DATE: {current_date}
Workspace Root: "{abs_workspace}"
{rules_section}
Available Tools:
1. `speak_text`, 2. `record_audio`, 3. `capture_webcam`, 4. `record_screen_video`, 5. `take_screenshot`, 6. `open_app`, 7. `browser_open`, 8. `browser_screenshot`, 9. `browser_click`, 10. `browser_type`, 11. `view_file`, 12. `write_file`, 13. `replace_file_content`, 14. `list_dir`, 15. `grep_search`, 16. `run_command`, 17. `git_checkpoint`, 18. `git_rollback`, 19. `git_diff`, 20. `web_search`, 21. `web_fetch`.

GUIDELINES FOR VIBE CODING:
- Always inspect existing code before editing to understand conventions and context.
- Keep edits precise and clean.
- Verify changes by running build or test commands (`run_command`) after making modifications.
- Create a `git_checkpoint` before making large structural refactors so changes can be reverted if needed."#
            )
            .as_str()
            .into(),
            AgentMode::Automatic => format!(
                r#"You are Gemma in AUTONOMOUS INNER MONOLOGUE MODE, powered by Gemma 4 26B.
In this mode, you maintain a continuous, human-like inner monologue before taking any action or giving an answer.

CURRENT SYSTEM DATE: {current_date}
Workspace Root: "{abs_workspace}"
{rules_section}
HUMAN-LIKE INNER MONOLOGUE INSTRUCTIONS:
- Express an ongoing, natural internal monologue of your thoughts, reflections, self-questions, and planning.
- When asked to visit a website (e.g. apnews.com), run `web_fetch` or `browser_open` first, read the extracted content, and if requested to speak/tell, invoke `speak_text`.

Available Tools:
- `speak_text`, `record_audio`, `capture_webcam`, `record_screen_video`, `take_screenshot`, `open_app`, `browser_open`, `browser_screenshot`, `browser_click`, `browser_type`, `web_search`, `web_fetch`, `view_file`, `write_file`, `replace_file_content`, `list_dir`, `grep_search`, `run_command`, `git_checkpoint`, `git_rollback`, `git_diff`."#
            )
            .as_str()
            .into(),
        }
    }
}
