use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    General,
    Coder,
    Automatic,
}

impl fmt::Display for AgentMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentMode::General => write!(f, "general"),
            AgentMode::Coder => write!(f, "coder"),
            AgentMode::Automatic => write!(f, "automatic"),
        }
    }
}

impl FromStr for AgentMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "general" => Ok(Self::General),
            "coder" => Ok(Self::Coder),
            "automatic" => Ok(Self::Automatic),
            _ => Err(format!("invalid agent mode '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Cuda,
    Rocm,
    Vulkan,
    Sycl,
    Cpu,
}

impl fmt::Display for BackendChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendChoice::Auto => write!(f, "auto"),
            BackendChoice::Cuda => write!(f, "cuda"),
            BackendChoice::Rocm => write!(f, "rocm"),
            BackendChoice::Vulkan => write!(f, "vulkan"),
            BackendChoice::Sycl => write!(f, "sycl"),
            BackendChoice::Cpu => write!(f, "cpu"),
        }
    }
}

impl FromStr for BackendChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "rocm" => Ok(Self::Rocm),
            "vulkan" => Ok(Self::Vulkan),
            "sycl" => Ok(Self::Sycl),
            "cpu" => Ok(Self::Cpu),
            _ => Err(format!("invalid backend '{value}'")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub server_url: String,
    pub api_key: String,
    pub model: String,
    pub hf: Option<String>,
    pub mode: AgentMode,
    pub workspace_root: PathBuf,
    pub temperature: f32,
    pub max_tokens: u32,
    pub ctx_size: u32,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub max_context_tokens: usize,
    pub auto_approve: bool,
    pub system_prompt: Option<String>,
    pub prompt: Option<String>,
    pub n_gpu_layers: Option<i32>,
    pub backend: BackendChoice,
    pub browser_exe: Option<String>,
    pub browser_profile: Option<String>,
    pub serve: bool,
    pub port: u16,
    pub mcp_servers: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        fn env(name: &str, fallback: &str) -> String {
            std::env::var(name).unwrap_or_else(|_| fallback.to_owned())
        }
        Self {
            server_url: env("MRML_SERVER_URL", "http://localhost:8080/v1"),
            api_key: env("MRML_API_KEY", "mitchell"),
            model: env("MRML_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0"),
            hf: Some(env("HF_MODEL", "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0")),
            mode: AgentMode::General,
            workspace_root: PathBuf::from(env("WORKSPACE_ROOT", ".")),
            temperature: 0.7,
            max_tokens: 8192,
            ctx_size: std::env::var("MRML_CTX_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(8192),
            cache_type_k: env("MRML_CACHE_TYPE_K", "auto"),
            cache_type_v: env("MRML_CACHE_TYPE_V", "auto"),
            max_context_tokens: 256_000,
            auto_approve: true,
            system_prompt: None,
            prompt: None,
            n_gpu_layers: std::env::var("MRML_GPU_LAYERS").ok().and_then(|v| v.parse().ok()),
            backend: std::env::var("MRML_BACKEND").ok().and_then(|v| v.parse().ok()).unwrap_or(BackendChoice::Auto),
            browser_exe: std::env::var("BROWSER_EXE").ok(),
            browser_profile: std::env::var("BROWSER_PROFILE").ok(),
            serve: false,
            port: 8080,
            mcp_servers: Vec::new(),
        }
    }
}

impl Config {
    pub fn parse() -> Self {
        let arguments = std::env::args().collect::<Vec<_>>();
        if arguments.iter().any(|argument| argument == "--help" || argument == "-h") {
            println!("{}", Self::help());
            std::process::exit(0);
        }
        if arguments.iter().any(|argument| argument == "--version" || argument == "-V") {
            println!("{}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        match Self::try_parse_from(arguments) {
            Ok(config) => config,
            Err(error) => {
                eprintln!("error: {error}\n\n{}", Self::help());
                std::process::exit(2);
            }
        }
    }

    pub fn try_parse_from<I, S>(arguments: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args = arguments.into_iter().map(Into::into);
        let _program = args.next();
        let mut config = Self::default();
        let mut args = args.peekable();
        while let Some(raw) = args.next() {
            if raw == "--help" || raw == "-h" {
                return Err(Self::help().to_owned());
            }
            if raw == "--version" || raw == "-V" {
                return Err(env!("CARGO_PKG_VERSION").to_owned());
            }
            let (name, inline) = raw.split_once('=').map_or((raw.as_str(), None), |(name, value)| (name, Some(value.to_owned())));
            let mut value = || inline.clone().or_else(|| args.next()).ok_or_else(|| format!("{name} requires a value"));
            match name {
                "--server-url" => config.server_url = value()?,
                "--api-key" => config.api_key = value()?,
                "--model" => config.model = value()?,
                "--hf" => config.hf = Some(value()?),
                "--mode" => config.mode = value()?.parse()?,
                "--workspace-root" => config.workspace_root = value()?.into(),
                "--temperature" => config.temperature = parse_value(name, &value()?)?,
                "--max-tokens" => config.max_tokens = parse_value(name, &value()?)?,
                "--ctx-size" => config.ctx_size = parse_value(name, &value()?)?,
                "--cache-type-k" => config.cache_type_k = parse_cache_type(name, value()?)?,
                "--cache-type-v" => config.cache_type_v = parse_cache_type(name, value()?)?,
                "--max-context-tokens" => config.max_context_tokens = parse_value(name, &value()?)?,
                "--auto-approve" => {
                    config.auto_approve = if let Some(value) = inline.as_deref() {
                        parse_bool(value)?
                    } else if let Some(candidate) = args.peek().filter(|value| parse_bool(value).is_ok()) {
                        let parsed = parse_bool(candidate)?;
                        args.next();
                        parsed
                    } else {
                        true
                    };
                }
                "--no-auto-approve" => config.auto_approve = false,
                "--system-prompt" => config.system_prompt = Some(value()?),
                "--prompt" | "-p" => config.prompt = Some(value()?),
                "--gpu-layers" => config.n_gpu_layers = Some(parse_value(name, &value()?)?),
                "--backend" => config.backend = value()?.parse()?,
                "--browser-exe" => config.browser_exe = Some(value()?),
                "--browser-profile" => config.browser_profile = Some(value()?),
                "--serve" => config.serve = true,
                "--port" => config.port = parse_value(name, &value()?)?,
                "--mcp-server" => config.mcp_servers.push(value()?),
                _ => return Err(format!("unknown argument '{raw}'")),
            }
        }
        Ok(config)
    }

    pub const fn help() -> &'static str {
        "MRML local GGUF inference and multimodal agent interface\n\nOptions:\n  --model <PATH|SPEC>       Model path or repository specifier\n  --hf <SPEC>               Hugging Face model specifier\n  --mode <MODE>             general, coder, or automatic\n  --workspace-root <PATH>   Workspace used by tools\n  --temperature <FLOAT>     Generation temperature\n  --max-tokens <COUNT>      Maximum generated tokens\n  --ctx-size <COUNT>        KV-cache context size\n  --cache-type-k <TYPE>     Key cache type\n  --cache-type-v <TYPE>     Value cache type\n  --backend <BACKEND>       auto, cuda, rocm, vulkan, sycl, or cpu\n  --gpu-layers <COUNT>      GPU layer count\n  -p, --prompt <TEXT>       One-shot prompt\n  --port <PORT>             HTTP server port\n  --mcp-server <COMMAND>    Add an MCP server\n  -h, --help                Print help\n  -V, --version             Print version"
    }
}

fn parse_value<T: FromStr>(name: &str, value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("invalid value '{value}' for {name}"))
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid boolean '{value}'")),
    }
}

fn parse_cache_type(name: &str, value: String) -> Result<String, String> {
    const TYPES: &[&str] = &["auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];
    if TYPES.contains(&value.as_str()) { Ok(value) } else { Err(format!("invalid cache type '{value}' for {name}")) }
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn parses_values_flags_aliases_and_repeated_options() {
        let config = Config::try_parse_from([
            "mrml", "--model", "model.gguf", "-p", "hello", "--ctx-size=4096",
            "--backend", "cuda", "--no-auto-approve", "--mcp-server", "one",
            "--mcp-server=two",
        ]).unwrap();
        assert_eq!(config.model, "model.gguf");
        assert_eq!(config.prompt.as_deref(), Some("hello"));
        assert_eq!(config.ctx_size, 4096);
        assert_eq!(config.backend, BackendChoice::Cuda);
        assert!(!config.auto_approve);
        assert_eq!(config.mcp_servers, ["one", "two"]);
    }

    #[test]
    fn rejects_unknown_options_and_invalid_cache_types() {
        assert!(Config::try_parse_from(["mrml", "--unknown"]).is_err());
        assert!(Config::try_parse_from(["mrml", "--cache-type-k", "bad"]).is_err());
    }
}

pub fn detect_os_name() -> String {
    if cfg!(windows) {
        return "Windows".to_string();
    }
    if cfg!(target_os = "macos") {
        return "macOS".to_string();
    }
    if Path::new("/.flatpak-info").exists() {
        return "Flatpak Sandbox (Freedesktop SDK 26.08)".to_string();
    }
    if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if let Some(pretty) = line.strip_prefix("PRETTY_NAME=") {
                return pretty.trim_matches('"').to_string();
            }
            if let Some(name) = line.strip_prefix("NAME=") {
                return name.trim_matches('"').to_string();
            }
        }
    }
    "Linux".to_string()
}

impl Config {
    pub fn get_system_prompt(&self, mode: AgentMode, rules_text: &str) -> String {
        let abs_workspace = std::fs::canonicalize(&self.workspace_root)
            .unwrap_or_else(|_| self.workspace_root.clone())
            .display()
            .to_string();

        let current_date = crate::platform::local_date_string();
        let os_name = detect_os_name();

        let rules_section = if rules_text.trim().is_empty() {
            String::new()
        } else {
            format!("\nPROJECT CUSTOM RULES:\n{}\n", rules_text)
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
            ),
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
            ),
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
            ),
        }
    }
}
