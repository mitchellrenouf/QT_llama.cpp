use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Multimodal Vision, Audio/Video & Autonomous Inner Monologue Agent for Gemma 4 26B over llama-server OpenAI API", long_about = None)]
pub struct Config {
    /// Base URL of the llama-server OpenAI API endpoint
    #[arg(long, env = "LLAMA_SERVER_URL", default_value = "http://localhost:8080/v1")]
    pub server_url: String,

    /// API key for authentication with llama-server
    #[arg(long, env = "LLAMA_API_KEY", default_value = "mitchell")]
    pub api_key: String,

    /// Model name to pass to llama-server
    #[arg(long, env = "LLAMA_MODEL", default_value = "ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0")]
    pub model: String,

    /// Agent operating mode: 'general', 'coder', or 'automatic' (inner monologue mode)
    #[arg(long, value_enum, default_value_t = AgentMode::General)]
    pub mode: AgentMode,

    /// Workspace root directory for workspace tool operations
    #[arg(long, env = "WORKSPACE_ROOT", default_value = ".")]
    pub workspace_root: PathBuf,

    /// Generation temperature
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

    /// Maximum tokens per completion turn
    #[arg(long, default_value_t = 8192)]
    pub max_tokens: u32,

    /// Max context tokens before auto-compaction triggers (default 256k)
    #[arg(long, default_value_t = 256000)]
    pub max_context_tokens: usize,

    /// Auto-approve tool command executions without asking interactively
    #[arg(long, default_value_t = true)]
    pub auto_approve: bool,

    /// Custom system prompt override
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Run in terminal CLI mode instead of the default Qt6 GUI mode
    #[arg(long)]
    pub cli: bool,

    /// Custom path to browser executable (e.g. /var/lib/flatpak/exports/bin/com.brave.Browser, com.brave.Browser, or /usr/bin/brave-origin)
    #[arg(long, env = "BROWSER_EXE")]
    pub browser_exe: Option<String>,

    /// Custom browser user-data-dir path or profile directory (e.g. "Default", ~/.var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser, or ~/.config/BraveSoftware/Brave-Origin)
    #[arg(long, env = "BROWSER_PROFILE")]
    pub browser_profile: Option<String>,
}

pub fn detect_os_name() -> String {
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

        let current_date = chrono::Local::now().format("%A, %B %e, %Y").to_string();
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
