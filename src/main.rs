mod agent;
mod client;
mod config;
mod diff;
mod gui;
pub mod hf;
mod markdown;
mod rules;
mod tools;

use agent::GemmaAgent;
use clap::Parser;
use config::{AgentMode, Config};
use colored::*;
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::parse();

    if !config.cli {
        match gui::launch_qt_gui(&config).await {
            Ok(_) => return Ok(()),
            Err(e) => {
                println!(
                    "{} (Falling back to CLI mode. Use '--cli' to suppress this warning.)",
                    format!("ℹ Qt6 GUI launch unavailable: {}", e).yellow()
                );
            }
        }
    }

    println!("{}", "==================================================".magenta());
    println!("{}", "   🚀 QT_llama.cpp GENERAL-PURPOSE & VIBE-CODING CLI   ".bright_cyan().bold());
    println!("{}", "==================================================".magenta());
    println!(" Mode        : {}", config.mode.to_string().bright_yellow().bold());
    println!(" Inference   : {}", "In-Process llama.cpp GGUF Engine".bright_green().bold());
    if let Some(hf_spec) = &config.hf {
        println!(" HuggingFace : {}", hf_spec.bright_cyan().bold());
    }
    println!(" Model Path  : {}", config.model.cyan());
    println!(" Workspace   : {}", config.workspace_root.display().to_string().green());
    println!(" Max Context : {} tokens (auto-compact enabled)", config.max_context_tokens.to_string().yellow());
    println!(" Auto-Approve: {}", config.auto_approve.to_string().bright_white());

    let mut agent = GemmaAgent::new(config);

    if agent.get_rules().has_rules() {
        println!(" Loaded Rules: {}", agent.loaded_rules_count().to_string().bright_green().bold());
        for src in &agent.get_rules().rule_sources {
            println!("   - {}", src.display().to_string().dimmed());
        }
    }

    println!("{}", "--------------------------------------------------".dimmed());

    let stdin = io::stdin();
    let mut handle = stdin.lock();

    if !agent.has_model_loaded() {
        if let Some(hf_spec) = config.hf.clone() {
            println!("{} Model weights for '{}' are not cached locally.", "📥".cyan(), hf_spec.bright_cyan().bold());
            print!("{} Download & load model weights now? [Y/n]: ", "❓".yellow());
            io::stdout().flush()?;
            let mut ans = String::new();
            if handle.read_line(&mut ans).is_ok() {
                let trimmed = ans.trim().to_lowercase();
                if trimmed.is_empty() || trimmed == "y" || trimmed == "yes" {
                    println!("\nFetching and downloading Hugging Face model: {}...", hf_spec.cyan());
                    match agent.load_hf_model(&hf_spec, |msg, _p, _cur, _tot| {
                        println!("  {}", msg);
                    }).await {
                        Ok(_) => println!("{} Model successfully loaded into in-process engine!\n", "✔".green().bold()),
                        Err(e) => println!("{} Failed to download model: {}\n", "✖".red(), e),
                    }
                }
            }
        }
    }

    if let Err(e) = agent.health_check().await {
        println!("{}", format!("ℹ Notice: {}", e).yellow());
    }

    println!("\nEnter your task or question (type {}/help{} for command menu, or {}/exit{} to quit):\n", "'".bright_cyan(), "'".bright_cyan(), "'".dimmed(), "'".dimmed());

    loop {
        let est_tokens = agent.estimate_tokens();
        let mode_str = agent.get_mode().to_string();
        print!("{} [{}] [~{} tokens] ", "👤 User:".green().bold(), mode_str.yellow(), est_tokens.to_string().dimmed());
        io::stdout().flush()?;

        let mut line = String::new();
        if handle.read_line(&mut line)? == 0 {
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts.first().copied().unwrap_or("").to_lowercase();

        match cmd.as_str() {
            "/exit" | "/quit" => {
                println!("{}", "Goodbye!".bright_yellow());
                break;
            }
            "/hf" | "/download" => {
                let spec = parts.get(1).copied().unwrap_or("ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0");
                println!("Fetching and downloading Hugging Face model: {}...", spec.cyan());
                match agent.load_hf_model(spec, |msg, _p, _cur, _tot| {
                    println!("  {}", msg);
                }).await {
                    Ok(_) => println!("{} Model {} loaded successfully!", "✔".green().bold(), spec.cyan()),
                    Err(e) => println!("{} Failed to download model: {}", "✖".red(), e),
                }
                continue;
            }
            "/model" => {
                if let Some(path_str) = parts.get(1) {
                    let p = std::path::PathBuf::from(path_str);
                    match agent.reload_model(&p) {
                        Ok(_) => println!("{} Local model loaded: {}", "✔".green().bold(), p.display().to_string().cyan()),
                        Err(e) => println!("{} Failed to load model: {}", "✖".red(), e),
                    }
                } else {
                    println!("Usage: /model <path-to-gguf>");
                }
                continue;
            }
            "/backend" => {
                if let Some(name) = parts.get(1) {
                    let choice = match name.to_lowercase().as_str() {
                        "cuda" => crate::config::BackendChoice::Cuda,
                        "rocm" | "hip" => crate::config::BackendChoice::Hipblas,
                        "sycl" | "oneapi" => crate::config::BackendChoice::Sycl,
                        "vulkan" => crate::config::BackendChoice::Vulkan,
                        "cpu" => crate::config::BackendChoice::Cpu,
                        "auto" => crate::config::BackendChoice::Auto,
                        _ => {
                            println!("{} Invalid backend. Use: cuda, rocm, sycl, vulkan, cpu, auto", "✖".red());
                            continue;
                        }
                    };
                    match agent.switch_backend(choice) {
                        Ok(_) => println!("{} Switched backend to: {}", "✔".green().bold(), name.bright_yellow().bold()),
                        Err(e) => println!("{} Failed to switch backend: {}", "✖".red(), e),
                    }
                } else {
                    println!("\nActive backend: {}\nUsage: /backend cuda | rocm | sycl | vulkan | cpu | auto\n", agent.get_config().backend.to_string().bright_yellow().bold());
                }
                continue;
            }
            "/mode" => {
                if let Some(target_mode) = parts.get(1).map(|s| s.to_lowercase()) {
                    match target_mode.as_str() {
                        "general" => agent.set_mode(AgentMode::General),
                        "coder" | "coding" => agent.set_mode(AgentMode::Coder),
                        "automatic" | "auto" => agent.set_mode(AgentMode::Automatic),
                        _ => println!("{} Invalid mode. Use '/mode general', '/mode coder', or '/mode automatic'.", "✖".red()),
                    }
                } else {
                    println!("\nActive Mode: {}\nUsage: /mode general | /mode coder | /mode automatic\n", agent.get_mode().to_string().bright_yellow().bold());
                }
                continue;
            }
            "/automatic" | "/auto" => {
                agent.set_mode(AgentMode::Automatic);
                continue;
            }
            "/status" => {
                agent.print_status();
                continue;
            }
            "/reset" | "/clear" => {
                agent.reset_context();
                continue;
            }
            "/compact" => {
                let limit = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(256000 / 2);
                if let Err(e) = agent.compact_context(limit).await {
                    println!("{} Failed to compact context: {}", "✖".red(), e);
                }
                continue;
            }
            "/save" => {
                let name = parts.get(1).copied().unwrap_or("default_session");
                match agent.save_session(name) {
                    Ok(path) => println!("{} Session saved to: {}", "✔".green(), path.display().to_string().cyan()),
                    Err(e) => println!("{} Failed to save session: {}", "✖".red(), e),
                }
                continue;
            }
            "/load" => {
                let name = parts.get(1).copied().unwrap_or("default_session");
                match agent.load_session(name) {
                    Ok(path) => println!("{} Session loaded from: {}", "✔".green(), path.display().to_string().cyan()),
                    Err(e) => println!("{} Failed to load session: {}", "✖".red(), e),
                }
                continue;
            }
            "/sessions" => {
                if let Err(e) = agent.list_sessions() {
                    println!("{} Failed to list sessions: {}", "✖".red(), e);
                }
                continue;
            }
            "/speech" => {
                let enabled = agent.toggle_speech();
                if enabled {
                    println!("{} Text-to-Speech (TTS) enabled. Gemma will speak responses aloud.", "🔊".green().bold());
                } else {
                    println!("{} Text-to-Speech (TTS) disabled.", "🔇".bright_yellow().bold());
                }
                continue;
            }
            "/help" => {
                println!("\n{}", "==================================================".cyan());
                println!("{}", "   💡 QT_llama.cpp COMMAND REFERENCE (/help)     ".bright_yellow().bold());
                println!("{}", "==================================================".cyan());
                println!("  /hf [repo:quant]               - Download & load Hugging Face GGUF model (default: ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0)");
                println!("  /model <path>                  - Load a local .gguf model file");
                println!("  /backend [cuda|rocm|sycl|vulkan|cpu|auto] - Switch active GPU/CPU compute backend");
                println!("  /speech                        - Toggle Text-to-Speech audio output");
                println!("  /mode [general|coder|automatic] - Switch between General, Coding & Automatic Modes");
                println!("  /automatic, /auto              - Switch instantly into Autonomous Mode");
                println!("  /status                        - View system telemetry, active mode, rules & token stats");
                println!("  /save [name]                   - Save current session history to .gemma/sessions/");
                println!("  /load [name]                   - Load a saved session history file");
                println!("  /sessions                      - List all saved sessions");
                println!("  /reset, /clear                 - Fully clear context to clean system prompt");
                println!("  /compact [tokens]              - Compact & summarize context");
                println!("  /help                          - Show this help menu");
                println!("  /exit, /quit                   - Exit the agent");
                println!("{}\n", "==================================================".cyan());
                continue;
            }
            _ => {}
        }

        if let Err(e) = agent.process_user_request(input).await {
            println!("\n{} Error processing request: {}", "✖".red().bold(), e);
        }

        println!();
    }

    Ok(())
}
