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
    println!("{}", "   🚀 GEMMA 4 GENERAL-PURPOSE & VIBE-CODING CLI   ".bright_cyan().bold());
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

    if let Err(e) = agent.health_check().await {
        println!("{}", format!("ℹ Notice: {}", e).yellow());
    }

    println!("\nEnter your task or question (type {}/help{} for command menu, or {}/exit{} to quit):\n", "'".bright_cyan(), "'".bright_cyan(), "'".dimmed(), "'".dimmed());

    let stdin = io::stdin();
    let mut handle = stdin.lock();

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
                println!("{}", "   💡 GEMMA 4 AGENT COMMAND REFERENCE (/help)     ".bright_yellow().bold());
                println!("{}", "==================================================".cyan());
                println!("  /speech                        - Toggle Text-to-Speech audio output (Disabled by default)");
                println!("  /mode [general|coder|automatic] - Switch between General, Coding & Automatic Inner Monologue Modes");
                println!("  /automatic, /auto              - Switch instantly into Autonomous Inner Monologue Mode");
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
