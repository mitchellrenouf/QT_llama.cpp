#![no_std]
#![cfg_attr(not(test), no_main)]

pub use mrml_agent::config;
use mrml_agent::{AgentMode, Config, MrmlAgent};
use mrml_runtime::{Text, Vector, mrml_format as format, mrml_print as print, mrml_println as println};
use mrml_terminal_style::Colorize;

fn application_main() -> mrml_error::Result<()> {
    mrml_tools::block_on(async_main())
}

mrml_runtime::mrml_entrypoint!(application_main);

async fn async_main() -> mrml_error::Result<()> {
    let config = Config::parse();

    println!(
        "{}",
        "==================================================".magenta()
    );
    println!(
        "{}",
        "   🚀 MRML GENERAL-PURPOSE & VIBE-CODING CLI   "
            .bright_cyan()
            .bold()
    );
    println!(
        "{}",
        "==================================================".magenta()
    );
    println!(
        " Mode        : {}",
        format!("{}", config.mode).bright_yellow().bold()
    );
    println!(
        " Inference   : {}",
        "Native MRML GGUF Engine".bright_green().bold()
    );
    if let Some(hf_spec) = &config.hf {
        println!(" HuggingFace : {}", hf_spec.bright_cyan().bold());
    }
    println!(" Model Path  : {}", config.model.cyan());
    println!(
        " Context Size: {} tokens (GPU KV cache)",
        format!("{}", config.ctx_size).bright_yellow().bold()
    );
    let auto_cache = if config.ctx_size >= 131_072 {
        "q4_0"
    } else {
        "f16"
    };
    let cache_k = if config.cache_type_k == "auto" {
        auto_cache
    } else {
        &config.cache_type_k
    };
    let cache_v = if config.cache_type_v == "auto" {
        auto_cache
    } else {
        &config.cache_type_v
    };
    println!(
        " KV Cache    : K={} / V={}",
        cache_k.bright_yellow(),
        cache_v.bright_yellow()
    );
    println!(
        " Max Context : {} tokens (auto-compact enabled)",
        format!("{}", config.max_context_tokens).yellow()
    );
    println!(
        " Auto-Approve: {}",
        format!("{}", config.auto_approve).bright_white()
    );

    let mut agent = MrmlAgent::new(config.clone());
    let _ = agent.init_mcp_servers().await;

    if agent.get_rules().has_rules() {
        println!(
            " Loaded Rules: {}",
            format!("{}", agent.loaded_rules_count()).bright_green().bold()
        );
        for src in &agent.get_rules().rule_sources {
            println!("   - {}", src.dimmed());
        }
    }

    println!(
        "{}",
        "--------------------------------------------------".dimmed()
    );

    if !agent.has_model_loaded() {
        if let Some(hf_spec) = config.hf.clone() {
            println!(
                "{} Model weights for '{}' are not cached locally.",
                "📥".cyan(),
                hf_spec.bright_cyan().bold()
            );
            print!(
                "{} Download & load model weights now? [Y/n]: ",
                "❓".yellow()
            );
            if let Ok(Some(ans)) = mrml_runtime::read_stdin_line() {
                let trimmed = Text::from(ans.trim()).to_ascii_lowercase();
                if trimmed.is_empty() || trimmed == "y" || trimmed == "yes" {
                    println!(
                        "\nFetching and downloading Hugging Face model: {}...",
                        hf_spec.cyan()
                    );
                    match agent
                        .load_hf_model(&hf_spec, |msg, _p, _cur, _tot| {
                            println!("  {}", msg);
                        })
                        .await
                    {
                        Ok(_) => println!(
                            "{} Model successfully loaded into in-process engine!\n",
                            "✔".green().bold()
                        ),
                        Err(e) => println!("{} Failed to download model: {}\n", "✖".red(), e),
                    }
                }
            }
        }
    }

    if let Err(e) = agent.health_check().await {
        println!("{}", format!("ℹ Notice: {}", e).yellow());
    }

    if let Some(prompt) = &config.prompt {
        println!("{} {}\n", "👤 User:".green().bold(), prompt.bright_white());
        let _ = agent.process_user_request(prompt).await?;
        return Ok(());
    }

    println!(
        "\nEnter your task or question (type {}/help{} for command menu, or {}/exit{} to quit):\n",
        "'".bright_cyan(),
        "'".bright_cyan(),
        "'".dimmed(),
        "'".dimmed()
    );

    loop {
        let est_tokens = agent.estimate_tokens();
        let mode_str = format!("{}", agent.get_mode());
        print!(
            "{} [{}] [~{} tokens] ",
            "👤 User:".green().bold(),
            mode_str.yellow(),
            format!("{}", est_tokens).dimmed()
        );

        let Some(line) = mrml_runtime::read_stdin_line()
            .map_err(|_| mrml_error::anyhow!("failed reading standard input"))?
        else {
            break;
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        let parts: Vector<&str> = input.split_whitespace().collect();
        let cmd = Text::from(parts.first().copied().unwrap_or("")).to_ascii_lowercase();

        match cmd.as_str() {
            "/exit" | "/quit" => {
                println!("{}", "Goodbye!".bright_yellow());
                break;
            }
            "/hf" | "/download" => {
                let spec = parts
                    .get(1)
                    .copied()
                    .unwrap_or("ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0");
                println!(
                    "Fetching and downloading Hugging Face model: {}...",
                    spec.cyan()
                );
                match agent
                    .load_hf_model(spec, |msg, _p, _cur, _tot| {
                        println!("  {}", msg);
                    })
                    .await
                {
                    Ok(_) => println!(
                        "{} Model {} loaded successfully!",
                        "✔".green().bold(),
                        spec.cyan()
                    ),
                    Err(e) => println!("{} Failed to download model: {}", "✖".red(), e),
                }
                continue;
            }
            "/model" => {
                if let Some(path_str) = parts.get(1) {
                    match agent.reload_model(path_str) {
                        Ok(_) => println!(
                            "{} Local model loaded: {}",
                            "✔".green().bold(),
                            path_str.cyan()
                        ),
                        Err(e) => println!("{} Failed to load model: {}", "✖".red(), e),
                    }
                } else {
                    println!("Usage: /model <path-to-gguf>");
                }
                continue;
            }
            "/backend" => {
                if let Some(name) = parts.get(1) {
                    let choice = if name.eq_ignore_ascii_case("cuda") {
                        crate::config::BackendChoice::Cuda
                    } else if name.eq_ignore_ascii_case("rocm") || name.eq_ignore_ascii_case("hip") {
                        crate::config::BackendChoice::Rocm
                    } else if name.eq_ignore_ascii_case("sycl") || name.eq_ignore_ascii_case("oneapi") {
                        crate::config::BackendChoice::Sycl
                    } else if name.eq_ignore_ascii_case("vulkan") {
                        crate::config::BackendChoice::Vulkan
                    } else if name.eq_ignore_ascii_case("cpu") {
                        crate::config::BackendChoice::Cpu
                    } else if name.eq_ignore_ascii_case("auto") {
                        crate::config::BackendChoice::Auto
                    } else {
                            println!(
                                "{} Invalid backend. Use: cuda, rocm, sycl, vulkan, cpu, auto",
                                "✖".red()
                            );
                            continue;
                    };
                    match agent.switch_backend(choice) {
                        Ok(_) => println!(
                            "{} Switched backend to: {}",
                            "✔".green().bold(),
                            name.bright_yellow().bold()
                        ),
                        Err(e) => println!("{} Failed to switch backend: {}", "✖".red(), e),
                    }
                } else {
                    println!(
                        "\nActive backend: {}\nUsage: /backend cuda | rocm | sycl | vulkan | cpu | auto\n",
                        format!("{}", agent.get_config().backend)
                            .bright_yellow()
                            .bold()
                    );
                }
                continue;
            }
            "/mode" => {
                if let Some(target_mode) = parts.get(1) {
                    if target_mode.eq_ignore_ascii_case("general") {
                        agent.set_mode(AgentMode::General)
                    } else if target_mode.eq_ignore_ascii_case("coder") || target_mode.eq_ignore_ascii_case("coding") {
                        agent.set_mode(AgentMode::Coder)
                    } else if target_mode.eq_ignore_ascii_case("automatic") || target_mode.eq_ignore_ascii_case("auto") {
                        agent.set_mode(AgentMode::Automatic)
                    } else {
                        println!(
                            "{} Invalid mode. Use '/mode general', '/mode coder', or '/mode automatic'.",
                            "✖".red()
                        )
                    }
                } else {
                    println!(
                        "\nActive Mode: {}\nUsage: /mode general | /mode coder | /mode automatic\n",
                        format!("{}", agent.get_mode()).bright_yellow().bold()
                    );
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
                let limit = parts
                    .get(1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(256000 / 2);
                if let Err(e) = agent.compact_context(limit).await {
                    println!("{} Failed to compact context: {}", "✖".red(), e);
                }
                continue;
            }
            "/save" => {
                let name = parts.get(1).copied().unwrap_or("default_session");
                match agent.save_session(name) {
                    Ok(path) => println!(
                        "{} Session saved to: {}",
                        "✔".green(),
                        path.cyan()
                    ),
                    Err(e) => println!("{} Failed to save session: {}", "✖".red(), e),
                }
                continue;
            }
            "/load" => {
                let name = parts.get(1).copied().unwrap_or("default_session");
                match agent.load_session(name) {
                    Ok(path) => println!(
                        "{} Session loaded from: {}",
                        "✔".green(),
                        path.cyan()
                    ),
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
                    println!(
                        "{} Text-to-Speech (TTS) enabled. MRML will speak responses aloud.",
                        "🔊".green().bold()
                    );
                } else {
                    println!(
                        "{} Text-to-Speech (TTS) disabled.",
                        "🔇".bright_yellow().bold()
                    );
                }
                continue;
            }
            "/help" => {
                println!(
                    "\n{}",
                    "==================================================".cyan()
                );
                println!(
                    "{}",
                    "   💡 MRML COMMAND REFERENCE (/help)     "
                        .bright_yellow()
                        .bold()
                );
                println!(
                    "{}",
                    "==================================================".cyan()
                );
                println!(
                    "  /hf [repo:quant]               - Download & load Hugging Face GGUF model (default: ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_0)"
                );
                println!("  /model <path>                  - Load a local .gguf model file");
                println!(
                    "  /backend [cuda|rocm|sycl|vulkan|cpu|auto] - Switch active GPU/CPU compute backend"
                );
                println!("  /speech                        - Toggle Text-to-Speech audio output");
                println!(
                    "  /mode [general|coder|automatic] - Switch between General, Coding & Automatic Modes"
                );
                println!(
                    "  /automatic, /auto              - Switch instantly into Autonomous Mode"
                );
                println!(
                    "  /status                        - View system telemetry, active mode, rules & token stats"
                );
                println!(
                    "  /save [name]                   - Save current session history to .mrml/sessions/"
                );
                println!("  /load [name]                   - Load a saved session history file");
                println!("  /sessions                      - List all saved sessions");
                println!(
                    "  /reset, /clear                 - Fully clear context to clean system prompt"
                );
                println!("  /compact [tokens]              - Compact & summarize context");
                println!("  /help                          - Show this help menu");
                println!("  /exit, /quit                   - Exit the agent");
                println!(
                    "{}\n",
                    "==================================================".cyan()
                );
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
