use anyhow::Result;
use colored::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::client::{ChatMessage, LlamaClient, StreamEvent};
use crate::config::{AgentMode, Config};
use crate::markdown::print_rich_markdown;
use crate::rules::WorkspaceRules;
use crate::tools::ToolRegistry;

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionState {
    pub mode: AgentMode,
    pub messages: Vec<ChatMessage>,
}

pub struct GemmaAgent {
    config: Config,
    client: LlamaClient,
    registry: ToolRegistry,
    history: Vec<ChatMessage>,
    workspace_rules: WorkspaceRules,
}

impl GemmaAgent {
    pub fn new(config: Config) -> Self {
        let client = LlamaClient::with_config(&config);
        let registry = ToolRegistry::new();
        let workspace_rules = WorkspaceRules::discover(&config.workspace_root);

        let system_prompt = if let Some(custom) = &config.system_prompt {
            custom.clone()
        } else {
            config.get_system_prompt(config.mode, &workspace_rules.combined_instructions)
        };

        let history = vec![ChatMessage::system(system_prompt)];

        crate::tools::media::set_speech_enabled(false);

        Self {
            config,
            client,
            registry,
            history,
            workspace_rules,
        }
    }

    pub fn toggle_speech(&mut self) -> bool {
        let current = crate::tools::media::is_speech_enabled();
        let new_state = !current;
        crate::tools::media::set_speech_enabled(new_state);
        new_state
    }

    pub fn is_speech_enabled(&self) -> bool {
        crate::tools::media::is_speech_enabled()
    }

    pub fn get_mode(&self) -> AgentMode {
        self.config.mode
    }

    pub fn set_mode(&mut self, mode: AgentMode) {
        self.config.mode = mode;
        println!("Switched to mode: {}", mode.to_string().cyan());
        self.reset_context();
    }

    pub fn reset_context(&mut self) {
        let system_prompt = self
            .config
            .get_system_prompt(self.config.mode, &self.workspace_rules.combined_instructions);
        self.history = vec![ChatMessage::system(system_prompt)];
        println!("{}", "Context cleared.".green());
    }

    pub fn get_rules(&self) -> &WorkspaceRules {
        &self.workspace_rules
    }

    pub fn loaded_rules_count(&self) -> usize {
        self.workspace_rules.rule_sources.len()
    }

    pub async fn health_check(&self) -> Result<String> {
        self.client.health_check().await
    }

    pub fn has_model_loaded(&self) -> bool {
        self.client.has_engine()
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn reload_model(&mut self, model_path: &std::path::Path) -> Result<()> {
        let n_layers = self.config.n_gpu_layers.unwrap_or(-1);
        let engine = llama_cpp_binding::LlamaEngine::new(model_path, n_layers, self.config.max_context_tokens as u32)?;
        self.client = crate::client::LlamaClient::with_engine(std::sync::Arc::new(engine), self.config.system_prompt.clone());
        self.config.model = model_path.display().to_string();
        Ok(())
    }

    pub fn estimated_tokens(&self) -> usize {
        let mut total_chars = 0;
        for msg in &self.history {
            if let Some(text) = msg.get_text_content() {
                total_chars += text.len();
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for tc in tool_calls {
                    total_chars += tc.function.name.len() + tc.function.arguments.len();
                }
            }
        }
        total_chars / 4
    }

    pub fn estimate_tokens(&self) -> usize {
        self.estimated_tokens()
    }

    pub fn print_status(&self) {
        println!("\n{}", "=== GEMMA AGENT SYSTEM TELEMETRY ===".bold().cyan());
        println!(" Mode           : {}", self.config.mode.to_string().bright_yellow().bold());
        println!(" Speech Output  : {}", if self.is_speech_enabled() { "Enabled".green() } else { "Disabled".dimmed() });
        println!(" Server URL     : {}", self.config.server_url.bright_green());
        println!(" Model          : {}", self.config.model.bright_white());
        println!(" Workspace Root : {}", self.config.workspace_root.display().to_string().bright_cyan());
        println!(" Current Tokens : ~{} (max: {})", self.estimated_tokens().to_string().bold(), self.config.max_context_tokens);
        println!(" History Messages: {}", self.history.len());
        println!(" Registered Tools: {}", self.registry.definitions().len());
        println!(" Loaded Rules   : {}", self.loaded_rules_count());
        for src in &self.workspace_rules.rule_sources {
            println!("   - {}", src.display());
        }
        println!("====================================\n");
    }

    pub async fn compact_context(&mut self, target_threshold: usize) -> Result<()> {
        println!("{}", "Compacting context...".yellow());
        let current_tokens = self.estimated_tokens();
        if current_tokens < target_threshold && self.history.len() <= 4 {
            println!("Context token count (~{}) is below threshold.", current_tokens);
            return Ok(());
        }

        let system_msg = self.history.first().cloned();
        let recent_msgs: Vec<ChatMessage> = self.history.iter().rev().take(4).rev().cloned().collect();

        let mut summary_req_msgs = Vec::new();
        summary_req_msgs.push(ChatMessage::system(
            "Summarize the conversation history concisely, retaining essential context, task goals, decisions, and file paths.",
        ));

        for msg in &self.history[1..self.history.len().saturating_sub(4)] {
            summary_req_msgs.push(msg.clone());
        }

        let req = crate::client::ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: summary_req_msgs,
            tools: None,
            tool_choice: None,
            temperature: Some(0.3),
            max_tokens: Some(1024),
            stream: Some(false),
        };

        match self.client.send_completion(&req).await {
            Ok(resp) => {
                if let Some(choice) = resp.choices.first() {
                    if let Some(summary_text) = choice.message.get_text_content() {
                        let summary_msg = ChatMessage::user(format!(
                            "[COMPACTED CONTEXT SUMMARY]:\n{}",
                            summary_text
                        ));

                        let mut new_history = Vec::new();
                        if let Some(sys) = system_msg {
                            new_history.push(sys);
                        }
                        new_history.push(summary_msg);
                        new_history.extend(recent_msgs);

                        self.history = new_history;
                        println!(
                            "Context compacted successfully! New token count: ~{}",
                            self.estimated_tokens()
                        );
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                println!("{}: {}", "Auto-compaction failed, keeping history".red(), e);
            }
        }
        Ok(())
    }

    pub fn save_session(&self, name: &str) -> Result<PathBuf> {
        let sessions_dir = self.config.workspace_root.join(".gemma").join("sessions");
        fs::create_dir_all(&sessions_dir)?;
        let file_path = sessions_dir.join(format!("{}.json", name));
        let serialized = serde_json::to_string_pretty(&self.history)?;
        fs::write(&file_path, serialized)?;
        println!("Session saved to: {}", file_path.display().to_string().cyan());
        Ok(file_path)
    }

    pub fn load_session(&mut self, name: &str) -> Result<PathBuf> {
        let file_path = self
            .config
            .workspace_root
            .join(".gemma")
            .join("sessions")
            .join(format!("{}.json", name));
        if !file_path.exists() {
            return Err(anyhow::anyhow!("Session file not found: {}", file_path.display()));
        }
        let content = fs::read_to_string(&file_path)?;
        let history: Vec<ChatMessage> = serde_json::from_str(&content)?;
        self.history = history;
        println!("Loaded session: {}", name.cyan());
        Ok(file_path)
    }

    pub fn list_sessions(&self) -> Result<Vec<String>> {
        let sessions_dir = self.config.workspace_root.join(".gemma").join("sessions");
        if !sessions_dir.exists() {
            return Ok(Vec::new());
        }

        let mut list = Vec::new();
        for entry in fs::read_dir(sessions_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    list.push(stem.to_string());
                }
            }
        }
        Ok(list)
    }

    pub async fn process_user_request(&mut self, user_input: &str) -> Result<()> {
        self.run_turn(user_input).await
    }

    pub async fn process_user_request_stream<F>(&mut self, user_input: &str, mut event_sink: F) -> Result<(String, String)>
    where
        F: FnMut(StreamEvent) + Send + 'static,
    {
        if self.estimated_tokens() >= self.config.max_context_tokens {
            self.compact_context(self.config.max_context_tokens / 2).await?;
        }

        self.history.push(ChatMessage::user(user_input));
        let tool_defs = self.registry.definitions();

        let mut final_content = String::new();
        let mut final_thought = String::new();
        let mut loop_count = 0;

        loop {
            loop_count += 1;
            if loop_count > 10 {
                break;
            }

            let req = crate::client::ChatCompletionRequest {
                model: self.config.model.clone(),
                messages: self.history.clone(),
                tools: Some(tool_defs.clone()),
                tool_choice: None,
                temperature: Some(self.config.temperature),
                max_tokens: Some(self.config.max_tokens),
                stream: Some(true),
            };

            let mut assembled_tool_calls = Vec::new();
            let assistant_msg = self
                .client
                .stream_completion(&req, |e| {
                    if let StreamEvent::ToolCallAssembled(ref tc) = e {
                        assembled_tool_calls.push(tc.clone());
                    }
                    event_sink(e);
                })
                .await?;

            let content = assistant_msg.get_text_content().unwrap_or_default().trim().to_string();
            let thought = assistant_msg.reasoning_content.clone().unwrap_or_default().trim().to_string();

            if !content.is_empty() {
                final_content = content.clone();
            }
            if !thought.is_empty() {
                final_thought = thought.clone();
            }

            self.history.push(assistant_msg.clone());

            let tool_calls = assistant_msg.tool_calls.clone().unwrap_or(assembled_tool_calls);
            if tool_calls.is_empty() {
                break;
            }

            for tool_call in tool_calls {
                let name = &tool_call.function.name;
                let raw_args = &tool_call.function.arguments;

                let parsed_args: serde_json::Value = match serde_json::from_str(raw_args) {
                    Ok(val) => val,
                    Err(e) => {
                        let err_msg = format!("Failed to parse arguments JSON: {}", e);
                        self.history.push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                        continue;
                    }
                };

                let tool_opt = self.registry.get(name);
                let tool_result = match tool_opt {
                    Some(tool) => tool.execute(&self.config.workspace_root, parsed_args).await,
                    None => Err(anyhow::anyhow!("Unknown tool requested: {}", name)),
                };

                match tool_result {
                    Ok(output) => {
                        event_sink(StreamEvent::ToolExecuted {
                            name: name.clone(),
                            result: output.clone(),
                        });
                        self.history.push(ChatMessage::tool(tool_call.id.clone(), name, output.clone()));
                    }
                    Err(e) => {
                        let err_msg = format!("Tool execution failed: {}", e);
                        event_sink(StreamEvent::ToolExecuted {
                            name: name.clone(),
                            result: err_msg.clone(),
                        });
                        self.history.push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                    }
                }
            }
        }

        Ok((final_content, final_thought))
    }

    pub async fn run_turn(&mut self, user_input: &str) -> Result<()> {
        if self.estimated_tokens() >= self.config.max_context_tokens {
            self.compact_context(self.config.max_context_tokens / 2).await?;
        }

        self.history.push(ChatMessage::user(user_input));
        let tool_defs = self.registry.definitions();

        let mut last_model_output: Option<String> = None;
        let mut model_output_repeat_count = 0;

        loop {
            let req = crate::client::ChatCompletionRequest {
                model: self.config.model.clone(),
                messages: self.history.clone(),
                tools: Some(tool_defs.clone()),
                tool_choice: None,
                temperature: Some(self.config.temperature),
                max_tokens: Some(self.config.max_tokens),
                stream: Some(true),
            };

            let mut reasoning_header_printed = false;
            let mut last_was_reasoning = false;
            let mut content_header_printed = false;
            let mut assembled_tool_calls = Vec::new();
            let mut last_metrics: Option<(usize, f64, f64)> = None;

            let assistant_msg_res = self
                .client
                .stream_completion(&req, |event| match event {
                    StreamEvent::Reasoning(text) => {
                        if !reasoning_header_printed && !text.trim().is_empty() {
                            println!("\n{}", "🧠 ──────────────── Thought Process ────────────────".bright_yellow().bold());
                            reasoning_header_printed = true;
                        }
                        if reasoning_header_printed {
                            print!("{}", text.yellow().dimmed());
                            let _ = std::io::stdout().flush();
                            last_was_reasoning = true;
                        }
                    }
                    StreamEvent::Content(text) => {
                        if last_was_reasoning {
                            println!("\n{}", "────────────────────────────────────────────────────".bright_yellow().dimmed());
                            print!("\n{}: ", "🤖 Gemma".bold().green());
                            let _ = std::io::stdout().flush();
                            last_was_reasoning = false;
                            content_header_printed = true;
                        } else if !content_header_printed {
                            print!("\n{}: ", "🤖 Gemma".bold().green());
                            let _ = std::io::stdout().flush();
                            content_header_printed = true;
                        }
                        print!("{}", text);
                        let _ = std::io::stdout().flush();
                    }
                    StreamEvent::ToolCallAssembled(tc) => {
                        assembled_tool_calls.push(tc);
                    }
                    StreamEvent::ToolExecuted { .. } => {}
                    StreamEvent::Metrics { token_count, elapsed_secs, tokens_per_sec } => {
                        last_metrics = Some((token_count, elapsed_secs, tokens_per_sec));
                    }
                    StreamEvent::Finish(_) => {}
                })
                .await;

            if reasoning_header_printed && last_was_reasoning {
                println!("\n{}", "────────────────────────────────────────────────────".bright_yellow().dimmed());
            }

            if let Some((tokens, elapsed, tps)) = last_metrics {
                println!(
                    "{}",
                    format!("⚡ [{} tokens in {:.2}s | {:.1} tk/s]", tokens, elapsed, tps)
                        .bright_black()
                        .italic()
                );
            }

            let assistant_msg = match assistant_msg_res {
                Ok(msg) => msg,
                Err(e) => {
                    println!("\n{}: {}", "Error calling model".red(), e);
                    return Err(e);
                }
            };

            let current_output_text = assistant_msg.get_text_content().unwrap_or_default().trim().to_string();
            if !current_output_text.is_empty() {
                if let Some(ref prev) = last_model_output {
                    if prev == &current_output_text {
                        model_output_repeat_count += 1;
                    } else {
                        model_output_repeat_count = 1;
                    }
                } else {
                    model_output_repeat_count = 1;
                }
                last_model_output = Some(current_output_text.clone());

                if model_output_repeat_count >= 10 {
                    println!(
                        "\n{}",
                        "⚠️ Model repetition detected: The model produced the exact same output 10 times consecutively. Stopping execution loop."
                            .yellow()
                            .bold()
                    );
                    break;
                }

                println!("\n{}", "─── 🎨 Rich Formatted Output ───".dimmed());
                print_rich_markdown(&current_output_text);
                println!("{}", "────────────────────────────────".dimmed());
            }

            self.history.push(assistant_msg.clone());

            let tool_calls = assistant_msg.tool_calls.clone().unwrap_or(assembled_tool_calls);
            if tool_calls.is_empty() {
                break;
            }

            for tool_call in tool_calls {
                let name = &tool_call.function.name;
                let raw_args = &tool_call.function.arguments;

                println!(
                    "\n{} Requesting tool {} with args: {}",
                    "🔧 [Tool Call]".bold().yellow(),
                    name.cyan(),
                    raw_args.dimmed()
                );

                let parsed_args: serde_json::Value = match serde_json::from_str(raw_args) {
                    Ok(val) => val,
                    Err(e) => {
                        let err_msg = format!("Failed to parse arguments JSON: {}", e);
                        self.history.push(ChatMessage::tool(tool_call.id, name, err_msg));
                        continue;
                    }
                };

                let tool_opt = self.registry.get(name);
                let tool_result = match tool_opt {
                    Some(tool) => {
                        println!("⚡ Executing {}", name.cyan());
                        tool.execute(&self.config.workspace_root, parsed_args).await
                    }
                    None => Err(anyhow::anyhow!("Unknown tool requested: {}", name)),
                };

                match tool_result {
                    Ok(output) => {
                        let display_output = if output.len() > 1000 {
                            format!("{}... (truncated)", crate::markdown::truncate_utf8(&output, 1000))
                        } else {
                            output.clone()
                        };
                        println!("📥 Tool Output:\n{}", display_output.dimmed());
                        self.history.push(ChatMessage::tool(tool_call.id.clone(), name, output.clone()));
                    }
                    Err(e) => {
                        let err_msg = format!("Tool execution failed: {}", e);
                        println!("{}: {}", "✖".red(), err_msg);
                        self.history.push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                    }
                }
            }
        }

        Ok(())
    }
}
