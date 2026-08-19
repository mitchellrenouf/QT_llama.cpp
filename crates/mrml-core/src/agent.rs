use anyhow::Result;
use mrml_runtime::{Vector, mrml_eprintln as eprintln, mrml_print as print, mrml_println as println};
use mrml_terminal_style::Colorize;

use crate::client::{ChatMessage, MrmlClient, StreamEvent};
use crate::config::{AgentMode, Config};
use crate::markdown::print_rich_markdown;
use crate::rules::WorkspaceRules;
use crate::tools::ToolRegistry;

#[allow(dead_code)]
#[derive(Debug)]
pub struct SessionState {
    pub mode: AgentMode,
    pub messages: Vector<ChatMessage>,
}

pub struct MrmlAgent {
    config: Config,
    client: MrmlClient,
    registry: ToolRegistry,
    history: Vector<ChatMessage>,
    workspace_rules: WorkspaceRules,
}

fn requests_live_local_time(input: &str) -> bool {
    let normalized = input.trim().to_ascii_lowercase();
    normalized.contains("what time is it")
        || normalized.contains("what's the time")
        || normalized.contains("current time")
        || normalized.contains("tell me the time")
        || normalized.contains("time right now")
}

fn verified_time_answer(tool_output: &str) -> Option<mrml_runtime::Text> {
    let stdout = tool_output
        .split("--- STDOUT ---")
        .nth(1)?
        .split("--- STDERR ---")
        .next()?;
    let value = stdout
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && *line != "(empty)")?;
    Some(format!("The current local time is **{value}**.").as_str().into())
}

fn verified_command_answer(tool_output: &str) -> Option<mrml_runtime::Text> {
    let stdout = tool_output
        .split("--- STDOUT ---")
        .nth(1)?
        .split("--- STDERR ---")
        .next()?
        .trim();
    if stdout.is_empty() || stdout == "(empty)" {
        None
    } else {
        Some(
            format!("The command printed:\n\n```text\n{}\n```", stdout)
                .as_str()
                .into(),
        )
    }
}

fn explicitly_requested_command(input: &str) -> Option<mrml_runtime::Text> {
    let normalized = input.to_ascii_lowercase();
    if !normalized.contains("run_command") {
        return None;
    }
    let start = normalized.find("execute ")? + "execute ".len();
    let remainder = &input[start..];
    let end = [", then", ", and then", " then tell", " and tell"]
        .iter()
        .filter_map(|marker| remainder.to_ascii_lowercase().find(marker))
        .min()
        .unwrap_or(remainder.len());
    let command = remainder[..end].trim().trim_matches(['`', '\'', '"']);
    (!command.is_empty()).then(|| command.into())
}

impl MrmlAgent {
    pub fn new(config: Config) -> Self {
        let client = MrmlClient::with_config(&config);
        let registry = ToolRegistry::new();
        let workspace_rules = WorkspaceRules::discover(&config.workspace_root);

        let system_prompt = if let Some(custom) = &config.system_prompt {
            custom.clone()
        } else {
            config
                .get_system_prompt(config.mode, &workspace_rules.combined_instructions)
                .as_str()
                .into()
        };

        let history = [ChatMessage::system(system_prompt)].into_iter().collect();

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
        self.client
            .set_thinking_enabled(crate::client::thinking_enabled_for_mode(mode));
        println!("Switched to mode: {}", mode.to_string().cyan());
        self.reset_context();
    }

    pub fn reset_context(&mut self) {
        let system_prompt = self.config.get_system_prompt(
            self.config.mode,
            &self.workspace_rules.combined_instructions,
        );
        self.history = [ChatMessage::system(system_prompt)].into_iter().collect();
        println!("{}", "Context cleared.".green());
    }

    pub fn get_rules(&self) -> &WorkspaceRules {
        &self.workspace_rules
    }

    pub fn get_client_arc(&self) -> mrml_runtime::Shared<MrmlClient> {
        mrml_runtime::Shared::new(self.client.clone())
    }

    pub fn loaded_rules_count(&self) -> usize {
        self.workspace_rules.rule_sources.len()
    }

    pub async fn health_check(&self) -> Result<mrml_runtime::Text> {
        self.client.health_check().await
    }

    pub async fn init_mcp_servers(&mut self) -> Result<()> {
        for server_cmd in &self.config.mcp_servers {
            let parts: Vector<&str> = server_cmd.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let program = parts[0];
            let args = &parts[1..];
            println!("🔌 Connecting to MCP Server: {}...", server_cmd.cyan());
            match crate::tools::mcp::McpClient::spawn(program, args).await {
                Ok(client) => match crate::tools::mcp::McpClient::list_tools(&client).await {
                    Ok(tools) => {
                        println!(
                            "   Loaded {} MCP tool(s):",
                            tools.len().to_string().bright_green().bold()
                        );
                        for t in tools {
                            println!(
                                "     - {} ({})",
                                t.name().bright_cyan(),
                                t.description().dimmed()
                            );
                            self.registry.register(t);
                        }
                    }
                    Err(e) => eprintln!("   Failed to list MCP tools: {}", e),
                },
                Err(e) => eprintln!("   Failed to spawn MCP client: {}", e),
            }
        }
        Ok(())
    }

    pub fn has_model_loaded(&self) -> bool {
        self.client.has_engine()
    }

    pub fn gpu_layer_residency(&self) -> Option<(usize, usize)> {
        self.client.gpu_layer_residency()
    }

    pub fn get_config(&self) -> &Config {
        &self.config
    }

    pub fn reload_model(&mut self, model_path: &str) -> Result<()> {
        let n_layers = self.config.n_gpu_layers.unwrap_or(-1);
        let backend_str = self.config.backend.to_string();
        let engine = mrml_model::ModelEngine::new(
            model_path,
            n_layers,
            self.config.ctx_size,
            &self.config.cache_type_k,
            &self.config.cache_type_v,
            Some(&backend_str),
        )
        .map_err(anyhow::Error::with_source)?;
        self.client = crate::client::MrmlClient::with_engine(
            mrml_runtime::Shared::new(engine),
            self.config.system_prompt.as_deref().map(Into::into),
            crate::client::thinking_enabled_for_mode(self.config.mode),
        );
        self.config.model = model_path.into();
        Ok(())
    }

    pub async fn load_hf_model<F>(&mut self, spec_str: &str, progress_cb: F) -> Result<()>
    where
        F: FnMut(&str, f32, usize, usize) + Send + 'static,
    {
        let spec = crate::hf::HfModelSpec::parse(spec_str)?;
        let files = crate::hf::resolve_or_fetch_hf_model(&spec, progress_cb).await?;
        self.reload_model(&files.primary_entry_file)?;
        self.config.hf = Some(spec_str.into());
        Ok(())
    }

    pub fn switch_backend(&mut self, backend: crate::config::BackendChoice) -> Result<()> {
        self.config.backend = backend;
        let model_path = if let Some(hf_spec_str) = &self.config.hf {
            crate::client::find_model_file(hf_spec_str)
                .or_else(|| crate::client::find_model_file(&self.config.model))
        } else {
            crate::client::find_model_file(&self.config.model)
        };
        if let Some(path) = model_path {
            self.reload_model(&path)?;
        }
        Ok(())
    }

    pub fn set_generation_settings(&mut self, temperature: Option<f32>, max_tokens: Option<u32>) {
        if let Some(temperature) = temperature {
            self.config.temperature = temperature.clamp(0.0, 2.0);
        }
        if let Some(max_tokens) = max_tokens {
            self.config.max_tokens = max_tokens.clamp(1, 32_768);
        }
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
        println!("\n{}", "=== MRML AGENT SYSTEM TELEMETRY ===".bold().cyan());
        println!(
            " Mode           : {}",
            self.config.mode.to_string().bright_yellow().bold()
        );
        println!(
            " Speech Output  : {}",
            if self.is_speech_enabled() {
                "Enabled".green()
            } else {
                "Disabled".dimmed()
            }
        );
        println!(
            " Server URL     : {}",
            self.config.server_url.bright_green()
        );
        println!(" Model          : {}", self.config.model.bright_white());
        println!(
            " Workspace Root : {}",
            self.config.workspace_root.bright_cyan()
        );
        println!(
            " Current Tokens : ~{} (max: {})",
            self.estimated_tokens().to_string().bold(),
            self.config.max_context_tokens
        );
        println!(" History Messages: {}", self.history.len());
        println!(" Registered Tools: {}", self.registry.definitions().len());
        println!(" Loaded Rules   : {}", self.loaded_rules_count());
        for src in &self.workspace_rules.rule_sources {
            println!("   - {}", src);
        }
        println!("====================================\n");
    }

    pub async fn compact_context(&mut self, target_threshold: usize) -> Result<()> {
        println!("{}", "Compacting context...".yellow());
        let current_tokens = self.estimated_tokens();
        if current_tokens < target_threshold && self.history.len() <= 4 {
            println!(
                "Context token count (~{}) is below threshold.",
                current_tokens
            );
            return Ok(());
        }

        let system_msg = self.history.first().cloned();
        let recent_msgs: Vector<ChatMessage> =
            self.history.iter().rev().take(4).rev().cloned().collect();

        let mut summary_req_msgs = Vector::new();
        summary_req_msgs.push(ChatMessage::system(
            "Summarize the conversation history concisely, retaining essential context, task goals, decisions, and file paths.",
        ));

        for msg in &self.history[1..self.history.len().saturating_sub(4)] {
            summary_req_msgs.push(msg.clone());
        }

        let req = crate::client::ChatCompletionRequest {
            model: self.config.model.as_str().into(),
            messages: summary_req_msgs.into_iter().collect(),
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

                        let mut new_history = Vector::new();
                        if let Some(sys) = system_msg {
                            new_history.push(sys);
                        }
                        new_history.push(summary_msg);
                        new_history.extend(recent_msgs);

                        self.history = new_history.into_iter().collect();
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

    pub fn save_session(&self, name: &str) -> Result<mrml_runtime::Text> {
        let sessions_dir = mrml_runtime::join_path(
            &mrml_runtime::join_path(&self.config.workspace_root, ".mrml"),
            "sessions",
        );
        mrml_runtime::create_dir_all(&sessions_dir)?;
        let file_path = mrml_runtime::join_path(&sessions_dir, &format!("{}.json", name));
        let serialized = serde_json::stringify(&serde_json::Value::Array(
            self.history.iter().map(ChatMessage::to_json).collect(),
        ));
        mrml_runtime::write_file(
            &file_path,
            serialized.as_bytes(),
        )?;
        println!(
            "Session saved to: {}",
            file_path.cyan()
        );
        Ok(file_path)
    }

    pub fn load_session(&mut self, name: &str) -> Result<mrml_runtime::Text> {
        let file_path = mrml_runtime::join_path(
            &mrml_runtime::join_path(
                &mrml_runtime::join_path(&self.config.workspace_root, ".mrml"),
                "sessions",
            ),
            &format!("{}.json", name),
        );
        if !mrml_runtime::path_is_file(&file_path) {
            return Err(anyhow::anyhow!(
                "Session file not found: {}",
                file_path
            ));
        }
        let content = mrml_runtime::read_file_text(
            &file_path,
        )?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        let history = value
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("session history must be a JSON array"))?
            .iter()
            .map(ChatMessage::from_json)
            .collect::<mrml_model::error::Result<Vector<_>>>()
            .map_err(anyhow::Error::with_source)?;
        self.history = history.into_iter().collect();
        println!("Loaded session: {}", name.cyan());
        Ok(file_path)
    }

    pub fn list_sessions(&self) -> Result<Vector<mrml_runtime::Text>> {
        let sessions_dir = mrml_runtime::join_path(
            &mrml_runtime::join_path(&self.config.workspace_root, ".mrml"),
            "sessions",
        );
        if !mrml_runtime::path_is_directory(&sessions_dir) {
            return Ok(Vector::new());
        }

        let mut list = Vector::new();
        for entry in mrml_runtime::read_directory(
            &sessions_dir,
        )? {
            if !entry.is_directory && entry.name.ends_with(".json") {
                list.push(entry.name[..entry.name.len() - 5].into());
            }
        }
        Ok(list)
    }

    pub async fn process_user_request(&mut self, user_input: &str) -> Result<()> {
        self.run_turn(user_input).await
    }

    pub async fn process_user_request_stream<F>(
        &mut self,
        user_input: &str,
        mut event_sink: F,
    ) -> Result<(mrml_runtime::Text, mrml_runtime::Text)>
    where
        F: FnMut(StreamEvent) + Send + 'static,
    {
        if self.estimated_tokens() >= self.config.max_context_tokens {
            self.compact_context(self.config.max_context_tokens / 2)
                .await?;
        }

        self.history.push(ChatMessage::user(user_input));
        // Keep the model prompt stable. Explicit commands are routed below without
        // inference, so changing the advertised schema set only perturbs ordinary
        // Gemma completions and cannot improve that fast path.
        let tool_defs = self.registry.definitions();

        // Live clock queries have one unambiguous tool dependency. Route them
        // deterministically instead of asking the language model to decide
        // whether it has clock access; the model still turns the verified tool
        // result into the user-facing answer.
        let is_clock_request = requests_live_local_time(user_input);
        let explicit_command = explicitly_requested_command(user_input);
        if is_clock_request || explicit_command.is_some() {
            #[cfg(windows)]
            let clock_command = "Get-Date -Format \"yyyy-MM-dd HH:mm:ss zzz\"";
            #[cfg(not(windows))]
            let clock_command = "date '+%Y-%m-%d %H:%M:%S %z'";
            let command = explicit_command.unwrap_or_else(|| clock_command.into());
            let call_id = format!("clock-{}", self.history.len());
            let args = serde_json::json!({ "command_line": (command.as_str()) });
            let tool_call = crate::client::ToolCall {
                id: call_id.as_str().into(),
                tool_type: "function".into(),
                function: crate::client::FunctionCall {
                    name: "run_command".into(),
                    arguments: args.to_string().as_str().into(),
                },
            };
            println!(
                "\n{} Requesting tool {} with args: {}",
                "🔧 [Tool Call]".bold().yellow(),
                "run_command".cyan(),
                args.to_string().dimmed()
            );
            self.history
                .push(ChatMessage::assistant(None, Some([tool_call].into())));
            let result = self
                .registry
                .get("run_command")
                .unwrap()
                .execute(
                    &self.config.workspace_root,
                    args,
                )
                .await;
            match result {
                Ok(output) => {
                    println!("⚡ Executing {}", "run_command".cyan());
                    println!("📥 Tool Output:\n{}", output.dimmed());
                    self.history
                        .push(ChatMessage::tool(call_id, "run_command", output.clone()));
                    event_sink(StreamEvent::ToolExecuted {
                        name: "run_command".into(),
                        result: output.as_str().into(),
                    });
                    if is_clock_request {
                        if let Some(answer) = verified_time_answer(&output) {
                            event_sink(StreamEvent::Content(answer.as_str().into()));
                            event_sink(StreamEvent::Finish("stop".into()));
                            self.history
                                .push(ChatMessage::assistant(Some(answer.as_str().into()), None));
                            return Ok((answer, mrml_runtime::Text::new()));
                        }
                    } else {
                        if let Some(answer) = verified_command_answer(&output) {
                            event_sink(StreamEvent::Content(answer.as_str().into()));
                            event_sink(StreamEvent::Finish("stop".into()));
                            self.history
                                .push(ChatMessage::assistant(Some(answer.as_str().into()), None));
                            return Ok((answer, mrml_runtime::Text::new()));
                        }
                    }
                }
                Err(error) => {
                    let output = format!("Tool execution failed: {error}");
                    println!("{}: {}", "✖".red(), output);
                    self.history
                        .push(ChatMessage::tool(call_id, "run_command", output));
                }
            }
        }

        let mut final_content = mrml_runtime::Text::new();
        let mut final_thought = mrml_runtime::Text::new();
        let mut loop_count = 0;

        loop {
            loop_count += 1;
            if loop_count > 10 {
                break;
            }

            let req = crate::client::ChatCompletionRequest {
                model: self.config.model.as_str().into(),
                messages: self.history.iter().cloned().collect(),
                tools: Some(tool_defs.iter().cloned().collect()),
                tool_choice: None,
                temperature: Some(self.config.temperature),
                max_tokens: Some(self.config.max_tokens),
                stream: Some(true),
            };

            let mut assembled_tool_calls = Vector::new();
            let assistant_msg = self
                .client
                .stream_completion(&req, |e| {
                    if let StreamEvent::ToolCallAssembled(ref tc) = e {
                        assembled_tool_calls.push(tc.clone());
                    }
                    event_sink(e);
                })
                .await?;

            let content = assistant_msg
                .get_text_content()
                .unwrap_or_default()
                .trim()
                .to_string();
            let thought = assistant_msg
                .reasoning_content
                .clone()
                .unwrap_or_default()
                .trim()
                .to_string();

            if !content.is_empty() {
                final_content = content.as_str().into();
            }
            if !thought.is_empty() {
                final_thought = thought.as_str().into();
            }

            self.history.push(assistant_msg.clone());

            let tool_calls = assistant_msg
                .tool_calls
                .clone()
                .unwrap_or_else(|| assembled_tool_calls.into_iter().collect());
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
                        self.history
                            .push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                        continue;
                    }
                };

                let tool_opt = self.registry.get(name);
                let tool_result = match tool_opt {
                    Some(tool) => {
                        tool.execute(
                            &self.config.workspace_root,
                            parsed_args,
                        )
                        .await
                    }
                    None => Err(mrml_tools::tool_error(format!(
                        "Unknown tool requested: {}",
                        name
                    ))),
                };

                match tool_result {
                    Ok(output) => {
                        event_sink(StreamEvent::ToolExecuted {
                                    name: name.as_str().into(),
                                    result: output.as_str().into(),
                        });
                        self.history.push(ChatMessage::tool(
                            tool_call.id.clone(),
                            name,
                            output.clone(),
                        ));
                    }
                    Err(e) => {
                        let err_msg = format!("Tool execution failed: {}", e);
                        event_sink(StreamEvent::ToolExecuted {
                                    name: name.as_str().into(),
                                    result: err_msg.as_str().into(),
                        });
                        self.history
                            .push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                    }
                }
            }
        }

        Ok((final_content, final_thought))
    }

    pub async fn run_turn(&mut self, user_input: &str) -> Result<()> {
        if self.estimated_tokens() >= self.config.max_context_tokens {
            self.compact_context(self.config.max_context_tokens / 2)
                .await?;
        }

        self.history.push(ChatMessage::user(user_input));
        let tool_defs = self.registry.definitions();

        let is_clock_request = requests_live_local_time(user_input);
        let explicit_command = explicitly_requested_command(user_input);
        if is_clock_request || explicit_command.is_some() {
            #[cfg(windows)]
            let clock_command = "Get-Date -Format \"yyyy-MM-dd HH:mm:ss zzz\"";
            #[cfg(not(windows))]
            let clock_command = "date '+%Y-%m-%d %H:%M:%S %z'";
            let command = explicit_command.unwrap_or_else(|| clock_command.into());
            let call_id = format!("clock-{}", self.history.len());
            let args = serde_json::json!({ "command_line": (command.as_str()) });
            let tool_call = crate::client::ToolCall {
                id: call_id.as_str().into(),
                tool_type: "function".into(),
                function: crate::client::FunctionCall {
                    name: "run_command".into(),
                    arguments: args.to_string().as_str().into(),
                },
            };
            println!(
                "\n{} Requesting tool {} with args: {}",
                "🔧 [Tool Call]".bold().yellow(),
                "run_command".cyan(),
                args.to_string().dimmed()
            );
            self.history
                .push(ChatMessage::assistant(None, Some([tool_call].into())));
            println!("⚡ Executing {}", "run_command".cyan());
            let result = self
                .registry
                .get("run_command")
                .unwrap()
                .execute(
                    &self.config.workspace_root,
                    args,
                )
                .await;
            match result {
                Ok(output) => {
                    println!("📥 Tool Output:\n{}", output.dimmed());
                    self.history
                        .push(ChatMessage::tool(call_id, "run_command", output.clone()));
                    if is_clock_request {
                        if let Some(answer) = verified_time_answer(&output) {
                            print!("\n{}: {}", "🤖 MRML".bold().green(), answer);
                            println!("\n{}", "─── 🎨 Rich Formatted Output ───".dimmed());
                            print_rich_markdown(&answer);
                            println!("{}", "────────────────────────────────".dimmed());
                            self.history
                                .push(ChatMessage::assistant(Some(answer.as_str().into()), None));
                            return Ok(());
                        }
                    } else {
                        if let Some(answer) = verified_command_answer(&output) {
                            print!("\n{}: {}", "🤖 MRML".bold().green(), answer);
                            println!("\n{}", "─── 🎨 Rich Formatted Output ───".dimmed());
                            print_rich_markdown(&answer);
                            println!("{}", "────────────────────────────────".dimmed());
                            self.history
                                .push(ChatMessage::assistant(Some(answer.as_str().into()), None));
                            return Ok(());
                        }
                    }
                }
                Err(error) => {
                    let output = format!("Tool execution failed: {error}");
                    println!("{}: {}", "✖".red(), output);
                    self.history
                        .push(ChatMessage::tool(call_id, "run_command", output));
                }
            }
        }

        let mut last_model_output: Option<mrml_runtime::Text> = None;
        let mut model_output_repeat_count = 0;

        let mut step_count = 0;
        loop {
            step_count += 1;
            if step_count > 15 {
                println!(
                    "\n{}",
                    "⚠️ Reached maximum autonomous step limit (15). Stopping."
                        .yellow()
                        .bold()
                );
                break;
            }

            let req = crate::client::ChatCompletionRequest {
                model: self.config.model.as_str().into(),
                messages: self.history.iter().cloned().collect(),
                tools: Some(tool_defs.iter().cloned().collect()),
                tool_choice: None,
                temperature: Some(self.config.temperature),
                max_tokens: Some(self.config.max_tokens),
                stream: Some(true),
            };

            let mut reasoning_header_printed = false;
            let mut last_was_reasoning = false;
            let mut content_header_printed = false;
            let mut assembled_tool_calls = Vector::new();
            let mut last_metrics: Option<(usize, f64, f64)> = None;

            let assistant_msg_res = self
                .client
                .stream_completion(&req, |event| match event {
                    StreamEvent::Reasoning(text) => {
                        if !reasoning_header_printed && !text.trim().is_empty() {
                            println!(
                                "\n{}",
                                "🧠 ──────────────── Thought Process ────────────────"
                                    .bright_yellow()
                                    .bold()
                            );
                            reasoning_header_printed = true;
                        }
                        if reasoning_header_printed {
                            print!("{}", text.yellow().dimmed());
                            if text.ends_with(' ') || text.ends_with('\n') {
                            }
                            last_was_reasoning = true;
                        }
                    }
                    StreamEvent::Content(text) => {
                        if last_was_reasoning {
                            println!(
                                "\n{}",
                                "────────────────────────────────────────────────────"
                                    .bright_yellow()
                                    .dimmed()
                            );
                            print!("\n{}: ", "🤖 MRML".bold().green());
                            last_was_reasoning = false;
                            content_header_printed = true;
                        } else if !content_header_printed {
                            print!("\n{}: ", "🤖 MRML".bold().green());
                            content_header_printed = true;
                        }
                        print!("{}", text);
                        if text.ends_with(' ')
                            || text.ends_with('\n')
                            || text.ends_with('?')
                            || text.ends_with('.')
                        {
                        }
                    }
                    StreamEvent::ToolCallAssembled(tc) => {
                        assembled_tool_calls.push(tc);
                    }
                    StreamEvent::ToolExecuted { .. } => {}
                    StreamEvent::Metrics {
                        token_count,
                        elapsed_secs,
                        tokens_per_sec,
                    } => {
                        last_metrics = Some((token_count, elapsed_secs, tokens_per_sec));
                    }
                    StreamEvent::Finish(_) => {}
                })
                .await;

            if reasoning_header_printed && last_was_reasoning {
                println!(
                    "\n{}",
                    "────────────────────────────────────────────────────"
                        .bright_yellow()
                        .dimmed()
                );
            }

            if let Some((tokens, elapsed, tps)) = last_metrics {
                println!(
                    "{}",
                    format!(
                        "⚡ [{} tokens in {:.2}s | {:.1} tk/s]",
                        tokens, elapsed, tps
                    )
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

            let current_output_text = assistant_msg
                .get_text_content()
                .unwrap_or_default()
                .trim()
                .to_string();
            if !current_output_text.is_empty() {
                if let Some(ref prev) = last_model_output {
                    if prev.as_str() == current_output_text {
                        model_output_repeat_count += 1;
                    } else {
                        model_output_repeat_count = 1;
                    }
                } else {
                    model_output_repeat_count = 1;
                }
                last_model_output = Some(current_output_text.as_str().into());

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

            let tool_calls = assistant_msg
                .tool_calls
                .clone()
                .unwrap_or_else(|| assembled_tool_calls.into_iter().collect());
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

                let normalized = crate::client::normalize_relaxed_json(raw_args);
                let parsed_args: serde_json::Value = match serde_json::from_str(&normalized) {
                    Ok(val) => val,
                    Err(e) => {
                        let err_msg = format!("Failed to parse arguments JSON: {}", e);
                        self.history
                            .push(ChatMessage::tool(tool_call.id, name, err_msg));
                        continue;
                    }
                };

                let tool_opt = self.registry.get(name);
                let tool_result = match tool_opt {
                    Some(tool) => {
                        println!("⚡ Executing {}", name.cyan());
                        tool.execute(
                            &self.config.workspace_root,
                            parsed_args,
                        )
                        .await
                    }
                    None => Err(mrml_tools::tool_error(format!(
                        "Unknown tool requested: {}",
                        name
                    ))),
                };

                match tool_result {
                    Ok(output) => {
                        let display_output = if output.len() > 1000 {
                            format!(
                                "{}... (truncated)",
                                crate::markdown::truncate_utf8(&output, 1000)
                            )
                        } else {
                            output.clone()
                        };
                        println!("📥 Tool Output:\n{}", display_output.dimmed());
                        self.history.push(ChatMessage::tool(
                            tool_call.id.clone(),
                            name,
                            output.clone(),
                        ));
                    }
                    Err(e) => {
                        let err_msg = format!("Tool execution failed: {}", e);
                        println!("{}: {}", "✖".red(), err_msg);
                        self.history
                            .push(ChatMessage::tool(tool_call.id.clone(), name, err_msg));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        explicitly_requested_command, requests_live_local_time, verified_command_answer,
        verified_time_answer,
    };

    #[test]
    fn routes_only_live_clock_questions() {
        assert!(requests_live_local_time("What time is it?"));
        assert!(requests_live_local_time("Tell me the current time please"));
        assert!(!requests_live_local_time(
            "Explain algorithm time complexity"
        ));
        assert!(!requests_live_local_time(
            "What time did the meeting start?"
        ));
    }

    #[test]
    fn formats_only_verified_command_stdout() {
        let output =
            "Exit Code: 0\n--- STDOUT ---\n2026-08-18 12:49:24 -02:30\n--- STDERR ---\n(empty)";
        assert_eq!(
            verified_time_answer(output).as_deref(),
            Some("The current local time is **2026-08-18 12:49:24 -02:30**.")
        );
        assert!(verified_time_answer("Tool execution failed").is_none());
    }

    #[test]
    fn extracts_only_explicit_run_commands() {
        assert_eq!(explicitly_requested_command(
            "Use the run_command tool to execute Write-Output MRML_TOOL_OK, then tell me what printed"
        ).as_deref(), Some("Write-Output MRML_TOOL_OK"));
        assert!(explicitly_requested_command("Explain what this command does").is_none());
    }

    #[test]
    fn formats_verified_command_stdout() {
        let output = "Exit Code: 0\n--- STDOUT ---\nMRML_TOOL_OK\n--- STDERR ---\n(empty)";
        assert_eq!(
            verified_command_answer(output).as_deref(),
            Some("The command printed:\n\n```text\nMRML_TOOL_OK\n```")
        );
    }
}
