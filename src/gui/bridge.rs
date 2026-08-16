use crate::agent::GemmaAgent;
use crate::config::AgentMode;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AgentEvent {
    StreamToken(String),
    ThinkingToken(String),
    ToolStarted { name: String, args: String },
    ToolFinished { name: String, result: String },
    TurnDone { total_tokens: usize },
    DownloadProgress {
        message: String,
        progress: f32,
        file_idx: usize,
        total_files: usize,
    },
    ModelLoaded { model_name: String },
    Error(String),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum UiCommand {
    SendMessage(String),
    SwitchMode(AgentMode),
    ToggleSpeech,
    ClearHistory,
    LoadHfModel(String),
    Stop,
}

#[allow(dead_code)]
pub struct AgentBridge {
    pub agent: Arc<Mutex<GemmaAgent>>,
    pub event_tx: mpsc::UnboundedSender<AgentEvent>,
    pub cmd_rx: mpsc::UnboundedReceiver<UiCommand>,
}

impl AgentBridge {
    pub fn create_channel_pair(
        agent: GemmaAgent,
    ) -> (
        Self,
        mpsc::UnboundedSender<UiCommand>,
        mpsc::UnboundedReceiver<AgentEvent>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<UiCommand>();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<AgentEvent>();

        let bridge = AgentBridge {
            agent: Arc::new(Mutex::new(agent)),
            event_tx,
            cmd_rx,
        };

        (bridge, cmd_tx, event_rx)
    }

    pub async fn run_loop(mut self) -> Result<()> {
        while let Some(cmd) = self.cmd_rx.recv().await {
            match cmd {
                UiCommand::SendMessage(user_text) => {
                    let agent_clone = self.agent.clone();
                    let tx = self.event_tx.clone();

                    let mut agent_lock = agent_clone.lock().await;
                    match agent_lock.process_user_request(&user_text).await {
                        Ok(_) => {
                            let total = agent_lock.estimate_tokens();
                            let _ = tx.send(AgentEvent::TurnDone { total_tokens: total });
                        }
                        Err(e) => {
                            let _ = tx.send(AgentEvent::Error(e.to_string()));
                        }
                    }
                }
                UiCommand::SwitchMode(mode) => {
                    let mut agent_lock = self.agent.lock().await;
                    agent_lock.set_mode(mode);
                }
                UiCommand::ToggleSpeech => {
                    let mut agent_lock = self.agent.lock().await;
                    agent_lock.toggle_speech();
                }
                UiCommand::ClearHistory => {
                    let mut agent_lock = self.agent.lock().await;
                    agent_lock.reset_context();
                }
                UiCommand::LoadHfModel(repo_spec) => {
                    let tx = self.event_tx.clone();
                    let agent_clone = self.agent.clone();
                    tokio::spawn(async move {
                        match crate::hf::HfModelSpec::parse(&repo_spec) {
                            Ok(spec) => {
                                let tx_progress = tx.clone();
                                let res = crate::hf::resolve_or_fetch_hf_model(&spec, move |msg, p, file_idx, total_files| {
                                    let _ = tx_progress.send(AgentEvent::DownloadProgress {
                                        message: msg.to_string(),
                                        progress: p,
                                        file_idx,
                                        total_files,
                                    });
                                }).await;

                                match res {
                                    Ok(model_files) => {
                                        let mut agent_lock = agent_clone.lock().await;
                                        if let Err(e) = agent_lock.reload_model(&model_files.primary_entry_file) {
                                            let _ = tx.send(AgentEvent::Error(format!("Failed to load GGUF model: {}", e)));
                                        } else {
                                            let _ = tx.send(AgentEvent::ModelLoaded {
                                                model_name: spec.repo_id.clone(),
                                            });
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(AgentEvent::Error(format!("Hugging Face model resolution error: {}", e)));
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(AgentEvent::Error(format!("Invalid Hugging Face spec: {}", e)));
                            }
                        }
                    });
                }
                UiCommand::Stop => {
                    break;
                }
            }
        }
        Ok(())
    }
}
