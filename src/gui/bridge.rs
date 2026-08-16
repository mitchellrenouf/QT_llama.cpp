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
    Error(String),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum UiCommand {
    SendMessage(String),
    SwitchMode(AgentMode),
    ToggleSpeech,
    ClearHistory,
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
                UiCommand::Stop => {
                    break;
                }
            }
        }
        Ok(())
    }
}
