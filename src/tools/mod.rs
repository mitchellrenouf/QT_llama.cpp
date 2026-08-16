pub mod browser;
pub mod desktop;
pub mod editor;
pub mod git;
pub mod media;
pub mod web;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::client::ToolDefinition;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn parameters(&self) -> Value;
    async fn execute(&self, workspace_root: &Path, args: Value) -> Result<String>;

    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::client::FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters(),
            },
        }
    }
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };

        // Editor & OS Tools
        registry.register(Arc::new(editor::ViewFileTool));
        registry.register(Arc::new(editor::WriteFileTool));
        registry.register(Arc::new(editor::ReplaceFileContentTool));
        registry.register(Arc::new(editor::ListDirTool));
        registry.register(Arc::new(editor::GrepSearchTool));
        registry.register(Arc::new(editor::RunCommandTool));

        // Git Safety Tools
        registry.register(Arc::new(git::GitCheckpointTool));
        registry.register(Arc::new(git::GitRollbackTool));
        registry.register(Arc::new(git::GitDiffTool));

        // Web Tools
        registry.register(Arc::new(web::WebSearchTool));
        registry.register(Arc::new(web::WebFetchTool));

        // Desktop Control Tools
        registry.register(Arc::new(desktop::TakeScreenshotTool));
        registry.register(Arc::new(desktop::OpenAppTool));

        // Browser Automation Tools
        registry.register(Arc::new(browser::BrowserOpenTool));
        registry.register(Arc::new(browser::BrowserGetContentTool));
        registry.register(Arc::new(browser::BrowserScreenshotTool));
        registry.register(Arc::new(browser::BrowserClickElementTool));
        registry.register(Arc::new(browser::BrowserClickTool));
        registry.register(Arc::new(browser::BrowserTypeTool));

        // Audio & Video Media Tools
        registry.register(Arc::new(media::SpeakTextTool));
        registry.register(Arc::new(media::RecordAudioTool));
        registry.register(Arc::new(media::CaptureWebcamTool));
        registry.register(Arc::new(media::RecordScreenVideoTool));

        registry
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.to_tool_definition()).collect()
    }
}
