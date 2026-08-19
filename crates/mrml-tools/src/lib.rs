pub mod browser;
pub mod desktop;
pub mod editor;
pub mod git;
pub mod html;
pub mod mcp;
pub mod media;
pub mod web;

pub mod diff;
pub mod encoding;
pub mod fs_walk;
pub mod markdown;
pub mod platform;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub type ToolError = anyhow::Error;

pub fn tool_error(message: impl Into<String>) -> ToolError {
    anyhow::message(message)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn execute(&self, workspace_root: &Path, args: Value) -> Result<String>;

    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: self.name().to_string(),
                description: self.description().to_string(),
                parameters: self.parameters(),
            },
        }
    }
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
            order: Vec::new(),
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
        let name = tool.name().to_string();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| tool.to_tool_definition())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_exposes_every_tool_in_stable_prompt_order() {
        let expected = [
            "view_file",
            "write_file",
            "replace_file_content",
            "list_dir",
            "grep_search",
            "run_command",
            "git_checkpoint",
            "git_rollback",
            "git_diff",
            "web_search",
            "web_fetch",
            "take_screenshot",
            "open_app",
            "browser_open",
            "browser_get_content",
            "browser_screenshot",
            "browser_click_element",
            "browser_click",
            "browser_type",
            "speak_text",
            "record_audio",
            "capture_webcam",
            "record_screen_video",
        ];
        let registry = ToolRegistry::new();
        let definitions = registry.definitions();
        let actual: Vec<_> = definitions
            .iter()
            .map(|definition| definition.function.name.as_str())
            .collect();
        assert_eq!(actual, expected);
        for definition in definitions {
            assert!(!definition.function.description.trim().is_empty());
            assert_eq!(definition.function.parameters["type"], "object");
            assert!(registry.get(&definition.function.name).is_some());
        }
    }

    #[test]
    fn replacing_a_registered_tool_does_not_reorder_it() {
        let mut registry = ToolRegistry::new();
        let before: Vec<_> = registry
            .definitions()
            .into_iter()
            .map(|item| item.function.name)
            .collect();
        registry.register(Arc::new(editor::ViewFileTool));
        let after: Vec<_> = registry
            .definitions()
            .into_iter()
            .map(|item| item.function.name)
            .collect();
        assert_eq!(before, after);
    }
}
