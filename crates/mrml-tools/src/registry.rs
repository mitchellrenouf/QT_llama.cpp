use crate::{browser, desktop, editor, git, media, web};
use core::future::Future;
use core::pin::Pin;
use mrml_error::Result;
use mrml_runtime::Text as String;
use mrml_runtime::{Owned, Shared, Text, Vector};
use serde_json::Value;

pub type ToolError = mrml_error::Error;

pub fn tool_error(message: impl core::fmt::Display) -> ToolError {
    mrml_error::message(message)
}

#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: Text,
    pub description: Text,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub tool_type: Text,
    pub function: FunctionDefinition,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    fn execute(
        &self,
        workspace_root: &str,
        args: Value,
    ) -> impl Future<Output = Result<String>> + Send;
    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: self.name().into(),
                description: self.description().into(),
                parameters: self.parameters(),
            },
        }
    }
}

type ToolFuture<'a> = Pin<Owned<dyn Future<Output = Result<String>> + Send + 'a>>;

pub trait DynTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    fn execute<'a>(&'a self, workspace_root: &'a str, args: Value) -> ToolFuture<'a>;
    fn to_tool_definition(&self) -> ToolDefinition;
}

impl<T: Tool> DynTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }
    fn description(&self) -> &str {
        Tool::description(self)
    }
    fn parameters(&self) -> Value {
        Tool::parameters(self)
    }
    fn execute<'a>(&'a self, workspace_root: &'a str, args: Value) -> ToolFuture<'a> {
        unsafe { Pin::new_unchecked(Owned::new(Tool::execute(self, workspace_root, args))) }
    }
    fn to_tool_definition(&self) -> ToolDefinition {
        Tool::to_tool_definition(self)
    }
}

pub struct ToolRegistry {
    tools: Vector<Shared<dyn DynTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: Vector::new(),
        };
        registry.register(Shared::new(editor::ViewFileTool));
        registry.register(Shared::new(editor::WriteFileTool));
        registry.register(Shared::new(editor::ReplaceFileContentTool));
        registry.register(Shared::new(editor::ListDirTool));
        registry.register(Shared::new(editor::GrepSearchTool));
        registry.register(Shared::new(editor::RunCommandTool));
        registry.register(Shared::new(git::GitCheckpointTool));
        registry.register(Shared::new(git::GitRollbackTool));
        registry.register(Shared::new(git::GitDiffTool));
        registry.register(Shared::new(web::WebSearchTool));
        registry.register(Shared::new(web::WebFetchTool));
        registry.register(Shared::new(desktop::TakeScreenshotTool));
        registry.register(Shared::new(desktop::OpenAppTool));
        registry.register(Shared::new(browser::BrowserOpenTool));
        registry.register(Shared::new(browser::BrowserGetContentTool));
        registry.register(Shared::new(browser::BrowserScreenshotTool));
        registry.register(Shared::new(browser::BrowserClickElementTool));
        registry.register(Shared::new(browser::BrowserClickTool));
        registry.register(Shared::new(browser::BrowserTypeTool));
        registry.register(Shared::new(media::SpeakTextTool));
        registry.register(Shared::new(media::RecordAudioTool));
        registry.register(Shared::new(media::CaptureWebcamTool));
        registry.register(Shared::new(media::RecordScreenVideoTool));
        registry
    }
    pub fn register(&mut self, tool: Shared<dyn DynTool>) {
        if let Some(existing) = self
            .tools
            .iter_mut()
            .find(|existing| existing.name() == tool.name())
        {
            existing.replace(tool);
        } else {
            self.tools.push(tool);
        }
    }
    pub fn get(&self, name: &str) -> Option<Shared<dyn DynTool>> {
        self.tools.iter().find(|tool| tool.name() == name).cloned()
    }
    pub fn definitions(&self) -> Vector<ToolDefinition> {
        self.tools
            .iter()
            .map(|tool| tool.to_tool_definition())
            .collect()
    }
}
