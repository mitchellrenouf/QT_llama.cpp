#![cfg_attr(not(feature = "std"), no_std)]

pub mod fixed_encoding;

#[cfg(feature = "std")]
pub mod browser;
#[cfg(feature = "std")]
pub mod desktop;
#[cfg(feature = "std")]
pub mod editor;
#[cfg(feature = "std")]
pub mod git;
#[cfg(feature = "runtime")]
pub mod html;
#[cfg(feature = "std")]
pub mod mcp;
#[cfg(feature = "std")]
pub mod media;
#[cfg(feature = "std")]
pub mod web;

#[cfg(feature = "runtime")]
pub mod diff;
#[cfg(feature = "runtime")]
pub mod encoding;
#[cfg(feature = "std")]
pub mod fs_walk;
#[cfg(feature = "std")]
pub mod markdown;
#[cfg(feature = "std")]
pub mod platform;
#[cfg(feature = "runtime")]
pub mod simple_regex;

#[cfg(feature = "std")]
use anyhow::Result;
#[cfg(feature = "std")]
use core::future::Future;
#[cfg(feature = "std")]
use core::pin::Pin;
#[cfg(feature = "std")]
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
#[cfg(feature = "std")]
use serde_json::Value;
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::path::Path;
#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(feature = "std")]
pub fn block_on<F: Future>(future: F) -> F::Output {
    unsafe fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(core::ptr::null(), &VTABLE)
    }
    unsafe fn no_op(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);

    let raw = RawWaker::new(core::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut context = Context::from_waker(&waker);
    let mut future = core::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => mrml_runtime::yield_now(),
        }
    }
}

#[cfg(feature = "std")]
pub type ToolError = anyhow::Error;

#[cfg(feature = "std")]
pub fn tool_error(message: impl core::fmt::Display) -> ToolError {
    anyhow::message(message)
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[cfg(feature = "std")]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    fn execute(
        &self,
        workspace_root: &Path,
        args: Value,
    ) -> impl Future<Output = Result<String>> + Send;

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

#[cfg(feature = "std")]
type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

#[cfg(feature = "std")]
pub trait DynTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    fn execute<'a>(&'a self, workspace_root: &'a Path, args: Value) -> ToolFuture<'a>;
    fn to_tool_definition(&self) -> ToolDefinition;
}

#[cfg(feature = "std")]
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

    fn execute<'a>(&'a self, workspace_root: &'a Path, args: Value) -> ToolFuture<'a> {
        Box::pin(Tool::execute(self, workspace_root, args))
    }

    fn to_tool_definition(&self) -> ToolDefinition {
        Tool::to_tool_definition(self)
    }
}

#[cfg(feature = "std")]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn DynTool>>,
    order: Vec<String>,
}

#[cfg(feature = "std")]
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

    pub fn register(&mut self, tool: Arc<dyn DynTool>) {
        let name = tool.name().to_string();
        if !self.tools.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn DynTool>> {
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

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn allocation_free_executor_repolls_pending_futures() {
        struct PendingOnce(bool);
        impl core::future::Future for PendingOnce {
            type Output = usize;

            fn poll(
                mut self: core::pin::Pin<&mut Self>,
                context: &mut core::task::Context<'_>,
            ) -> core::task::Poll<Self::Output> {
                if self.0 {
                    core::task::Poll::Ready(42)
                } else {
                    self.0 = true;
                    context.waker().wake_by_ref();
                    core::task::Poll::Pending
                }
            }
        }

        assert_eq!(block_on(PendingOnce(false)), 42);
    }

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
