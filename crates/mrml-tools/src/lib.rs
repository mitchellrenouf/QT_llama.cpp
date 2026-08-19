#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(test, not(feature = "std")))]
extern crate std;

pub mod fixed_encoding;

#[cfg(feature = "runtime")]
pub mod browser;
#[cfg(feature = "runtime")]
pub mod desktop;
#[cfg(feature = "runtime")]
pub mod editor;
#[cfg(feature = "runtime")]
pub mod git;
#[cfg(feature = "runtime")]
pub mod html;
#[cfg(feature = "runtime")]
pub mod mcp;
#[cfg(feature = "runtime")]
pub mod media;
#[cfg(feature = "runtime")]
pub mod web;

#[cfg(feature = "runtime")]
pub mod diff;
#[cfg(feature = "runtime")]
pub mod encoding;
#[cfg(feature = "runtime")]
pub mod fs_walk;
#[cfg(feature = "runtime")]
pub mod markdown;
#[cfg(feature = "runtime")]
pub mod platform;
#[cfg(feature = "runtime")]
pub mod simple_regex;

#[cfg(feature = "runtime")]
use anyhow::Result;
#[cfg(feature = "runtime")]
use core::future::Future;
#[cfg(feature = "runtime")]
use core::pin::Pin;
#[cfg(feature = "runtime")]
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
#[cfg(feature = "runtime")]
use mrml_runtime::{Owned, Text};
#[cfg(all(feature = "runtime", not(feature = "std")))]
use mrml_runtime::Text as String;
#[cfg(feature = "std")]
use std::string::String;
#[cfg(feature = "std")]
use mrml_runtime::{Shared, Vector};
#[cfg(feature = "runtime")]
use serde_json::Value;
#[cfg(feature = "runtime")]
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

#[cfg(feature = "runtime")]
pub type ToolError = anyhow::Error;

#[cfg(feature = "runtime")]
pub fn tool_error(message: impl core::fmt::Display) -> ToolError {
    anyhow::message(message)
}

#[cfg(feature = "runtime")]
#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    pub name: Text,
    pub description: Text,
    pub parameters: serde_json::Value,
}

#[cfg(feature = "runtime")]
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub tool_type: Text,
    pub function: FunctionDefinition,
}

#[cfg(feature = "runtime")]
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

#[cfg(feature = "runtime")]
type ToolFuture<'a> = Pin<Owned<dyn Future<Output = Result<String>> + Send + 'a>>;

#[cfg(feature = "runtime")]
pub trait DynTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    fn execute<'a>(&'a self, workspace_root: &'a str, args: Value) -> ToolFuture<'a>;
    fn to_tool_definition(&self) -> ToolDefinition;
}

#[cfg(feature = "runtime")]
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

#[cfg(feature = "std")]
pub struct ToolRegistry {
    tools: Vector<Shared<dyn DynTool>>,
}

#[cfg(feature = "std")]
impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: Vector::new(),
        };

        // Editor & OS Tools
        registry.register(Shared::new(editor::ViewFileTool));
        registry.register(Shared::new(editor::WriteFileTool));
        registry.register(Shared::new(editor::ReplaceFileContentTool));
        registry.register(Shared::new(editor::ListDirTool));
        registry.register(Shared::new(editor::GrepSearchTool));
        registry.register(Shared::new(editor::RunCommandTool));

        // Git Safety Tools
        registry.register(Shared::new(git::GitCheckpointTool));
        registry.register(Shared::new(git::GitRollbackTool));
        registry.register(Shared::new(git::GitDiffTool));

        // Web Tools
        registry.register(Shared::new(web::WebSearchTool));
        registry.register(Shared::new(web::WebFetchTool));

        // Desktop Control Tools
        registry.register(Shared::new(desktop::TakeScreenshotTool));
        registry.register(Shared::new(desktop::OpenAppTool));

        // Browser Automation Tools
        registry.register(Shared::new(browser::BrowserOpenTool));
        registry.register(Shared::new(browser::BrowserGetContentTool));
        registry.register(Shared::new(browser::BrowserScreenshotTool));
        registry.register(Shared::new(browser::BrowserClickElementTool));
        registry.register(Shared::new(browser::BrowserClickTool));
        registry.register(Shared::new(browser::BrowserTypeTool));

        // Audio & Video Media Tools
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
        registry.register(Shared::new(editor::ViewFileTool));
        let after: Vec<_> = registry
            .definitions()
            .into_iter()
            .map(|item| item.function.name)
            .collect();
        assert_eq!(before, after);
    }
}
