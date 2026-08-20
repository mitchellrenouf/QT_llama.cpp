#![no_std]

pub mod fixed_encoding;

#[cfg(feature = "runtime")]
pub mod browser;
#[cfg(feature = "runtime")]
pub mod desktop;
#[cfg(feature = "runtime")]
pub mod diff;
#[cfg(feature = "runtime")]
pub mod editor;
#[cfg(feature = "runtime")]
pub mod encoding;
#[cfg(feature = "runtime")]
pub mod executor;
#[cfg(feature = "runtime")]
pub mod fs_walk;
#[cfg(feature = "runtime")]
pub mod git;
#[cfg(feature = "runtime")]
pub mod html;
#[cfg(feature = "runtime")]
pub mod markdown;
#[cfg(feature = "runtime")]
pub mod mcp;
#[cfg(feature = "runtime")]
pub mod media;
#[cfg(feature = "runtime")]
pub mod platform;
#[cfg(feature = "runtime")]
pub mod registry;
#[cfg(feature = "runtime")]
pub mod simple_regex;
#[cfg(feature = "runtime")]
pub mod web;

#[cfg(feature = "runtime")]
pub use executor::block_on;
#[cfg(feature = "runtime")]
pub use registry::{
    DynTool, FunctionDefinition, Tool, ToolDefinition, ToolError, ToolRegistry, tool_error,
};

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;
    use mrml_runtime::{Shared, Vector};

    #[test]
    fn registry_exposes_every_tool_in_stable_prompt_order() {
        let expected = [
            "view_file", "write_file", "replace_file_content", "list_dir", "grep_search",
            "run_command", "git_checkpoint", "git_rollback", "git_diff", "web_search",
            "web_fetch", "take_screenshot", "open_app", "browser_open",
            "browser_get_content", "browser_screenshot", "browser_click_element",
            "browser_click", "browser_type", "speak_text", "record_audio",
            "capture_webcam", "record_screen_video",
        ];
        let registry = ToolRegistry::new();
        let definitions = registry.definitions();
        let actual: Vector<_> = definitions.iter().map(|definition| definition.function.name.as_str()).collect();
        assert_eq!(actual, expected.as_slice());
        for definition in &definitions {
            assert!(!definition.function.description.trim().is_empty());
            assert_eq!(definition.function.parameters["type"], "object");
            assert!(registry.get(&definition.function.name).is_some());
        }
    }

    #[test]
    fn replacing_a_registered_tool_does_not_reorder_it() {
        let mut registry = ToolRegistry::new();
        let before: Vector<_> = registry.definitions().into_iter().map(|item| item.function.name).collect();
        registry.register(Shared::new(editor::ViewFileTool));
        let after: Vector<_> = registry.definitions().into_iter().map(|item| item.function.name).collect();
        assert_eq!(before, after);
    }
}
