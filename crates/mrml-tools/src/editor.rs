use crate::Tool;
use crate::diff::format_colorized_diff;
use anyhow::{Result, anyhow};
use mrml_runtime::{Vector, mrml_print as print};
use serde_json::json;
use mrml_runtime::Command;

pub struct ViewFileTool;
impl Tool for ViewFileTool {
    fn name(&self) -> &'static str {
        "view_file"
    }

    fn description(&self) -> &'static str {
        "View contents of a file in the workspace, with optional start_line and end_line parameters."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the target file from workspace root"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed start line number"
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed end line number"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing path"))?;
        let full_path = mrml_runtime::join_path(workspace_root, path_str);
        if !mrml_runtime::path_is_file(&full_path) {
            return Err(anyhow!("File not found: {}", path_str));
        }

        let content = mrml_runtime::read_file_text(&full_path)?;
        let lines: Vector<&str> = content.lines().collect();

        let start_line = args["start_line"].as_u64().map(|v| v as usize).unwrap_or(1);
        let end_line = args["end_line"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(lines.len());

        let start = if start_line > 0 { start_line - 1 } else { 0 };
        let end = end_line.min(lines.len());

        if start >= lines.len() {
            return Ok(format!("File {} has only {} lines.", path_str, lines.len()));
        }

        let mut output = String::new();
        for idx in start..end {
            output.push_str(&format!("{:4} | {}\n", idx + 1, lines[idx]));
        }

        Ok(output)
    }
}

pub struct WriteFileTool;
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn description(&self) -> &'static str {
        "Create a new file or completely rewrite an existing file in the workspace."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to target file"
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing path"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing content"))?;
        let full_path = mrml_runtime::join_path(workspace_root, path_str);

        let old_content = mrml_runtime::read_file_text(&full_path).unwrap_or_default();

        if let Some(parent) = mrml_runtime::parent_path(&full_path) {
            mrml_runtime::create_dir_all(parent)?;
        }

        mrml_runtime::write_file(&full_path, content.as_bytes())?;

        let diff_str = format_colorized_diff(path_str, &old_content, content);
        print!("{}", diff_str);

        Ok(format!(
            "Successfully wrote {} bytes to file '{}'.",
            content.len(),
            path_str
        ))
    }
}

pub struct ReplaceFileContentTool;
impl Tool for ReplaceFileContentTool {
    fn name(&self) -> &'static str {
        "replace_file_content"
    }

    fn description(&self) -> &'static str {
        "Replace a specific contiguous block of text in an existing file."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to target file"
                },
                "target_content": {
                    "type": "string",
                    "description": "Exact text substring to search for and replace"
                },
                "replacement_content": {
                    "type": "string",
                    "description": "New replacement text"
                }
            },
            "required": ["path", "target_content", "replacement_content"]
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing path"))?;
        let target = args["target_content"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing target_content"))?;
        let replacement = args["replacement_content"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing replacement_content"))?;
        let full_path = mrml_runtime::join_path(workspace_root, path_str);

        if !mrml_runtime::path_is_file(&full_path) {
            return Err(anyhow!("File not found: {}", path_str));
        }

        let content = mrml_runtime::read_file_text(&full_path)?;
        if !content.contains(target) {
            return Err(anyhow!(
                "Target content not found in file '{}'. Ensure exact match including whitespace.",
                path_str
            ));
        }

        let updated = content.replacen(target, replacement, 1);
        mrml_runtime::write_file(&full_path, updated.as_bytes())?;

        let diff_str = format_colorized_diff(path_str, &content, &updated);
        print!("{}", diff_str);

        Ok(format!(
            "Successfully replaced target content in file '{}'.",
            path_str
        ))
    }
}

pub struct ListDirTool;
impl Tool for ListDirTool {
    fn name(&self) -> &'static str {
        "list_dir"
    }

    fn description(&self) -> &'static str {
        "List files and directories in a given workspace path."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path (defaults to workspace root if omitted)"
                }
            }
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let rel_path = args["path"].as_str().unwrap_or(".");
        let full_path = mrml_runtime::join_path(workspace_root, rel_path);
        if !mrml_runtime::path_is_directory(&full_path) {
            return Err(anyhow!("Directory not found: {}", rel_path));
        }

        let mut output = String::new();
        let entries = mrml_runtime::read_directory(&full_path)?;

        output.push_str(&format!("Contents of '{}':\n", rel_path));
        for entry in entries {
            let file_name = entry.name;

            if file_name.starts_with('.') || file_name == "target" {
                continue;
            }

            let kind = if entry.is_directory { "DIR " } else { "FILE" };
            let size = if !entry.is_directory && !entry.is_symlink {
                let entry_path = mrml_runtime::join_path(&full_path, &file_name);
                mrml_runtime::File::open(&entry_path)
                .and_then(|file| file.len())
                .map(|length| format!(" ({} bytes)", length))
                .unwrap_or_default()
            } else {
                String::new()
            };

            output.push_str(&format!(" [{}] {}{}\n", kind, file_name, size));
        }

        Ok(output)
    }
}

pub struct GrepSearchTool;
impl Tool for GrepSearchTool {
    fn name(&self) -> &'static str {
        "grep_search"
    }

    fn description(&self) -> &'static str {
        "Search for text or regex pattern across workspace files."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "String or regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Optional subdirectory path to search within"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let query_str = args["query"]
            .as_str()
            .ok_or_else(|| anyhow!("Missing query"))?;
        let sub_path = args["path"].as_str().unwrap_or(".");
        let search_path = mrml_runtime::join_path(workspace_root, sub_path);
        let pattern = crate::simple_regex::Regex::new(query_str)?;
        let mut matches = Vec::new();
        for path in crate::fs_walk::paths(&search_path) {
            if !mrml_runtime::path_is_file(&path) {
                continue;
            }
            let rel = path
                .strip_prefix(workspace_root)
                .map(|path| path.trim_start_matches(['/', '\\']))
                .unwrap_or(&path);
            if rel
                .split(['/', '\\'])
                .any(|part| part == ".git" || part == "target")
            {
                continue;
            }
            if let Ok(content) = mrml_runtime::read_file_text(&path) {
                for (line_no, line) in content.lines().enumerate() {
                    if pattern.is_match(line) {
                        matches.push(format!("{}:{}: {}", rel, line_no + 1, line.trim()));
                        if matches.len() == 50 {
                            matches.push("... (results truncated to 50 matches)".to_string());
                            break;
                        }
                    }
                }
            }
            if matches.len() > 50 {
                break;
            }
        }

        if matches.is_empty() {
            Ok(format!("No matches found for query '{}'", query_str))
        } else {
            Ok(matches.join("\n"))
        }
    }
}

pub struct RunCommandTool;
impl Tool for RunCommandTool {
    fn name(&self) -> &'static str {
        "run_command"
    }

    fn description(&self) -> &'static str {
        "Execute a terminal shell command in the workspace. Use this tool for live system information such as the current local time (Get-Date on Windows; date on Linux/macOS), builds, tests, and other command-line tasks."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command_line": {
                    "type": "string",
                    "description": "Command string to execute (e.g. 'cargo check', 'cargo test', 'git status')"
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory relative to workspace root"
                }
            },
            "required": ["command_line"]
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let cmd_str = args["command_line"]
            .as_str()
            .or_else(|| args["command"].as_str())
            .ok_or_else(|| anyhow!("Missing command_line (or command)"))?;
        let cwd_str = args["cwd"].as_str().unwrap_or(".");
        let exec_dir = mrml_runtime::join_path(workspace_root, cwd_str);

        #[cfg(windows)]
        let output = Command::new("powershell.exe")
            .args([
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                cmd_str,
            ])
            .current_dir(exec_dir.as_str())
            .output()?;

        #[cfg(not(windows))]
        let output = Command::new("sh")
            .args(["-c", cmd_str])
            .current_dir(exec_dir.as_str())
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let exit_code = output.status.code();

        Ok(format!(
            "Exit Code: {}\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
            exit_code,
            if stdout.trim().is_empty() {
                "(empty)"
            } else {
                stdout.trim()
            },
            if stderr.trim().is_empty() {
                "(empty)"
            } else {
                stderr.trim()
            }
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};

    static WORKSPACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_workspace() -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mrml-tools-editor-{}-{}-{}",
            std::process::id(),
            crate::platform::unix_timestamp_millis(),
            WORKSPACE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn file_tools_write_view_replace_list_and_search() {
        crate::block_on(async {
            let root = test_workspace();
            let write = WriteFileTool
                .execute(
                    root.to_str().unwrap(),
                    json!({"path":"src/note.txt", "content":"alpha\nbeta\n"}),
                )
                .await
                .unwrap();
            assert!(write.contains("11 bytes"));

            let viewed = ViewFileTool
                .execute(
                    root.to_str().unwrap(),
                    json!({"path":"src/note.txt", "start_line":2, "end_line":2}),
                )
                .await
                .unwrap();
            assert!(viewed.contains("2 | beta"));

            ReplaceFileContentTool
            .execute(
                root.to_str().unwrap(),
                json!({
                    "path":"src/note.txt", "target_content":"beta", "replacement_content":"gamma"
                }),
            )
            .await
            .unwrap();
            let listed = ListDirTool
                .execute(root.to_str().unwrap(), json!({"path":"src"}))
                .await
                .unwrap();
            assert!(listed.contains("note.txt"));
            let matches = GrepSearchTool
                .execute(root.to_str().unwrap(), json!({"query":"g.mm.", "path":"src"}))
                .await
                .unwrap();
            assert!(matches.contains("gamma"));
            let _ = std::fs::remove_dir_all(root);
        });
    }

    #[test]
    fn run_command_captures_stdout_and_exit_status() {
        crate::block_on(async {
            let root = test_workspace();
            #[cfg(windows)]
            let command = "Write-Output tool-ok";
            #[cfg(not(windows))]
            let command = "printf tool-ok";
            let output = RunCommandTool
                .execute(root.to_str().unwrap(), json!({"command_line": command}))
                .await
                .unwrap();
            assert!(output.contains("Exit Code: 0"));
            assert!(output.contains("tool-ok"));
            let _ = std::fs::remove_dir_all(root);
        });
    }
}
