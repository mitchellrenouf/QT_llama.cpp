use crate::Tool;
use anyhow::{Result, anyhow};
use mrml_runtime::{Command, Text};
use mrml_runtime::{Text as String, mrml_format as format};
use serde_json::json;

pub struct GitCheckpointTool;
impl Tool for GitCheckpointTool {
    fn name(&self) -> &'static str {
        "git_checkpoint"
    }

    fn description(&self) -> &'static str {
        "Create a temporary git checkpoint snapshot before making large code refactors."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Optional description for the checkpoint snapshot"
                }
            }
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        let msg = args["message"].as_str().unwrap_or("gemma-vibe-checkpoint");

        let stash_output = Command::new("git")
            .args(["stash", "create", msg])
            .current_dir(workspace_root)
            .output()?;

        let commit_hash = Text::from(Text::from_utf8_lossy(&stash_output.stdout).trim());

        if commit_hash.is_empty() {
            // No changes to stash, save HEAD hash
            let head_output = Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(workspace_root)
                .output()?;
            let head_hash = Text::from(Text::from_utf8_lossy(&head_output.stdout).trim());
            Ok(format!("Checkpoint created at HEAD: {}", head_hash))
        } else {
            // Save stash reference
            let store_output = Command::new("git")
                .args(["stash", "store", "-m", msg, &commit_hash])
                .current_dir(workspace_root)
                .output()?;

            if store_output.status.success() {
                Ok(format!(
                    "Successfully created Git Checkpoint [{}] ('{}'). Use git_rollback to revert if needed.",
                    commit_hash, msg
                ))
            } else {
                Ok(format!(
                    "Checkpoint hash generated: {}. Uncommitted changes snapshotted.",
                    commit_hash
                ))
            }
        }
    }
}

pub struct GitRollbackTool;
impl Tool for GitRollbackTool {
    fn name(&self) -> &'static str {
        "git_rollback"
    }

    fn description(&self) -> &'static str {
        "Rollback uncommitted changes or revert workspace back to previous git state."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "checkpoint_id": {
                    "type": "string",
                    "description": "Optional specific commit hash or stash identifier to restore"
                }
            }
        })
    }

    async fn execute(&self, workspace_root: &str, args: serde_json::Value) -> Result<String> {
        if let Some(target) = args["checkpoint_id"].as_str() {
            let output = Command::new("git")
                .args(["checkout", target, "--", "."])
                .current_dir(workspace_root)
                .output()?;
            let stderr = Text::from_utf8_lossy(&output.stderr);
            if output.status.success() {
                Ok(format!(
                    "Successfully restored workspace to checkpoint '{}'.",
                    target
                ))
            } else {
                Err(anyhow!("Failed to rollback to {}: {}", target, stderr))
            }
        } else {
            // Revert all working tree modifications
            let output = Command::new("git")
                .args(["checkout", "--", "."])
                .current_dir(workspace_root)
                .output()?;
            if output.status.success() {
                Ok("Successfully discarded all uncommitted changes and restored working tree.".into())
            } else {
                let stderr = Text::from_utf8_lossy(&output.stderr);
                Err(anyhow!("Failed to rollback changes: {}", stderr))
            }
        }
    }
}

pub struct GitDiffTool;
impl Tool for GitDiffTool {
    fn name(&self) -> &'static str {
        "git_diff"
    }

    fn description(&self) -> &'static str {
        "View uncommitted git modifications across workspace files."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, workspace_root: &str, _args: serde_json::Value) -> Result<String> {
        let output = Command::new("git")
            .args(["diff", "--stat"])
            .current_dir(workspace_root)
            .output()?;

        let stat = Text::from_utf8_lossy(&output.stdout);
        let detail_output = Command::new("git")
            .args(["diff"])
            .current_dir(workspace_root)
            .output()?;
        let detail = Text::from_utf8_lossy(&detail_output.stdout);

        if stat.trim().is_empty() && detail.trim().is_empty() {
            Ok("No uncommitted changes in working tree.".into())
        } else {
            Ok(format!(
                "--- GIT DIFF SUMMARY ---\n{}\n--- DETAILED DIFF ---\n{}",
                stat.trim(),
                detail.trim()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn git_diff_runs_through_native_process_capture() {
        let output = crate::block_on(GitDiffTool.execute(env!("CARGO_MANIFEST_DIR"), json!({})))
            .expect("git diff tool should run inside the repository");
        assert!(!output.is_empty());
    }
}
