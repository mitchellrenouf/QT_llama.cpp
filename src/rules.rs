use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct WorkspaceRules {
    pub rule_sources: Vec<PathBuf>,
    pub combined_instructions: String,
}

impl WorkspaceRules {
    pub fn discover(workspace_root: &Path) -> Self {
        let candidate_names = [
            "GEMMA.md",
            "AGENTS.md",
            "CLAUDE.md",
            ".gemma/rules",
            ".agent/rules",
            ".cursorrules",
        ];

        let mut sources = Vec::new();
        let mut instructions = String::new();

        for name in &candidate_names {
            let candidate_path = workspace_root.join(name);
            if candidate_path.is_file() {
                if let Ok(content) = fs::read_to_string(&candidate_path) {
                    if !content.trim().is_empty() {
                        sources.push(candidate_path.clone());
                        instructions.push_str(&format!(
                            "\n--- PROJECT RULE ({}) ---\n{}\n",
                            name,
                            content.trim()
                        ));
                    }
                }
            }
        }

        Self {
            rule_sources: sources,
            combined_instructions: instructions,
        }
    }

    pub fn has_rules(&self) -> bool {
        !self.rule_sources.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rules_discovery() {
        let unique_id = format!("gemma_rules_test_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
        let temp_dir = std::env::temp_dir().join(unique_id);
        let _ = fs::create_dir_all(&temp_dir);
        let rule_file = temp_dir.join("GEMMA.md");
        fs::write(&rule_file, "Always write unit tests for new code.").unwrap();

        let rules = WorkspaceRules::discover(&temp_dir);
        assert!(rules.has_rules());
        assert!(rules.combined_instructions.contains("Always write unit tests"));

        let _ = fs::remove_file(rule_file);
        let _ = fs::remove_dir_all(temp_dir);
    }
}
