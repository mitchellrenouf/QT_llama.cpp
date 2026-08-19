use core::fmt::Write as _;
use mrml_runtime::{Text, Vector};

#[derive(Debug, Clone, Default)]
pub struct WorkspaceRules {
    pub rule_sources: Vector<Text>,
    pub combined_instructions: Text,
}

impl WorkspaceRules {
    pub fn discover(workspace_root: &str) -> Self {
        let candidate_names = [
            "MRML.md",
            "AGENTS.md",
            "CLAUDE.md",
            ".mrml/rules",
            ".agent/rules",
            ".cursorrules",
        ];

        let mut sources = Vector::new();
        let mut instructions = Text::new();

        for name in &candidate_names {
            let candidate_path = mrml_runtime::join_path(workspace_root, name);
            if crate::platform::path_is_file(&candidate_path) {
                if let Ok(content) = mrml_runtime::read_file_text(&candidate_path) {
                    if !content.trim().is_empty() {
                        sources.push(candidate_path.clone());
                        write!(
                            instructions,
                            "\n--- PROJECT RULE ({}) ---\n{}\n",
                            name,
                            content.trim()
                        )
                        .expect("MRML rule text allocation failed");
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
    use std::fs;

    #[test]
    fn test_rules_discovery() {
        let unique_id = format!(
            "gemma_rules_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let temp_dir = std::env::temp_dir().join(unique_id);
        let _ = fs::create_dir_all(&temp_dir);
        let rule_file = temp_dir.join("MRML.md");
        fs::write(&rule_file, "Always write unit tests for new code.").unwrap();

        let rules = WorkspaceRules::discover(temp_dir.to_str().unwrap());
        assert!(rules.has_rules());
        assert!(
            rules
                .combined_instructions
                .contains("Always write unit tests")
        );

        let _ = fs::remove_file(rule_file);
        let _ = fs::remove_dir_all(temp_dir);
    }
}
