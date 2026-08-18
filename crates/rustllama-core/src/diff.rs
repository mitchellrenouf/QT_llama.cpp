use colored::*;
use similar::{ChangeTag, TextDiff};

pub fn format_colorized_diff(file_path: &str, old_text: &str, new_text: &str) -> String {
    let diff = TextDiff::from_lines(old_text, new_text);
    let mut output = String::new();

    output.push_str(&format!(
        "\n{}\n",
        format!("--- diff for '{}' ---", file_path).cyan().bold()
    ));

    for change in diff.iter_all_changes() {
        let line = change.to_string();
        match change.tag() {
            ChangeTag::Delete => {
                output.push_str(&format!("- {}", line.red()));
            }
            ChangeTag::Insert => {
                output.push_str(&format!("+ {}", line.green()));
            }
            ChangeTag::Equal => {
                if line.len() > 120 {
                    output.push_str(&format!("  {}\n", crate::markdown::truncate_utf8(&line, 120).dimmed()));
                } else {
                    output.push_str(&format!("  {}", line.dimmed()));
                }
            }
        }
    }

    output.push_str(&format!("{}\n", "-----------------------".cyan().bold()));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_colorized_diff() {
        let old = "fn main() {\n    println!(\"Hello\");\n}\n";
        let new = "fn main() {\n    println!(\"Hello World\");\n}\n";
        let diff = format_colorized_diff("src/main.rs", old, new);

        assert!(diff.contains("Hello"));
        assert!(diff.contains("Hello World"));
        assert!(diff.contains("--- diff for 'src/main.rs' ---"));
    }
}
