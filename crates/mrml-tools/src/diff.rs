//! Line-oriented diff rendering used by editing tools.
use mrml_terminal_style::Colorize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Change<'a> {
    Delete(&'a str),
    Insert(&'a str),
    Equal(&'a str),
}

const MAX_LCS_CELLS: usize = 4_000_000;

fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

fn line_diff<'a>(old_text: &'a str, new_text: &'a str) -> Vec<Change<'a>> {
    let old = lines(old_text);
    let new = lines(new_text);
    let mut prefix = 0;
    while prefix < old.len() && prefix < new.len() && old[prefix] == new[prefix] {
        prefix += 1;
    }

    let mut old_end = old.len();
    let mut new_end = new.len();
    while old_end > prefix && new_end > prefix && old[old_end - 1] == new[new_end - 1] {
        old_end -= 1;
        new_end -= 1;
    }

    let mut changes = Vec::with_capacity(old.len() + new.len());
    changes.extend(old[..prefix].iter().copied().map(Change::Equal));
    let old_middle = &old[prefix..old_end];
    let new_middle = &new[prefix..new_end];

    if old_middle.len().saturating_mul(new_middle.len()) > MAX_LCS_CELLS {
        changes.extend(old_middle.iter().copied().map(Change::Delete));
        changes.extend(new_middle.iter().copied().map(Change::Insert));
    } else {
        let columns = new_middle.len() + 1;
        let mut lengths = vec![0u32; (old_middle.len() + 1) * columns];
        for old_index in (0..old_middle.len()).rev() {
            for new_index in (0..new_middle.len()).rev() {
                let cell = old_index * columns + new_index;
                lengths[cell] = if old_middle[old_index] == new_middle[new_index] {
                    lengths[(old_index + 1) * columns + new_index + 1] + 1
                } else {
                    lengths[(old_index + 1) * columns + new_index]
                        .max(lengths[old_index * columns + new_index + 1])
                };
            }
        }

        let (mut old_index, mut new_index) = (0, 0);
        while old_index < old_middle.len() && new_index < new_middle.len() {
            if old_middle[old_index] == new_middle[new_index] {
                changes.push(Change::Equal(old_middle[old_index]));
                old_index += 1;
                new_index += 1;
            } else if lengths[(old_index + 1) * columns + new_index]
                >= lengths[old_index * columns + new_index + 1]
            {
                changes.push(Change::Delete(old_middle[old_index]));
                old_index += 1;
            } else {
                changes.push(Change::Insert(new_middle[new_index]));
                new_index += 1;
            }
        }
        changes.extend(old_middle[old_index..].iter().copied().map(Change::Delete));
        changes.extend(new_middle[new_index..].iter().copied().map(Change::Insert));
    }

    changes.extend(old[old_end..].iter().copied().map(Change::Equal));
    changes
}

pub fn format_colorized_diff(file_path: &str, old_text: &str, new_text: &str) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "\n{}\n",
        format!("--- diff for '{}' ---", file_path).cyan().bold()
    ));

    for change in line_diff(old_text, new_text) {
        let line = match change {
            Change::Delete(line) | Change::Insert(line) | Change::Equal(line) => line,
        };
        match change {
            Change::Delete(_) => {
                output.push_str(&format!("- {}", line.red()));
            }
            Change::Insert(_) => {
                output.push_str(&format!("+ {}", line.green()));
            }
            Change::Equal(_) => {
                if line.len() > 120 {
                    output.push_str(&format!(
                        "  {}\n",
                        crate::markdown::truncate_utf8(&line, 120).dimmed()
                    ));
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

    #[test]
    fn line_diff_preserves_insertions_deletions_and_shared_lines() {
        assert_eq!(
            line_diff("one\ntwo\nfour\n", "one\nthree\nfour\n"),
            vec![
                Change::Equal("one\n"),
                Change::Delete("two\n"),
                Change::Insert("three\n"),
                Change::Equal("four\n"),
            ]
        );
    }

    #[test]
    fn line_diff_handles_final_lines_without_newlines() {
        assert_eq!(
            line_diff("alpha\nbeta", "alpha\ngamma"),
            vec![
                Change::Equal("alpha\n"),
                Change::Delete("beta"),
                Change::Insert("gamma"),
            ]
        );
    }
}
