#![no_std]

use mrml_runtime::{Text, Vector};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileState {
    Added,
    Modified,
    Deleted,
    Renamed,
    Copied,
    Unmerged,
    Untracked,
    Ignored,
    Other,
}

impl FileState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
            Self::Renamed => "renamed",
            Self::Copied => "copied",
            Self::Unmerged => "conflict",
            Self::Untracked => "untracked",
            Self::Ignored => "ignored",
            Self::Other => "changed",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Change {
    pub index: char,
    pub worktree: char,
    pub path: Text,
    pub original_path: Option<Text>,
}

impl Change {
    pub fn staged(&self) -> bool {
        !matches!(self.index, ' ' | '?' | '!')
    }

    pub fn unstaged(&self) -> bool {
        !matches!(self.worktree, ' ' | '!')
    }

    pub fn state(&self) -> FileState {
        let code = if self.worktree != ' ' {
            self.worktree
        } else {
            self.index
        };
        match code {
            'A' => FileState::Added,
            'M' => FileState::Modified,
            'D' => FileState::Deleted,
            'R' => FileState::Renamed,
            'C' => FileState::Copied,
            'U' => FileState::Unmerged,
            '?' => FileState::Untracked,
            '!' => FileState::Ignored,
            _ => FileState::Other,
        }
    }
}

/// Parse `git status --porcelain=v1 -z`. Rename/copy records consume two NUL fields.
pub fn parse_porcelain(source: &[u8]) -> Vector<Change> {
    let fields = source.split(|byte| *byte == 0).collect::<Vector<_>>();
    let mut changes = Vector::new();
    let mut index = 0;
    while index < fields.len() {
        let field = fields[index];
        if field.len() < 4 {
            index += 1;
            continue;
        }
        let x = field[0] as char;
        let y = field[1] as char;
        let path = Text::from_utf8_lossy(&field[3..]);
        let rename = matches!(x, 'R' | 'C') || matches!(y, 'R' | 'C');
        let original_path = if rename && index + 1 < fields.len() && !fields[index + 1].is_empty() {
            index += 1;
            Some(Text::from_utf8_lossy(fields[index]))
        } else {
            None
        };
        changes.push(Change {
            index: x,
            worktree: y,
            path,
            original_path,
        });
        index += 1;
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_staged_unstaged_untracked_and_spaces() {
        let parsed = parse_porcelain(b"M  src/lib.rs\0 M notes with spaces.md\0?? new.txt\0");
        assert_eq!(parsed.len(), 3);
        assert!(parsed[0].staged());
        assert!(!parsed[0].unstaged());
        assert!(!parsed[1].staged());
        assert!(parsed[1].unstaged());
        assert_eq!(parsed[1].path, "notes with spaces.md");
        assert_eq!(parsed[2].state(), FileState::Untracked);
    }

    #[test]
    fn consumes_rename_pair_without_phantom_change() {
        let parsed = parse_porcelain(b"R  new name.rs\0old name.rs\0 M next.rs\0");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].path, "new name.rs");
        assert_eq!(parsed[0].original_path.as_deref(), Some("old name.rs"));
        assert_eq!(parsed[1].path, "next.rs");
    }
}
