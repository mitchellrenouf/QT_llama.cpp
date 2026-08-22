use core::fmt;
use mrml_runtime::{
    FileError, Text, Vector, canonical_path, create_dir_all, join_path, parent_path, path_exists,
    path_is_directory, path_is_file, read_directory, read_file_bounded, read_file_text_bounded,
    write_file,
};

use crate::{Index, IndexError, ObjectId, ObjectKind, encode_loose_object};

const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;
const MAX_WORKTREE_FILE: usize = 256 * 1024 * 1024;
const MAX_WORKTREE_ENTRIES: usize = 1_000_000;
const MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Repository { pub worktree: Text, pub git_dir: Text }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeChangeKind { Modified, Deleted, Untracked }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeChange { pub path: Text, pub kind: NativeChangeKind }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepositoryError {
    File(FileError), Index(IndexError), NotRepository, AlreadyExists, UnsupportedLayout,
    InvalidHead, InvalidReference, TooManyFiles, TooDeep, UnsupportedFileType, FileTooLarge,
}

impl Repository {
    pub fn discover(start: &str) -> Result<Self, RepositoryError> {
        let resolved = canonical_path(start)?;
        let mut current = if path_is_directory(&resolved) { resolved } else {
            parent_path(&resolved).ok_or(RepositoryError::NotRepository)?.into()
        };
        loop {
            let marker = join_path(&current, ".git");
            if path_is_directory(&marker) {
                return Ok(Self { worktree: current, git_dir: marker });
            }
            if path_exists(&marker) { return Err(RepositoryError::UnsupportedLayout); }
            let Some(parent) = parent_path(&current) else { break };
            if current == parent { break; }
            current = parent.into();
        }
        Err(RepositoryError::NotRepository)
    }

    pub fn init(path: &str) -> Result<Self, RepositoryError> {
        create_dir_all(path)?;
        let worktree = canonical_path(path)?;
        let git_dir = join_path(&worktree, ".git");
        if path_exists(&git_dir) { return Err(RepositoryError::AlreadyExists); }
        create_dir_all(&join_path(&git_dir, "objects/info"))?;
        create_dir_all(&join_path(&git_dir, "objects/pack"))?;
        create_dir_all(&join_path(&git_dir, "refs/heads"))?;
        create_dir_all(&join_path(&git_dir, "refs/tags"))?;
        write_file(&join_path(&git_dir, "HEAD"), b"ref: refs/heads/main\n")?;
        write_file(&join_path(&git_dir, "config"), b"[core]\n\trepositoryformatversion = 0\n\tbare = false\n")?;
        write_file(&join_path(&git_dir, "description"), b"Unnamed MRML repository\n")?;
        Ok(Self { worktree, git_dir })
    }

    pub fn current_branch(&self) -> Result<Option<Text>, RepositoryError> {
        let head = read_file_text_bounded(&join_path(&self.git_dir, "HEAD"), 4096)?;
        let head = head.trim();
        if let Some(reference) = head.strip_prefix("ref: ") {
            validate_reference(reference)?;
            return Ok(reference.strip_prefix("refs/heads/").map(Into::into));
        }
        ObjectId::parse(head).map(|_| None).ok_or(RepositoryError::InvalidHead)
    }

    pub fn head(&self) -> Result<Option<ObjectId>, RepositoryError> {
        let head = read_file_text_bounded(&join_path(&self.git_dir, "HEAD"), 4096)?;
        let head = head.trim();
        if let Some(reference) = head.strip_prefix("ref: ") {
            validate_reference(reference)?;
            let path = join_path(&self.git_dir, reference);
            if !path_is_file(&path) { return Ok(None); }
            let value = read_file_text_bounded(&path, 4096)?;
            return ObjectId::parse(value.trim()).map(Some).ok_or(RepositoryError::InvalidHead);
        }
        ObjectId::parse(head).map(Some).ok_or(RepositoryError::InvalidHead)
    }

    pub fn index(&self) -> Result<Index, RepositoryError> {
        let path = join_path(&self.git_dir, "index");
        if !path_is_file(&path) { return Ok(Index::empty()); }
        Ok(Index::parse(&read_file_bounded(&path, MAX_INDEX_BYTES)?)?)
    }

    pub fn write_object(&self, kind: ObjectKind, contents: &[u8]) -> Result<ObjectId, RepositoryError> {
        let (id, encoded) = encode_loose_object(kind, contents);
        let hex = id.to_hex();
        let directory = join_path(&join_path(&self.git_dir, "objects"), &hex[..2]);
        let path = join_path(&directory, &hex[2..]);
        if !path_is_file(&path) {
            create_dir_all(&directory)?;
            write_file(&path, &encoded)?;
        }
        Ok(id)
    }

    pub fn changes(&self) -> Result<Vector<NativeChange>, RepositoryError> {
        let index = self.index()?;
        let mut changes = Vector::new();
        for entry in index.entries.iter().filter(|entry| entry.stage == 0) {
            let path = join_path(&self.worktree, &entry.path);
            if !path_exists(&path) {
                changes.push(NativeChange { path: entry.path.clone(), kind: NativeChangeKind::Deleted });
            } else if !path_is_file(&path) {
                return Err(RepositoryError::UnsupportedFileType);
            } else {
                let bytes = read_file_bounded(&path, MAX_WORKTREE_FILE).map_err(|error| {
                    if error == FileError::ReadFailed { RepositoryError::FileTooLarge } else { error.into() }
                })?;
                if ObjectId::blob(&bytes) != entry.id {
                    changes.push(NativeChange { path: entry.path.clone(), kind: NativeChangeKind::Modified });
                }
            }
        }
        self.collect_untracked(&index, &self.worktree, "", 0, &mut changes)?;
        Ok(changes)
    }

    fn collect_untracked(&self, index: &Index, directory: &str, prefix: &str, depth: usize, changes: &mut Vector<NativeChange>) -> Result<(), RepositoryError> {
        if depth > MAX_DEPTH { return Err(RepositoryError::TooDeep); }
        for entry in read_directory(directory)? {
            if depth == 0 && entry.name == ".git" { continue; }
            if entry.is_symlink { return Err(RepositoryError::UnsupportedFileType); }
            let disk_path = join_path(directory, &entry.name);
            let relative = if prefix.is_empty() { entry.name.clone() } else {
                let mut value = Text::from(prefix); value.push('/'); value.push_str(&entry.name); value
            };
            if entry.is_directory {
                self.collect_untracked(index, &disk_path, &relative, depth + 1, changes)?;
            } else if index.entry(&relative).is_none() {
                if changes.len() >= MAX_WORKTREE_ENTRIES { return Err(RepositoryError::TooManyFiles); }
                changes.push(NativeChange { path: relative, kind: NativeChangeKind::Untracked });
            }
        }
        Ok(())
    }
}

fn validate_reference(reference: &str) -> Result<(), RepositoryError> {
    if !reference.starts_with("refs/") || reference.ends_with('/') || reference.contains("..")
        || reference.contains("@{") || reference.contains(['\\', ' ', '~', '^', ':', '?', '*', '['])
        || reference.split('/').any(|part| part.is_empty() || part.starts_with('.') || part.ends_with('.') || part.ends_with(".lock"))
        || reference.chars().any(char::is_control)
    { Err(RepositoryError::InvalidReference) } else { Ok(()) }
}

impl From<FileError> for RepositoryError { fn from(value: FileError) -> Self { Self::File(value) } }
impl From<IndexError> for RepositoryError { fn from(value: IndexError) -> Self { Self::Index(value) } }

impl fmt::Display for RepositoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => write!(formatter, "{error}"), Self::Index(error) => write!(formatter, "{error}"),
            Self::NotRepository => formatter.write_str("not an MRML Git repository"),
            Self::AlreadyExists => formatter.write_str("repository metadata already exists"),
            Self::UnsupportedLayout => formatter.write_str("linked worktrees and gitdir files are not supported yet"),
            Self::InvalidHead => formatter.write_str("invalid HEAD"), Self::InvalidReference => formatter.write_str("unsafe or invalid reference"),
            Self::TooManyFiles => formatter.write_str("working tree contains too many files"), Self::TooDeep => formatter.write_str("working tree is too deeply nested"),
            Self::UnsupportedFileType => formatter.write_str("symbolic links and special files are not supported yet"),
            Self::FileTooLarge => formatter.write_str("working-tree file exceeds the native status limit"),
        }
    }
}
impl core::error::Error for RepositoryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_runtime::{process_id, remove_dir_all, temporary_directory};

    fn root(name: &str) -> Text { join_path(&temporary_directory(), &mrml_runtime::mrml_format!("mrml-git-{name}-{}", process_id())) }

    #[test]
    fn initializes_and_discovers_without_a_git_process() {
        let path = root("init");
        let repository = Repository::init(&path).unwrap();
        assert_eq!(repository.current_branch().unwrap().as_deref(), Some("main"));
        assert_eq!(repository.head().unwrap(), None);
        let nested = join_path(&path, "src/deep");
        create_dir_all(&nested).unwrap();
        assert_eq!(Repository::discover(&nested).unwrap(), repository);
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn reports_untracked_files_natively() {
        let path = root("status");
        let repository = Repository::init(&path).unwrap();
        write_file(&join_path(&path, "hello.txt"), b"native").unwrap();
        assert_eq!(repository.changes().unwrap(), Vector::from([NativeChange { path: "hello.txt".into(), kind: NativeChangeKind::Untracked }]));
        remove_dir_all(&path).unwrap();
    }
}
