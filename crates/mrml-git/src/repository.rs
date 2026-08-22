use core::fmt;
use mrml_runtime::{
    FileError, Text, Vector, canonical_path, create_dir_all, join_path, parent_path, path_exists,
    path_is_directory, path_is_file, read_directory, read_file_bounded, read_file_text_bounded,
    write_file,
};

use crate::{
    Commit, Index, IndexEntry, IndexError, Object, ObjectError, ObjectId, ObjectKind,
    decode_loose_object, encode_loose_object, parse_tree,
};

const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;
const MAX_WORKTREE_FILE: usize = 256 * 1024 * 1024;
const MAX_WORKTREE_ENTRIES: usize = 1_000_000;
const MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Repository {
    pub worktree: Text,
    pub git_dir: Text,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeChangeKind {
    Modified,
    Deleted,
    Untracked,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeChange {
    pub path: Text,
    pub kind: NativeChangeKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepositoryError {
    File(FileError),
    Index(IndexError),
    NotRepository,
    AlreadyExists,
    UnsupportedLayout,
    InvalidHead,
    InvalidReference,
    TooManyFiles,
    TooDeep,
    UnsupportedFileType,
    FileTooLarge,
    InvalidWorktreePath,
    ConflictedIndex,
    DetachedHead,
    InvalidIdentity,
    ReferenceExists,
    ReferenceMissing,
    CurrentBranch,
    Object(ObjectError),
    WorktreeDirty,
}

impl Repository {
    pub fn discover(start: &str) -> Result<Self, RepositoryError> {
        let resolved = canonical_path(start)?;
        let mut current = if path_is_directory(&resolved) {
            resolved
        } else {
            parent_path(&resolved)
                .ok_or(RepositoryError::NotRepository)?
                .into()
        };
        loop {
            let marker = join_path(&current, ".git");
            if path_is_directory(&marker) {
                return Ok(Self {
                    worktree: current,
                    git_dir: marker,
                });
            }
            if path_exists(&marker) {
                return Err(RepositoryError::UnsupportedLayout);
            }
            let Some(parent) = parent_path(&current) else {
                break;
            };
            if current == parent {
                break;
            }
            current = parent.into();
        }
        Err(RepositoryError::NotRepository)
    }

    pub fn init(path: &str) -> Result<Self, RepositoryError> {
        create_dir_all(path)?;
        let worktree = canonical_path(path)?;
        let git_dir = join_path(&worktree, ".git");
        if path_exists(&git_dir) {
            return Err(RepositoryError::AlreadyExists);
        }
        create_dir_all(&join_path(&git_dir, "objects/info"))?;
        create_dir_all(&join_path(&git_dir, "objects/pack"))?;
        create_dir_all(&join_path(&git_dir, "refs/heads"))?;
        create_dir_all(&join_path(&git_dir, "refs/tags"))?;
        write_file(&join_path(&git_dir, "HEAD"), b"ref: refs/heads/main\n")?;
        write_file(
            &join_path(&git_dir, "config"),
            b"[core]\n\trepositoryformatversion = 0\n\tbare = false\n",
        )?;
        write_file(
            &join_path(&git_dir, "description"),
            b"Unnamed MRML repository\n",
        )?;
        Ok(Self { worktree, git_dir })
    }

    pub fn current_branch(&self) -> Result<Option<Text>, RepositoryError> {
        let head = read_file_text_bounded(&join_path(&self.git_dir, "HEAD"), 4096)?;
        let head = head.trim();
        if let Some(reference) = head.strip_prefix("ref: ") {
            validate_reference(reference)?;
            return Ok(reference.strip_prefix("refs/heads/").map(Into::into));
        }
        ObjectId::parse(head)
            .map(|_| None)
            .ok_or(RepositoryError::InvalidHead)
    }

    pub fn head(&self) -> Result<Option<ObjectId>, RepositoryError> {
        let head = read_file_text_bounded(&join_path(&self.git_dir, "HEAD"), 4096)?;
        let head = head.trim();
        if let Some(reference) = head.strip_prefix("ref: ") {
            validate_reference(reference)?;
            let path = join_path(&self.git_dir, reference);
            if !path_is_file(&path) {
                return Ok(None);
            }
            let value = read_file_text_bounded(&path, 4096)?;
            return ObjectId::parse(value.trim())
                .map(Some)
                .ok_or(RepositoryError::InvalidHead);
        }
        ObjectId::parse(head)
            .map(Some)
            .ok_or(RepositoryError::InvalidHead)
    }

    pub fn index(&self) -> Result<Index, RepositoryError> {
        let path = join_path(&self.git_dir, "index");
        if !path_is_file(&path) {
            return Ok(Index::empty());
        }
        Ok(Index::parse(&read_file_bounded(&path, MAX_INDEX_BYTES)?)?)
    }

    pub fn write_object(
        &self,
        kind: ObjectKind,
        contents: &[u8],
    ) -> Result<ObjectId, RepositoryError> {
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

    pub fn read_object(&self, id: ObjectId) -> Result<Object, RepositoryError> {
        let hex = id.to_hex();
        let path = join_path(
            &join_path(&self.git_dir, "objects"),
            &mrml_runtime::mrml_format!("{}/{}", &hex[..2], &hex[2..]),
        );
        let encoded = read_file_bounded(&path, MAX_WORKTREE_FILE)?;
        let object = decode_loose_object(&encoded)?;
        let (computed, _) = encode_loose_object(object.kind, &object.contents);
        if computed != id {
            return Err(RepositoryError::InvalidHead);
        }
        Ok(object)
    }

    pub fn resolve_revision(&self, revision: &str) -> Result<ObjectId, RepositoryError> {
        if revision == "HEAD" { return self.head()?.ok_or(RepositoryError::ReferenceMissing); }
        if let Some(id) = ObjectId::parse(revision) { return Ok(id); }
        for prefix in ["refs/heads", "refs/tags"] {
            let reference = mrml_runtime::mrml_format!("{prefix}/{revision}");
            validate_reference(&reference)?;
            let path = join_path(&self.git_dir, &reference);
            if path_is_file(&path) {
                let value = read_file_text_bounded(&path, 4096)?;
                return ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference);
            }
        }
        Err(RepositoryError::ReferenceMissing)
    }

    pub fn read_commit(&self, id: ObjectId) -> Result<Commit, RepositoryError> {
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Commit { return Err(RepositoryError::InvalidReference); }
        Ok(Commit::parse(&object.contents)?)
    }

    pub fn history(&self, start: ObjectId, limit: usize) -> Result<Vector<(ObjectId, Commit)>, RepositoryError> {
        let mut history = Vector::new();
        let mut next = Some(start);
        while let Some(id) = next {
            if history.len() >= limit { break; }
            let commit = self.read_commit(id)?;
            next = commit.parents.first().copied();
            history.push((id, commit));
        }
        Ok(history)
    }

    pub fn tree_index(&self, tree: ObjectId) -> Result<Index, RepositoryError> {
        let mut index = Index::empty();
        self.flatten_tree(tree, "", 0, &mut index)?;
        Ok(index)
    }

    fn flatten_tree(&self, id: ObjectId, prefix: &str, depth: usize, index: &mut Index) -> Result<(), RepositoryError> {
        if depth > MAX_DEPTH { return Err(RepositoryError::TooDeep); }
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Tree { return Err(RepositoryError::InvalidReference); }
        for entry in parse_tree(&object.contents)? {
            let path = if prefix.is_empty() { entry.name.clone() } else { mrml_runtime::mrml_format!("{prefix}/{}", entry.name) };
            if entry.mode == 0o40000 {
                self.flatten_tree(entry.id, &path, depth + 1, index)?;
            } else if matches!(entry.mode, 0o100644 | 0o100755) {
                if index.entries.len() >= MAX_WORKTREE_ENTRIES { return Err(RepositoryError::TooManyFiles); }
                let object = self.read_object(entry.id)?;
                if object.kind != ObjectKind::Blob { return Err(RepositoryError::InvalidReference); }
                index.upsert(IndexEntry { path, id: entry.id, mode: entry.mode, size: object.contents.len().try_into().map_err(|_| RepositoryError::FileTooLarge)?, stage: 0 });
            } else {
                return Err(RepositoryError::UnsupportedFileType);
            }
        }
        Ok(())
    }

    pub fn switch_branch(&self, branch: &str) -> Result<ObjectId, RepositoryError> {
        validate_reference(&mrml_runtime::mrml_format!("refs/heads/{branch}"))?;
        if !self.changes()?.is_empty() { return Err(RepositoryError::WorktreeDirty); }
        let id = self.resolve_revision(branch)?;
        let commit = self.read_commit(id)?;
        let target = self.tree_index(commit.tree)?;
        let current = self.index()?;
        let mut files: Vector<(Text, Vector<u8>)> = Vector::new();
        for entry in &target.entries {
            let object = self.read_object(entry.id)?;
            if object.kind != ObjectKind::Blob { return Err(RepositoryError::InvalidReference); }
            files.push((entry.path.clone(), object.contents));
        }
        for entry in &current.entries {
            if target.entry(&entry.path).is_none() {
                let path = join_path(&self.worktree, &entry.path);
                if path_is_file(&path) { mrml_runtime::remove_file(&path)?; }
            }
        }
        for (relative, contents) in files {
            let path = join_path(&self.worktree, &relative);
            if let Some(parent) = parent_path(&path) { create_dir_all(parent)?; }
            write_file(&path, &contents)?;
        }
        self.write_index(&target)?;
        write_file(&join_path(&self.git_dir, "HEAD"), mrml_runtime::mrml_format!("ref: refs/heads/{branch}\n").as_bytes())?;
        Ok(id)
    }

    fn write_index(&self, index: &Index) -> Result<(), RepositoryError> {
        let encoded = index.encode()?;
        let lock = join_path(&self.git_dir, "index.lock");
        if path_exists(&lock) { return Err(RepositoryError::AlreadyExists); }
        write_file(&lock, &encoded)?;
        mrml_runtime::rename_file(&lock, &join_path(&self.git_dir, "index"))?;
        Ok(())
    }

    pub fn restore(&self, paths: &[Text]) -> Result<(), RepositoryError> {
        let index = self.index()?;
        let mut files: Vector<(Text, Vector<u8>)> = Vector::new();
        for path in paths {
            validate_worktree_path(path)?;
            let entry = index.entry(path).ok_or(RepositoryError::ReferenceMissing)?;
            let object = self.read_object(entry.id)?;
            if object.kind != ObjectKind::Blob { return Err(RepositoryError::InvalidReference); }
            files.push((path.clone(), object.contents));
        }
        for (relative, contents) in files {
            let path = join_path(&self.worktree, &relative);
            if let Some(parent) = parent_path(&path) { create_dir_all(parent)?; }
            write_file(&path, &contents)?;
        }
        Ok(())
    }

    pub fn unstage(&self, paths: &[Text]) -> Result<(), RepositoryError> {
        let mut index = self.index()?;
        let head = match self.head()? {
            Some(id) => Some(self.tree_index(self.read_commit(id)?.tree)?),
            None => None,
        };
        for path in paths {
            validate_worktree_path(path)?;
            if let Some(entry) = head.as_ref().and_then(|value| value.entry(path)).cloned() {
                index.upsert(entry);
            } else {
                index.remove(path);
            }
        }
        self.write_index(&index)
    }

    pub fn branches(&self) -> Result<Vector<(Text, ObjectId)>, RepositoryError> {
        let mut branches = Vector::new();
        let root = join_path(&self.git_dir, "refs/heads");
        if path_is_directory(&root) {
            self.collect_references(&root, "", &mut branches)?;
        }
        branches.sort_unstable_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        Ok(branches)
    }

    fn collect_references(
        &self,
        directory: &str,
        prefix: &str,
        output: &mut Vector<(Text, ObjectId)>,
    ) -> Result<(), RepositoryError> {
        for entry in read_directory(directory)? {
            if entry.is_symlink {
                return Err(RepositoryError::UnsupportedFileType);
            }
            let path = join_path(directory, &entry.name);
            let name = if prefix.is_empty() {
                entry.name.clone()
            } else {
                mrml_runtime::mrml_format!("{}/{}", prefix, entry.name)
            };
            if entry.is_directory {
                self.collect_references(&path, &name, output)?;
            } else {
                let value = read_file_text_bounded(&path, 4096)?;
                let id = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
                output.push((name, id));
            }
        }
        Ok(())
    }

    pub fn create_branch(&self, name: &str, switch: bool) -> Result<ObjectId, RepositoryError> {
        let reference = mrml_runtime::mrml_format!("refs/heads/{name}");
        validate_reference(&reference)?;
        let id = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let path = join_path(&self.git_dir, &reference);
        if path_exists(&path) {
            return Err(RepositoryError::ReferenceExists);
        }
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        write_file(&path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        if switch {
            write_file(
                &join_path(&self.git_dir, "HEAD"),
                mrml_runtime::mrml_format!("ref: {reference}\n").as_bytes(),
            )?;
        }
        Ok(id)
    }

    pub fn delete_branch(&self, name: &str) -> Result<(), RepositoryError> {
        let reference = mrml_runtime::mrml_format!("refs/heads/{name}");
        validate_reference(&reference)?;
        if self.current_branch()?.as_deref() == Some(name) {
            return Err(RepositoryError::CurrentBranch);
        }
        let path = join_path(&self.git_dir, &reference);
        if !path_is_file(&path) {
            return Err(RepositoryError::ReferenceMissing);
        }
        mrml_runtime::remove_file(&path)?;
        Ok(())
    }

    pub fn tags(&self) -> Result<Vector<(Text, ObjectId)>, RepositoryError> {
        let mut tags = Vector::new();
        let root = join_path(&self.git_dir, "refs/tags");
        if path_is_directory(&root) {
            self.collect_references(&root, "", &mut tags)?;
        }
        tags.sort_unstable_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
        Ok(tags)
    }

    pub fn create_tag(&self, name: &str) -> Result<ObjectId, RepositoryError> {
        let reference = mrml_runtime::mrml_format!("refs/tags/{name}");
        validate_reference(&reference)?;
        let id = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let path = join_path(&self.git_dir, &reference);
        if path_exists(&path) {
            return Err(RepositoryError::ReferenceExists);
        }
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        write_file(&path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        Ok(id)
    }

    pub fn remotes(&self) -> Result<Vector<(Text, Text)>, RepositoryError> {
        let config = read_file_text_bounded(&join_path(&self.git_dir, "config"), 1024 * 1024)?;
        let mut output = Vector::new();
        let mut current: Option<Text> = None;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                current = remote_section(line).map(Into::into);
            } else if let Some(name) = &current {
                if let Some((key, value)) = line.split_once('=') {
                    if key.trim() == "url" { output.push((name.clone(), value.trim().into())); }
                }
            }
        }
        Ok(output)
    }

    pub fn set_remote(&self, name: &str, url: &str, require_existing: bool) -> Result<(), RepositoryError> {
        validate_config_name(name)?;
        if url.is_empty() || url.chars().any(char::is_control) { return Err(RepositoryError::InvalidReference); }
        let path = join_path(&self.git_dir, "config");
        let config = read_file_text_bounded(&path, 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[remote \"{name}\"]");
        let mut output = Text::new();
        let mut in_target = false;
        let mut found = false;
        let mut wrote_url = false;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                if in_target && !wrote_url { output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n")); }
                in_target = target == line;
                found |= in_target;
                wrote_url = false;
            }
            if in_target && line.split_once('=').is_some_and(|(key, _)| key.trim() == "url") {
                output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n"));
                wrote_url = true;
            } else {
                output.push_str(raw); output.push('\n');
            }
        }
        if in_target && !wrote_url { output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n")); }
        if !found {
            if require_existing { return Err(RepositoryError::ReferenceMissing); }
            output.push_str(&mrml_runtime::mrml_format!("\n{target}\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/{name}/*\n"));
        }
        write_file(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn config_value(&self, section: &str, key: &str) -> Result<Option<Text>, RepositoryError> {
        validate_config_name(section)?; validate_config_name(key)?;
        let config = read_file_text_bounded(&join_path(&self.git_dir, "config"), 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[{section}]");
        let mut active = false;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') { active = target == line; }
            else if active {
                if let Some((found, value)) = line.split_once('=') { if found.trim() == key { return Ok(Some(value.trim().into())); } }
            }
        }
        Ok(None)
    }

    pub fn stage(&self, paths: &[Text]) -> Result<(), RepositoryError> {
        let mut index = self.index()?;
        for relative in paths {
            validate_worktree_path(relative)?;
            let disk_path = join_path(&self.worktree, relative);
            if !path_exists(&disk_path) {
                index.remove(relative);
                continue;
            }
            if !path_is_file(&disk_path) {
                return Err(RepositoryError::UnsupportedFileType);
            }
            let contents = read_file_bounded(&disk_path, MAX_WORKTREE_FILE).map_err(|error| {
                if error == FileError::ReadFailed {
                    RepositoryError::FileTooLarge
                } else {
                    error.into()
                }
            })?;
            let id = self.write_object(ObjectKind::Blob, &contents)?;
            index.upsert(crate::IndexEntry {
                path: relative.clone(),
                id,
                mode: 0o100644,
                size: contents.len() as u32,
                stage: 0,
            });
        }
        let encoded = index.encode()?;
        let temporary = join_path(&self.git_dir, "index.lock");
        if path_exists(&temporary) {
            return Err(RepositoryError::AlreadyExists);
        }
        write_file(&temporary, &encoded)?;
        mrml_runtime::rename_file(&temporary, &join_path(&self.git_dir, "index"))?;
        Ok(())
    }

    pub fn write_tree(&self) -> Result<ObjectId, RepositoryError> {
        let index = self.index()?;
        if index.entries.iter().any(|entry| entry.stage != 0) {
            return Err(RepositoryError::ConflictedIndex);
        }
        self.write_tree_prefix(&index, "")
    }

    fn write_tree_prefix(&self, index: &Index, prefix: &str) -> Result<ObjectId, RepositoryError> {
        let mut items: Vector<TreeItem> = Vector::new();
        for entry in &index.entries {
            let Some(remainder) = entry.path.strip_prefix(prefix) else {
                continue;
            };
            if let Some(split) = remainder.find('/') {
                let name = &remainder[..split];
                if items.iter().any(|item| item.directory && item.name == name) {
                    continue;
                }
                let mut child_prefix = Text::from(prefix);
                child_prefix.push_str(name);
                child_prefix.push('/');
                let id = self.write_tree_prefix(index, &child_prefix)?;
                items.push(TreeItem {
                    name: name.into(),
                    mode: 0o40000,
                    id,
                    directory: true,
                });
            } else if !remainder.is_empty() {
                items.push(TreeItem {
                    name: remainder.into(),
                    mode: entry.mode,
                    id: entry.id,
                    directory: false,
                });
            }
        }
        items.sort_unstable_by(|left, right| tree_name_cmp(left, right));
        let mut contents = Vector::new();
        for item in items {
            let mode = mrml_runtime::mrml_format!("{:o}", item.mode);
            contents.extend(mode.as_bytes().iter().copied());
            contents.push(b' ');
            contents.extend(item.name.as_bytes().iter().copied());
            contents.push(0);
            contents.extend(item.id.0);
        }
        self.write_object(ObjectKind::Tree, &contents)
    }

    pub fn commit(
        &self,
        message: &str,
        name: &str,
        email: &str,
        timestamp: u64,
    ) -> Result<ObjectId, RepositoryError> {
        validate_identity(name, email)?;
        if message.trim().is_empty() || message.chars().any(|character| character == '\0') {
            return Err(RepositoryError::InvalidIdentity);
        }
        let branch = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let tree = self.write_tree()?;
        let parent = self.head()?;
        let mut contents = mrml_runtime::mrml_format!("tree {}\n", tree);
        if let Some(parent) = parent {
            contents.push_str(&mrml_runtime::mrml_format!("parent {}\n", parent));
        }
        contents.push_str(&mrml_runtime::mrml_format!(
            "author {} <{}> {} +0000\ncommitter {} <{}> {} +0000\n\n",
            name,
            email,
            timestamp,
            name,
            email,
            timestamp
        ));
        contents.push_str(message.trim());
        contents.push('\n');
        let id = self.write_object(ObjectKind::Commit, contents.as_bytes())?;
        let reference = mrml_runtime::mrml_format!("refs/heads/{}", branch);
        validate_reference(&reference)?;
        let path = join_path(&self.git_dir, &reference);
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        let lock = mrml_runtime::mrml_format!("{}.lock", path);
        if path_exists(&lock) {
            return Err(RepositoryError::AlreadyExists);
        }
        write_file(&lock, mrml_runtime::mrml_format!("{}\n", id).as_bytes())?;
        mrml_runtime::rename_file(&lock, &path)?;
        Ok(id)
    }

    pub fn changes(&self) -> Result<Vector<NativeChange>, RepositoryError> {
        let index = self.index()?;
        let mut changes = Vector::new();
        for entry in index.entries.iter().filter(|entry| entry.stage == 0) {
            let path = join_path(&self.worktree, &entry.path);
            if !path_exists(&path) {
                changes.push(NativeChange {
                    path: entry.path.clone(),
                    kind: NativeChangeKind::Deleted,
                });
            } else if !path_is_file(&path) {
                return Err(RepositoryError::UnsupportedFileType);
            } else {
                let bytes = read_file_bounded(&path, MAX_WORKTREE_FILE).map_err(|error| {
                    if error == FileError::ReadFailed {
                        RepositoryError::FileTooLarge
                    } else {
                        error.into()
                    }
                })?;
                if ObjectId::blob(&bytes) != entry.id {
                    changes.push(NativeChange {
                        path: entry.path.clone(),
                        kind: NativeChangeKind::Modified,
                    });
                }
            }
        }
        self.collect_untracked(&index, &self.worktree, "", 0, &mut changes)?;
        Ok(changes)
    }

    fn collect_untracked(
        &self,
        index: &Index,
        directory: &str,
        prefix: &str,
        depth: usize,
        changes: &mut Vector<NativeChange>,
    ) -> Result<(), RepositoryError> {
        if depth > MAX_DEPTH {
            return Err(RepositoryError::TooDeep);
        }
        for entry in read_directory(directory)? {
            if depth == 0 && entry.name == ".git" {
                continue;
            }
            if entry.is_symlink {
                return Err(RepositoryError::UnsupportedFileType);
            }
            let disk_path = join_path(directory, &entry.name);
            let relative = if prefix.is_empty() {
                entry.name.clone()
            } else {
                let mut value = Text::from(prefix);
                value.push('/');
                value.push_str(&entry.name);
                value
            };
            if entry.is_directory {
                self.collect_untracked(index, &disk_path, &relative, depth + 1, changes)?;
            } else if index.entry(&relative).is_none() {
                if changes.len() >= MAX_WORKTREE_ENTRIES {
                    return Err(RepositoryError::TooManyFiles);
                }
                changes.push(NativeChange {
                    path: relative,
                    kind: NativeChangeKind::Untracked,
                });
            }
        }
        Ok(())
    }
}

fn validate_reference(reference: &str) -> Result<(), RepositoryError> {
    if !reference.starts_with("refs/")
        || reference.ends_with('/')
        || reference.contains("..")
        || reference.contains("@{")
        || reference.contains(['\\', ' ', '~', '^', ':', '?', '*', '['])
        || reference.split('/').any(|part| {
            part.is_empty()
                || part.starts_with('.')
                || part.ends_with('.')
                || part.ends_with(".lock")
        })
        || reference.chars().any(char::is_control)
    {
        Err(RepositoryError::InvalidReference)
    } else {
        Ok(())
    }
}

fn validate_worktree_path(path: &str) -> Result<(), RepositoryError> {
    if path.is_empty()
        || mrml_runtime::path_is_absolute(path)
        || path.contains('\\')
        || path
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
        || path.chars().any(char::is_control)
    {
        Err(RepositoryError::InvalidWorktreePath)
    } else {
        Ok(())
    }
}

struct TreeItem {
    name: Text,
    mode: u32,
    id: ObjectId,
    directory: bool,
}

fn tree_name_cmp(left: &TreeItem, right: &TreeItem) -> core::cmp::Ordering {
    let mut index = 0;
    loop {
        let left_byte = left
            .name
            .as_bytes()
            .get(index)
            .copied()
            .or_else(|| left.directory.then_some(b'/'));
        let right_byte = right
            .name
            .as_bytes()
            .get(index)
            .copied()
            .or_else(|| right.directory.then_some(b'/'));
        match (left_byte, right_byte) {
            (Some(a), Some(b)) if a == b => index += 1,
            (Some(a), Some(b)) => return a.cmp(&b),
            (None, None) => return core::cmp::Ordering::Equal,
            (None, Some(_)) => return core::cmp::Ordering::Less,
            (Some(_), None) => return core::cmp::Ordering::Greater,
        }
    }
}

fn validate_identity(name: &str, email: &str) -> Result<(), RepositoryError> {
    if name.trim().is_empty()
        || email.trim().is_empty()
        || name.contains(['\n', '\r', '<', '>'])
        || email.contains(['\n', '\r', '<', '>'])
    {
        Err(RepositoryError::InvalidIdentity)
    } else {
        Ok(())
    }
}

fn validate_config_name(value: &str) -> Result<(), RepositoryError> {
    if value.is_empty() || value.starts_with('-') || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_')) {
        Err(RepositoryError::InvalidReference)
    } else { Ok(()) }
}

fn remote_section(line: &str) -> Option<&str> {
    line.strip_prefix("[remote \"")?.strip_suffix("\"]").filter(|name| validate_config_name(name).is_ok())
}

impl From<FileError> for RepositoryError {
    fn from(value: FileError) -> Self {
        Self::File(value)
    }
}
impl From<IndexError> for RepositoryError {
    fn from(value: IndexError) -> Self {
        Self::Index(value)
    }
}
impl From<ObjectError> for RepositoryError {
    fn from(value: ObjectError) -> Self {
        Self::Object(value)
    }
}

impl fmt::Display for RepositoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => write!(formatter, "{error}"),
            Self::Index(error) => write!(formatter, "{error}"),
            Self::Object(error) => write!(formatter, "{error}"),
            Self::WorktreeDirty => formatter.write_str("working tree is not clean"),
            Self::NotRepository => formatter.write_str("not an MRML Git repository"),
            Self::AlreadyExists => formatter.write_str("repository metadata already exists"),
            Self::UnsupportedLayout => {
                formatter.write_str("linked worktrees and gitdir files are not supported yet")
            }
            Self::InvalidHead => formatter.write_str("invalid HEAD"),
            Self::InvalidReference => formatter.write_str("unsafe or invalid reference"),
            Self::TooManyFiles => formatter.write_str("working tree contains too many files"),
            Self::TooDeep => formatter.write_str("working tree is too deeply nested"),
            Self::UnsupportedFileType => {
                formatter.write_str("symbolic links and special files are not supported yet")
            }
            Self::FileTooLarge => {
                formatter.write_str("working-tree file exceeds the native status limit")
            }
            Self::InvalidWorktreePath => formatter.write_str("unsafe or invalid working-tree path"),
            Self::ConflictedIndex => {
                formatter.write_str("cannot write a tree from an unmerged index")
            }
            Self::DetachedHead => formatter.write_str("native commit requires a branch HEAD"),
            Self::InvalidIdentity => formatter.write_str("invalid commit identity or message"),
            Self::ReferenceExists => formatter.write_str("reference already exists"),
            Self::ReferenceMissing => {
                formatter.write_str("reference or starting commit does not exist")
            }
            Self::CurrentBranch => formatter.write_str("cannot delete the current branch"),
        }
    }
}
impl core::error::Error for RepositoryError {}

#[cfg(test)]
mod tests {
    use super::*;
    use mrml_runtime::{process_id, remove_dir_all, temporary_directory};

    fn root(name: &str) -> Text {
        join_path(
            &temporary_directory(),
            &mrml_runtime::mrml_format!("mrml-git-{name}-{}", process_id()),
        )
    }

    #[test]
    fn initializes_and_discovers_without_a_git_process() {
        let path = root("init");
        let repository = Repository::init(&path).unwrap();
        assert_eq!(
            repository.current_branch().unwrap().as_deref(),
            Some("main")
        );
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
        assert_eq!(
            repository.changes().unwrap(),
            Vector::from([NativeChange {
                path: "hello.txt".into(),
                kind: NativeChangeKind::Untracked
            }])
        );
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn stages_blob_and_updates_index_natively() {
        let path = root("stage");
        let repository = Repository::init(&path).unwrap();
        write_file(&join_path(&path, "hello.txt"), b"native").unwrap();
        repository
            .stage(&Vector::from([Text::from("hello.txt")]))
            .unwrap();
        assert_eq!(
            repository.index().unwrap().entry("hello.txt").unwrap().id,
            ObjectId::blob(b"native")
        );
        assert!(repository.changes().unwrap().is_empty());
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn creates_tree_commit_and_branch_reference_natively() {
        let path = root("commit");
        let repository = Repository::init(&path).unwrap();
        create_dir_all(&join_path(&path, "src")).unwrap();
        write_file(&join_path(&path, "src/lib.rs"), b"pub fn native() {}\n").unwrap();
        repository
            .stage(&Vector::from([Text::from("src/lib.rs")]))
            .unwrap();
        let commit = repository
            .commit("initial", "MRML", "mrml@example.invalid", 1_700_000_000)
            .unwrap();
        assert_eq!(repository.head().unwrap(), Some(commit));
        assert!(path_is_file(&join_path(
            &repository.git_dir,
            &mrml_runtime::mrml_format!(
                "objects/{}/{}",
                &commit.to_hex()[..2],
                &commit.to_hex()[2..]
            )
        )));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn creates_lists_and_deletes_native_refs() {
        let path = root("refs");
        let repository = Repository::init(&path).unwrap();
        write_file(&join_path(&path, "tracked"), b"value").unwrap();
        repository
            .stage(&Vector::from([Text::from("tracked")]))
            .unwrap();
        let head = repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        assert_eq!(
            repository.create_branch("topic/nested", false).unwrap(),
            head
        );
        assert!(
            repository
                .branches()
                .unwrap()
                .iter()
                .any(|(name, id)| name == "topic/nested" && *id == head)
        );
        repository.delete_branch("topic/nested").unwrap();
        assert!(
            !repository
                .branches()
                .unwrap()
                .iter()
                .any(|(name, _)| name == "topic/nested")
        );
        assert_eq!(repository.create_tag("v1").unwrap(), head);
        assert_eq!(repository.tags().unwrap()[0], (Text::from("v1"), head));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn switches_branches_by_materializing_authenticated_tree() {
        let path = root("switch");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        write_file(&file, b"main").unwrap();
        repository.stage(&Vector::from([Text::from("tracked")])).unwrap();
        let main = repository.commit("main", "MRML", "mrml@example.invalid", 1).unwrap();
        repository.create_branch("topic", true).unwrap();
        write_file(&file, b"topic").unwrap();
        repository.stage(&Vector::from([Text::from("tracked")])).unwrap();
        repository.commit("topic", "MRML", "mrml@example.invalid", 2).unwrap();
        assert_eq!(repository.switch_branch("main").unwrap(), main);
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"main");
        assert_eq!(repository.current_branch().unwrap().as_deref(), Some("main"));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn restores_worktree_and_unstages_to_head() {
        let path = root("restore");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        write_file(&file, b"base").unwrap();
        let paths = Vector::from([Text::from("tracked")]);
        repository.stage(&paths).unwrap();
        repository.commit("base", "MRML", "mrml@example.invalid", 1).unwrap();
        write_file(&file, b"changed").unwrap();
        repository.stage(&paths).unwrap();
        repository.unstage(&paths).unwrap();
        assert_eq!(repository.index().unwrap().entry("tracked").unwrap().id, ObjectId::blob(b"base"));
        repository.restore(&paths).unwrap();
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"base");
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn adds_and_updates_remote_configuration_natively() {
        let path = root("remote");
        let repository = Repository::init(&path).unwrap();
        repository.set_remote("origin", "git@example.invalid:owner/repo.git", false).unwrap();
        assert_eq!(repository.remotes().unwrap()[0], (Text::from("origin"), Text::from("git@example.invalid:owner/repo.git")));
        repository.set_remote("origin", "ssh://git@example.invalid/other/repo.git", true).unwrap();
        assert_eq!(repository.remotes().unwrap()[0].1, "ssh://git@example.invalid/other/repo.git");
        assert!(repository.set_remote("missing", "git@example.invalid:x/y", true).is_err());
        remove_dir_all(&path).unwrap();
    }
}
