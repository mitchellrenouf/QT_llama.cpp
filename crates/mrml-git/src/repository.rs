use core::fmt;
use mrml_runtime::{
    FileError, Text, Vector, canonical_path, create_dir_all, join_path, parent_path, path_exists,
    path_is_directory, path_is_file, read_directory, read_file_bounded, read_file_text_bounded,
    write_file,
};
use mrml_ssh::{RsaPrivateKey, RsaPublicKey, sign_sshsig, verify_sshsig};

use crate::{
    Commit, FileDiff, Index, IndexEntry, IndexError, Object, ObjectError, ObjectId, ObjectKind,
    PackError, PackObject, decode_loose_object, encode_loose_object, encode_pack, parse_pack,
    parse_tree,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlameLine {
    pub commit: ObjectId,
    pub author: Text,
    pub line_number: usize,
    pub text: Text,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeOutcome {
    UpToDate,
    FastForward(ObjectId),
    Merged(ObjectId),
    Conflicts(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RebaseOutcome {
    UpToDate,
    Rebased { count: usize, head: ObjectId },
    Conflicts(usize),
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
    Pack(PackError),
    WorktreeDirty,
    MergeRequired,
    Signing,
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

    pub fn import_pack(&self, source: &[u8]) -> Result<Vector<ObjectId>, RepositoryError> {
        let objects = parse_pack(source)?;
        let mut ids = Vector::new();
        for object in objects {
            let id = self.write_object(object.kind, &object.contents)?;
            if id != object.id {
                return Err(RepositoryError::InvalidHead);
            }
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn pack_reachable(&self, tip: ObjectId) -> Result<Vector<u8>, RepositoryError> {
        let mut pending = Vector::from([tip]);
        let mut seen = Vector::new();
        let mut objects = Vector::new();
        while let Some(id) = pending.pop() {
            if seen.contains(&id) {
                continue;
            }
            if seen.len() >= 1_000_000 {
                return Err(RepositoryError::TooManyFiles);
            }
            let object = self.read_object(id)?;
            match object.kind {
                ObjectKind::Commit => {
                    let commit = Commit::parse(&object.contents)?;
                    pending.push(commit.tree);
                    pending.extend(commit.parents.iter().copied());
                }
                ObjectKind::Tree => {
                    pending.extend(parse_tree(&object.contents)?.iter().map(|entry| entry.id))
                }
                ObjectKind::Tag => {
                    let text = core::str::from_utf8(&object.contents)
                        .map_err(|_| RepositoryError::InvalidReference)?;
                    let target = text
                        .lines()
                        .find_map(|line| line.strip_prefix("object "))
                        .and_then(ObjectId::parse)
                        .ok_or(RepositoryError::InvalidReference)?;
                    pending.push(target);
                }
                ObjectKind::Blob => {}
            }
            seen.push(id);
            objects.push(PackObject {
                id,
                kind: object.kind,
                contents: object.contents,
            });
        }
        encode_pack(&objects).map_err(Into::into)
    }

    pub fn resolve_revision(&self, revision: &str) -> Result<ObjectId, RepositoryError> {
        if revision == "HEAD" {
            return self.head()?.ok_or(RepositoryError::ReferenceMissing);
        }
        if let Some(id) = ObjectId::parse(revision) {
            return Ok(id);
        }
        if revision.starts_with("refs/") {
            validate_reference(revision)?;
            let value = read_file_text_bounded(&join_path(&self.git_dir, revision), 4096)?;
            return ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference);
        }
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
        if object.kind != ObjectKind::Commit {
            return Err(RepositoryError::InvalidReference);
        }
        Ok(Commit::parse(&object.contents)?)
    }

    pub fn history(
        &self,
        start: ObjectId,
        limit: usize,
    ) -> Result<Vector<(ObjectId, Commit)>, RepositoryError> {
        let mut history = Vector::new();
        let mut next = Some(start);
        while let Some(id) = next {
            if history.len() >= limit {
                break;
            }
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

    fn flatten_tree(
        &self,
        id: ObjectId,
        prefix: &str,
        depth: usize,
        index: &mut Index,
    ) -> Result<(), RepositoryError> {
        if depth > MAX_DEPTH {
            return Err(RepositoryError::TooDeep);
        }
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Tree {
            return Err(RepositoryError::InvalidReference);
        }
        for entry in parse_tree(&object.contents)? {
            let path = if prefix.is_empty() {
                entry.name.clone()
            } else {
                mrml_runtime::mrml_format!("{prefix}/{}", entry.name)
            };
            if entry.mode == 0o40000 {
                self.flatten_tree(entry.id, &path, depth + 1, index)?;
            } else if matches!(entry.mode, 0o100644 | 0o100755) {
                if index.entries.len() >= MAX_WORKTREE_ENTRIES {
                    return Err(RepositoryError::TooManyFiles);
                }
                let object = self.read_object(entry.id)?;
                if object.kind != ObjectKind::Blob {
                    return Err(RepositoryError::InvalidReference);
                }
                index.upsert(IndexEntry {
                    path,
                    id: entry.id,
                    mode: entry.mode,
                    size: object
                        .contents
                        .len()
                        .try_into()
                        .map_err(|_| RepositoryError::FileTooLarge)?,
                    stage: 0,
                });
            } else {
                return Err(RepositoryError::UnsupportedFileType);
            }
        }
        Ok(())
    }

    pub fn switch_branch(&self, branch: &str) -> Result<ObjectId, RepositoryError> {
        validate_reference(&mrml_runtime::mrml_format!("refs/heads/{branch}"))?;
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let id = self.resolve_revision(branch)?;
        let commit = self.read_commit(id)?;
        let target = self.tree_index(commit.tree)?;
        let current = self.index()?;
        let mut files: Vector<(Text, Vector<u8>)> = Vector::new();
        for entry in &target.entries {
            let object = self.read_object(entry.id)?;
            if object.kind != ObjectKind::Blob {
                return Err(RepositoryError::InvalidReference);
            }
            files.push((entry.path.clone(), object.contents));
        }
        for entry in &current.entries {
            if target.entry(&entry.path).is_none() {
                let path = join_path(&self.worktree, &entry.path);
                if path_is_file(&path) {
                    mrml_runtime::remove_file(&path)?;
                }
            }
        }
        for (relative, contents) in files {
            let path = join_path(&self.worktree, &relative);
            if let Some(parent) = parent_path(&path) {
                create_dir_all(parent)?;
            }
            write_file(&path, &contents)?;
        }
        self.write_index(&target)?;
        write_file(
            &join_path(&self.git_dir, "HEAD"),
            mrml_runtime::mrml_format!("ref: refs/heads/{branch}\n").as_bytes(),
        )?;
        Ok(id)
    }

    pub fn checkout_branch_at(
        &self,
        branch: &str,
        id: ObjectId,
    ) -> Result<ObjectId, RepositoryError> {
        let reference = mrml_runtime::mrml_format!("refs/heads/{branch}");
        validate_reference(&reference)?;
        self.read_commit(id)?;
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let path = join_path(&self.git_dir, &reference);
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        write_file(&path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        self.switch_branch(branch)
    }

    pub fn fast_forward(&self, revision: &str) -> Result<MergeOutcome, RepositoryError> {
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let current = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let target = self.resolve_revision(revision)?;
        if self.is_ancestor(target, current)? {
            return Ok(MergeOutcome::UpToDate);
        }
        if !self.is_ancestor(current, target)? {
            return Err(RepositoryError::MergeRequired);
        }
        let commit = self.read_commit(target)?;
        let index = self.tree_index(commit.tree)?;
        self.materialize_index(&index)?;
        let branch = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let path = join_path(
            &self.git_dir,
            &mrml_runtime::mrml_format!("refs/heads/{branch}"),
        );
        let lock = mrml_runtime::mrml_format!("{path}.lock");
        if path_exists(&lock) {
            return Err(RepositoryError::AlreadyExists);
        }
        write_file(&lock, mrml_runtime::mrml_format!("{target}\n").as_bytes())?;
        mrml_runtime::rename_file(&lock, &path)?;
        Ok(MergeOutcome::FastForward(target))
    }

    pub fn is_ancestor(
        &self,
        ancestor: ObjectId,
        descendant: ObjectId,
    ) -> Result<bool, RepositoryError> {
        const MAX_COMMITS: usize = 1_000_000;
        let mut pending = Vector::from([descendant]);
        let mut seen: Vector<ObjectId> = Vector::new();
        while let Some(id) = pending.pop() {
            if id == ancestor {
                return Ok(true);
            }
            if seen.iter().any(|value| *value == id) {
                continue;
            }
            if seen.len() >= MAX_COMMITS {
                return Err(RepositoryError::TooManyFiles);
            }
            seen.push(id);
            pending.extend(self.read_commit(id)?.parents.iter().copied());
        }
        Ok(false)
    }

    pub fn merge(
        &self,
        revision: &str,
        name: &str,
        email: &str,
        timestamp: u64,
    ) -> Result<MergeOutcome, RepositoryError> {
        validate_identity(name, email)?;
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let current = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let target = self.resolve_revision(revision)?;
        if self.is_ancestor(target, current)? {
            return Ok(MergeOutcome::UpToDate);
        }
        if !self.is_ancestor(current, target)? {
            write_file(
                &join_path(&self.git_dir, "ORIG_HEAD"),
                mrml_runtime::mrml_format!("{current}\n").as_bytes(),
            )?;
            return self.three_way_merge(current, target, revision, name, email, timestamp);
        }
        let commit = self.read_commit(target)?;
        let target_index = self.tree_index(commit.tree)?;
        self.materialize_index(&target_index)?;
        let branch = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let reference = mrml_runtime::mrml_format!("refs/heads/{branch}");
        let path = join_path(&self.git_dir, &reference);
        let lock = mrml_runtime::mrml_format!("{path}.lock");
        if path_exists(&lock) {
            return Err(RepositoryError::AlreadyExists);
        }
        write_file(&lock, mrml_runtime::mrml_format!("{target}\n").as_bytes())?;
        mrml_runtime::rename_file(&lock, &path)?;
        Ok(MergeOutcome::FastForward(target))
    }

    fn merge_base(&self, ours: ObjectId, theirs: ObjectId) -> Result<ObjectId, RepositoryError> {
        const LIMIT: usize = 1_000_000;
        let mut ours_seen = Vector::new();
        let mut pending = Vector::from([ours]);
        while let Some(id) = pending.pop() {
            if ours_seen.iter().any(|value| *value == id) {
                continue;
            }
            if ours_seen.len() >= LIMIT {
                return Err(RepositoryError::TooManyFiles);
            }
            ours_seen.push(id);
            pending.extend(self.read_commit(id)?.parents.iter().copied());
        }
        let mut theirs_seen = Vector::new();
        let mut queue = Vector::from([theirs]);
        let mut cursor = 0;
        while cursor < queue.len() {
            let id = queue[cursor];
            cursor += 1;
            if ours_seen.iter().any(|value| *value == id) {
                return Ok(id);
            }
            if theirs_seen.iter().any(|value| *value == id) {
                continue;
            }
            if theirs_seen.len() >= LIMIT {
                return Err(RepositoryError::TooManyFiles);
            }
            theirs_seen.push(id);
            queue.extend(self.read_commit(id)?.parents.iter().copied());
        }
        Err(RepositoryError::MergeRequired)
    }

    fn three_way_merge(
        &self,
        ours: ObjectId,
        theirs: ObjectId,
        label: &str,
        name: &str,
        email: &str,
        timestamp: u64,
    ) -> Result<MergeOutcome, RepositoryError> {
        let base = self.merge_base(ours, theirs)?;
        let base_index = self.tree_index(self.read_commit(base)?.tree)?;
        let ours_index = self.tree_index(self.read_commit(ours)?.tree)?;
        let theirs_index = self.tree_index(self.read_commit(theirs)?.tree)?;
        let mut names: Vector<Text> = base_index
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        for source in [&ours_index, &theirs_index] {
            for entry in &source.entries {
                if !names.iter().any(|path| path == &entry.path) {
                    names.push(entry.path.clone());
                }
            }
        }
        names.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        let mut merged = Index::empty();
        let mut conflicts: Vector<(Text, Option<Vector<u8>>, Option<Vector<u8>>)> = Vector::new();
        for path in names {
            let b = base_index.entry(&path);
            let o = ours_index.entry(&path);
            let t = theirs_index.entry(&path);
            let chosen = if same_entry(o, t) {
                o
            } else if same_entry(o, b) {
                t
            } else if same_entry(t, b) {
                o
            } else {
                None
            };
            if same_entry(o, t) || same_entry(o, b) || same_entry(t, b) {
                if let Some(entry) = chosen {
                    merged.upsert(entry.clone());
                }
            } else {
                if let Some(entry) = b {
                    let mut value = entry.clone();
                    value.stage = 1;
                    merged.upsert(value);
                }
                if let Some(entry) = o {
                    let mut value = entry.clone();
                    value.stage = 2;
                    merged.upsert(value);
                }
                if let Some(entry) = t {
                    let mut value = entry.clone();
                    value.stage = 3;
                    merged.upsert(value);
                }
                conflicts.push((
                    path,
                    match o {
                        Some(e) => Some(self.read_blob(e.id)?),
                        None => None,
                    },
                    match t {
                        Some(e) => Some(self.read_blob(e.id)?),
                        None => None,
                    },
                ));
            }
        }
        if !conflicts.is_empty() {
            let clean = Index {
                version: 2,
                entries: merged
                    .entries
                    .iter()
                    .filter(|entry| entry.stage == 0)
                    .cloned()
                    .collect(),
            };
            self.materialize_index(&clean)?;
            for (path, ours_bytes, theirs_bytes) in &conflicts {
                let disk = join_path(&self.worktree, path);
                if let Some(parent) = parent_path(&disk) {
                    create_dir_all(parent)?;
                }
                let ours_text = ours_bytes
                    .as_deref()
                    .and_then(|bytes| core::str::from_utf8(bytes).ok());
                let theirs_text = theirs_bytes
                    .as_deref()
                    .and_then(|bytes| core::str::from_utf8(bytes).ok());
                if let (Some(ours_text), Some(theirs_text)) = (ours_text, theirs_text) {
                    let marked = mrml_runtime::mrml_format!(
                        "<<<<<<< HEAD\n{}=======\n{}>>>>>>> {}\n",
                        ours_text,
                        theirs_text,
                        label
                    );
                    write_file(&disk, marked.as_bytes())?;
                } else if let Some(bytes) = ours_bytes {
                    write_file(&disk, bytes)?;
                }
            }
            self.write_index(&merged)?;
            write_file(
                &join_path(&self.git_dir, "MERGE_HEAD"),
                mrml_runtime::mrml_format!("{theirs}\n").as_bytes(),
            )?;
            return Ok(MergeOutcome::Conflicts(conflicts.len()));
        }
        self.materialize_index(&merged)?;
        let tree = self.write_tree_from_index(&merged)?;
        let message = mrml_runtime::mrml_format!("Merge {label}");
        let id =
            self.create_commit_object(tree, &[ours, theirs], &message, name, email, timestamp)?;
        self.update_current_branch(id)?;
        let original = join_path(&self.git_dir, "ORIG_HEAD");
        if path_is_file(&original) {
            mrml_runtime::remove_file(&original)?;
        }
        Ok(MergeOutcome::Merged(id))
    }

    pub fn abort_merge(&self) -> Result<ObjectId, RepositoryError> {
        let original_path = join_path(&self.git_dir, "ORIG_HEAD");
        let merge_path = join_path(&self.git_dir, "MERGE_HEAD");
        if !path_is_file(&original_path) || !path_is_file(&merge_path) {
            return Err(RepositoryError::ReferenceMissing);
        }
        let value = read_file_text_bounded(&original_path, 4096)?;
        let original = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
        let index = self.tree_index(self.read_commit(original)?.tree)?;
        self.materialize_index(&index)?;
        self.update_current_branch(original)?;
        mrml_runtime::remove_file(&merge_path)?;
        mrml_runtime::remove_file(&original_path)?;
        Ok(original)
    }

    pub fn cherry_pick(
        &self,
        revision: &str,
        committer_name: &str,
        committer_email: &str,
        timestamp: u64,
    ) -> Result<MergeOutcome, RepositoryError> {
        validate_identity(committer_name, committer_email)?;
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let ours = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let theirs = self.resolve_revision(revision)?;
        let picked = self.read_commit(theirs)?;
        if picked.parents.len() > 1 {
            return Err(RepositoryError::MergeRequired);
        }
        let base_index = match picked.parents.first() {
            Some(id) => self.tree_index(self.read_commit(*id)?.tree)?,
            None => Index::empty(),
        };
        let ours_index = self.tree_index(self.read_commit(ours)?.tree)?;
        let theirs_index = self.tree_index(picked.tree)?;
        let mut names: Vector<Text> = base_index
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        for source in [&ours_index, &theirs_index] {
            for entry in &source.entries {
                if !names.iter().any(|path| path == &entry.path) {
                    names.push(entry.path.clone());
                }
            }
        }
        names.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        let mut merged = Index::empty();
        let mut conflicts: Vector<(Text, Option<Vector<u8>>, Option<Vector<u8>>)> = Vector::new();
        for path in names {
            let b = base_index.entry(&path);
            let o = ours_index.entry(&path);
            let t = theirs_index.entry(&path);
            let chosen = if same_entry(o, t) {
                o
            } else if same_entry(o, b) {
                t
            } else if same_entry(t, b) {
                o
            } else {
                None
            };
            if same_entry(o, t) || same_entry(o, b) || same_entry(t, b) {
                if let Some(entry) = chosen {
                    merged.upsert(entry.clone());
                }
            } else {
                if let Some(entry) = b {
                    let mut value = entry.clone();
                    value.stage = 1;
                    merged.upsert(value);
                }
                if let Some(entry) = o {
                    let mut value = entry.clone();
                    value.stage = 2;
                    merged.upsert(value);
                }
                if let Some(entry) = t {
                    let mut value = entry.clone();
                    value.stage = 3;
                    merged.upsert(value);
                }
                conflicts.push((
                    path,
                    match o {
                        Some(e) => Some(self.read_blob(e.id)?),
                        None => None,
                    },
                    match t {
                        Some(e) => Some(self.read_blob(e.id)?),
                        None => None,
                    },
                ));
            }
        }
        write_file(
            &join_path(&self.git_dir, "ORIG_HEAD"),
            mrml_runtime::mrml_format!("{ours}\n").as_bytes(),
        )?;
        if !conflicts.is_empty() {
            let clean = Index {
                version: 2,
                entries: merged
                    .entries
                    .iter()
                    .filter(|entry| entry.stage == 0)
                    .cloned()
                    .collect(),
            };
            self.materialize_index(&clean)?;
            for (path, ours_bytes, theirs_bytes) in &conflicts {
                let disk = join_path(&self.worktree, path);
                if let Some(parent) = parent_path(&disk) {
                    create_dir_all(parent)?;
                }
                if let (Some(a), Some(b)) = (
                    ours_bytes
                        .as_deref()
                        .and_then(|v| core::str::from_utf8(v).ok()),
                    theirs_bytes
                        .as_deref()
                        .and_then(|v| core::str::from_utf8(v).ok()),
                ) {
                    write_file(
                        &disk,
                        mrml_runtime::mrml_format!(
                            "<<<<<<< HEAD\n{a}=======\n{b}>>>>>>> {revision}\n"
                        )
                        .as_bytes(),
                    )?;
                } else if let Some(bytes) = ours_bytes {
                    write_file(&disk, bytes)?;
                }
            }
            self.write_index(&merged)?;
            write_file(
                &join_path(&self.git_dir, "CHERRY_PICK_HEAD"),
                mrml_runtime::mrml_format!("{theirs}\n").as_bytes(),
            )?;
            return Ok(MergeOutcome::Conflicts(conflicts.len()));
        }
        self.materialize_index(&merged)?;
        let tree = self.write_tree_from_index(&merged)?;
        let mut contents = mrml_runtime::mrml_format!(
            "tree {tree}\nparent {ours}\nauthor {}\ncommitter {committer_name} <{committer_email}> {timestamp} +0000\n\n{}",
            picked.author,
            picked.message
        );
        if !contents.ends_with('\n') {
            contents.push('\n');
        }
        let id = self.write_object(ObjectKind::Commit, contents.as_bytes())?;
        self.update_current_branch(id)?;
        let original = join_path(&self.git_dir, "ORIG_HEAD");
        if path_is_file(&original) {
            mrml_runtime::remove_file(&original)?;
        }
        Ok(MergeOutcome::Merged(id))
    }

    pub fn abort_cherry_pick(&self) -> Result<ObjectId, RepositoryError> {
        self.abort_operation("CHERRY_PICK_HEAD")
    }

    pub fn rebase(
        &self,
        revision: &str,
        committer_name: &str,
        committer_email: &str,
        timestamp: u64,
    ) -> Result<RebaseOutcome, RepositoryError> {
        validate_identity(committer_name, committer_email)?;
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let original = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let target = self.resolve_revision(revision)?;
        if self.is_ancestor(target, original)? {
            return Ok(RebaseOutcome::UpToDate);
        }
        let base = self.merge_base(original, target)?;
        let mut commits = Vector::new();
        let mut cursor = original;
        while cursor != base {
            let commit = self.read_commit(cursor)?;
            if commit.parents.len() != 1 {
                return Err(RepositoryError::MergeRequired);
            }
            commits.push(cursor);
            cursor = commit.parents[0];
            if commits.len() > 1_000_000 {
                return Err(RepositoryError::TooManyFiles);
            }
        }
        commits.reverse();
        write_file(
            &join_path(&self.git_dir, "REBASE_ORIG_HEAD"),
            mrml_runtime::mrml_format!("{original}\n").as_bytes(),
        )?;
        let target_index = self.tree_index(self.read_commit(target)?.tree)?;
        self.materialize_index(&target_index)?;
        self.update_current_branch(target)?;
        let mut head = target;
        for (offset, commit) in commits.iter().enumerate() {
            write_file(
                &join_path(&self.git_dir, "REBASE_HEAD"),
                mrml_runtime::mrml_format!("{commit}\n").as_bytes(),
            )?;
            match self.cherry_pick(
                &commit.to_hex(),
                committer_name,
                committer_email,
                timestamp.saturating_add(offset as u64),
            )? {
                MergeOutcome::Merged(id) => head = id,
                MergeOutcome::Conflicts(count) => return Ok(RebaseOutcome::Conflicts(count)),
                _ => return Err(RepositoryError::MergeRequired),
            }
        }
        for name in ["REBASE_HEAD", "REBASE_ORIG_HEAD"] {
            let path = join_path(&self.git_dir, name);
            if path_is_file(&path) {
                mrml_runtime::remove_file(&path)?;
            }
        }
        Ok(RebaseOutcome::Rebased {
            count: commits.len(),
            head,
        })
    }

    pub fn abort_rebase(&self) -> Result<ObjectId, RepositoryError> {
        let original_path = join_path(&self.git_dir, "REBASE_ORIG_HEAD");
        if !path_is_file(&original_path) {
            return Err(RepositoryError::ReferenceMissing);
        }
        let value = read_file_text_bounded(&original_path, 4096)?;
        let original = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
        let index = self.tree_index(self.read_commit(original)?.tree)?;
        self.materialize_index(&index)?;
        self.update_current_branch(original)?;
        for name in [
            "REBASE_HEAD",
            "REBASE_ORIG_HEAD",
            "CHERRY_PICK_HEAD",
            "ORIG_HEAD",
        ] {
            let path = join_path(&self.git_dir, name);
            if path_is_file(&path) {
                mrml_runtime::remove_file(&path)?;
            }
        }
        Ok(original)
    }

    fn abort_operation(&self, state: &str) -> Result<ObjectId, RepositoryError> {
        let original_path = join_path(&self.git_dir, "ORIG_HEAD");
        let state_path = join_path(&self.git_dir, state);
        if !path_is_file(&original_path) || !path_is_file(&state_path) {
            return Err(RepositoryError::ReferenceMissing);
        }
        let value = read_file_text_bounded(&original_path, 4096)?;
        let original = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
        let index = self.tree_index(self.read_commit(original)?.tree)?;
        self.materialize_index(&index)?;
        self.update_current_branch(original)?;
        mrml_runtime::remove_file(&state_path)?;
        mrml_runtime::remove_file(&original_path)?;
        Ok(original)
    }

    fn materialize_index(&self, target: &Index) -> Result<(), RepositoryError> {
        let current = self.index()?;
        let mut files: Vector<(Text, Vector<u8>)> = Vector::new();
        for entry in &target.entries {
            files.push((entry.path.clone(), self.read_blob(entry.id)?));
        }
        for entry in &current.entries {
            if target.entry(&entry.path).is_none() {
                let path = join_path(&self.worktree, &entry.path);
                if path_is_file(&path) {
                    mrml_runtime::remove_file(&path)?;
                }
            }
        }
        for (relative, contents) in files {
            let path = join_path(&self.worktree, &relative);
            if let Some(parent) = parent_path(&path) {
                create_dir_all(parent)?;
            }
            write_file(&path, &contents)?;
        }
        self.write_index(target)
    }

    fn write_index(&self, index: &Index) -> Result<(), RepositoryError> {
        let encoded = index.encode()?;
        let lock = join_path(&self.git_dir, "index.lock");
        if path_exists(&lock) {
            return Err(RepositoryError::AlreadyExists);
        }
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
            if object.kind != ObjectKind::Blob {
                return Err(RepositoryError::InvalidReference);
            }
            files.push((path.clone(), object.contents));
        }
        for (relative, contents) in files {
            let path = join_path(&self.worktree, &relative);
            if let Some(parent) = parent_path(&path) {
                create_dir_all(parent)?;
            }
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

    pub fn diff(&self, staged: bool, paths: &[Text]) -> Result<Vector<FileDiff>, RepositoryError> {
        for path in paths {
            validate_worktree_path(path)?;
        }
        let index = self.index()?;
        let base = if staged {
            match self.head()? {
                Some(id) => self.tree_index(self.read_commit(id)?.tree)?,
                None => Index::empty(),
            }
        } else {
            index.clone()
        };
        let mut names: Vector<Text> = base
            .entries
            .iter()
            .filter(|entry| entry.stage == 0)
            .map(|entry| entry.path.clone())
            .collect();
        if staged {
            for entry in &index.entries {
                if entry.stage == 0 && !names.iter().any(|name| name == &entry.path) {
                    names.push(entry.path.clone());
                }
            }
        }
        names.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        let mut output = Vector::new();
        for name in names {
            if !paths.is_empty()
                && !paths.iter().any(|path| {
                    name.as_str() == path.as_str()
                        || name
                            .strip_prefix(path.as_str())
                            .is_some_and(|rest| rest.starts_with('/'))
                })
            {
                continue;
            }
            let old_entry = base.entry(&name);
            let new_entry = if staged {
                index.entry(&name)
            } else {
                old_entry
            };
            let old = match old_entry {
                Some(entry) => Some(self.read_blob(entry.id)?),
                None => None,
            };
            let new = if staged {
                match new_entry {
                    Some(entry) => Some(self.read_blob(entry.id)?),
                    None => None,
                }
            } else {
                let path = join_path(&self.worktree, &name);
                if path_is_file(&path) {
                    Some(read_file_bounded(&path, MAX_WORKTREE_FILE)?)
                } else {
                    None
                }
            };
            if old != new {
                output.push(FileDiff {
                    path: name,
                    old,
                    new,
                });
            }
        }
        Ok(output)
    }

    pub fn diff_revision(
        &self,
        revision: &str,
        paths: &[Text],
    ) -> Result<Vector<FileDiff>, RepositoryError> {
        for path in paths {
            validate_worktree_path(path)?;
        }
        let id = self.resolve_revision(revision)?;
        let base = self.tree_index(self.read_commit(id)?.tree)?;
        let mut names: Vector<Text> = base
            .entries
            .iter()
            .map(|entry| entry.path.clone())
            .collect();
        let index = self.index()?;
        for entry in &index.entries {
            if entry.stage == 0 && !names.iter().any(|name| name == &entry.path) {
                names.push(entry.path.clone());
            }
        }
        names.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        let mut output = Vector::new();
        for name in names {
            if !paths.is_empty()
                && !paths.iter().any(|path| {
                    name.as_str() == path.as_str()
                        || name
                            .strip_prefix(path.as_str())
                            .is_some_and(|rest| rest.starts_with('/'))
                })
            {
                continue;
            }
            let old = match base.entry(&name) {
                Some(entry) => Some(self.read_blob(entry.id)?),
                None => None,
            };
            let disk = join_path(&self.worktree, &name);
            let new = if path_is_file(&disk) {
                Some(read_file_bounded(&disk, MAX_WORKTREE_FILE)?)
            } else {
                None
            };
            if old != new {
                output.push(FileDiff {
                    path: name,
                    old,
                    new,
                });
            }
        }
        Ok(output)
    }

    pub fn conflicted_paths(&self) -> Result<Vector<Text>, RepositoryError> {
        let index = self.index()?;
        let mut paths = Vector::new();
        for entry in &index.entries {
            if entry.stage != 0 && !paths.iter().any(|path| path == &entry.path) {
                paths.push(entry.path.clone());
            }
        }
        Ok(paths)
    }

    pub fn file_history(
        &self,
        path: &str,
        limit: usize,
    ) -> Result<Vector<(ObjectId, Commit)>, RepositoryError> {
        validate_worktree_path(path)?;
        let mut result = Vector::new();
        let mut next = self.head()?;
        let mut previous_blob = None;
        while let Some(id) = next {
            let commit = self.read_commit(id)?;
            let tree = self.tree_index(commit.tree)?;
            let blob = tree.entry(path).map(|entry| entry.id);
            if blob != previous_blob {
                result.push((id, commit.clone()));
                previous_blob = blob;
            }
            next = commit.parents.first().copied();
            if result.len() >= limit {
                break;
            }
        }
        Ok(result)
    }

    pub fn blame(&self, path: &str) -> Result<Vector<BlameLine>, RepositoryError> {
        validate_worktree_path(path)?;
        let head = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let mut child_commit = self.read_commit(head)?;
        let child_index = self.tree_index(child_commit.tree)?;
        let child_blob = child_index
            .entry(path)
            .ok_or(RepositoryError::ReferenceMissing)?
            .id;
        let mut child_text = Text::try_from_utf8(self.read_blob(child_blob)?)
            .map_err(|_| RepositoryError::UnsupportedFileType)?;
        let final_lines: Vector<Text> = split_lines(&child_text)
            .iter()
            .map(|line| Text::from(*line))
            .collect();
        let mut origins: Vector<(Option<usize>, ObjectId, Text)> = final_lines
            .iter()
            .enumerate()
            .map(|(index, _)| (Some(index), head, child_commit.author.clone()))
            .collect();
        loop {
            let Some(parent_id) = child_commit.parents.first().copied() else {
                break;
            };
            let parent_commit = self.read_commit(parent_id)?;
            let parent_index = self.tree_index(parent_commit.tree)?;
            let Some(parent_entry) = parent_index.entry(path) else {
                break;
            };
            let parent_text = Text::try_from_utf8(self.read_blob(parent_entry.id)?)
                .map_err(|_| RepositoryError::UnsupportedFileType)?;
            let mapping = line_mapping(&split_lines(&child_text), &split_lines(&parent_text));
            for (position, commit, author) in &mut origins {
                if let Some(child_position) = *position {
                    if let Some(parent_position) = mapping.get(child_position).copied().flatten() {
                        *position = Some(parent_position);
                        *commit = parent_id;
                        *author = parent_commit.author.clone();
                    } else {
                        *position = None;
                    }
                }
            }
            child_commit = parent_commit;
            child_text = parent_text;
        }
        let mut output = Vector::new();
        for (index, line) in final_lines.into_iter().enumerate() {
            let (_, commit, author) = origins[index].clone();
            output.push(BlameLine {
                commit,
                author,
                line_number: index + 1,
                text: line,
            });
        }
        Ok(output)
    }

    fn read_blob(&self, id: ObjectId) -> Result<Vector<u8>, RepositoryError> {
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Blob {
            return Err(RepositoryError::InvalidReference);
        }
        Ok(object.contents)
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

    pub fn create_signed_tag(
        &self,
        name: &str,
        message: &str,
        tagger: &str,
        email: &str,
        timestamp: u64,
        key: &RsaPrivateKey,
    ) -> Result<ObjectId, RepositoryError> {
        let reference = mrml_runtime::mrml_format!("refs/tags/{name}");
        validate_reference(&reference)?;
        let path = join_path(&self.git_dir, &reference);
        if path_exists(&path) {
            return Err(RepositoryError::ReferenceExists);
        }
        validate_identity(tagger, email)?;
        if message.trim().is_empty() || message.contains('\0') {
            return Err(RepositoryError::InvalidIdentity);
        }
        let target = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let mut unsigned = mrml_runtime::mrml_format!(
            "object {target}\ntype commit\ntag {name}\ntagger {tagger} <{email}> {timestamp} +0000\n\n"
        );
        unsigned.push_str(message.trim());
        unsigned.push('\n');
        let signature =
            sign_sshsig(key, "git", unsigned.as_bytes()).map_err(|_| RepositoryError::Signing)?;
        let mut contents = unsigned;
        contents.push_str(&signature);
        let id = self.write_object(ObjectKind::Tag, contents.as_bytes())?;
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        write_file(&path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        Ok(id)
    }

    pub fn verify_tag_signature(
        &self,
        id: ObjectId,
        key: &RsaPublicKey,
    ) -> Result<(), RepositoryError> {
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Tag {
            return Err(RepositoryError::InvalidReference);
        }
        let marker = b"-----BEGIN SSH SIGNATURE-----";
        let offset = object
            .contents
            .windows(marker.len())
            .position(|part| part == marker)
            .ok_or(RepositoryError::Signing)?;
        if offset == 0 || object.contents[offset - 1] != b'\n' {
            return Err(RepositoryError::Signing);
        }
        let unsigned = &object.contents[..offset];
        let armor = core::str::from_utf8(&object.contents[offset..])
            .map_err(|_| RepositoryError::Signing)?;
        verify_sshsig(key, "git", unsigned, armor).map_err(|_| RepositoryError::Signing)
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
                    if key.trim() == "url" {
                        output.push((name.clone(), value.trim().into()));
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn set_remote(
        &self,
        name: &str,
        url: &str,
        require_existing: bool,
    ) -> Result<(), RepositoryError> {
        validate_config_name(name)?;
        if url.is_empty() || url.chars().any(char::is_control) {
            return Err(RepositoryError::InvalidReference);
        }
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
                if in_target && !wrote_url {
                    output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n"));
                }
                in_target = target == line;
                found |= in_target;
                wrote_url = false;
            }
            if in_target
                && line
                    .split_once('=')
                    .is_some_and(|(key, _)| key.trim() == "url")
            {
                output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n"));
                wrote_url = true;
            } else {
                output.push_str(raw);
                output.push('\n');
            }
        }
        if in_target && !wrote_url {
            output.push_str(&mrml_runtime::mrml_format!("\turl = {url}\n"));
        }
        if !found {
            if require_existing {
                return Err(RepositoryError::ReferenceMissing);
            }
            output.push_str(&mrml_runtime::mrml_format!(
                "\n{target}\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/{name}/*\n"
            ));
        }
        write_file(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn update_remote_ref(
        &self,
        remote: &str,
        source: &str,
        id: ObjectId,
    ) -> Result<(), RepositoryError> {
        validate_config_name(remote)?;
        let branch = source
            .strip_prefix("refs/heads/")
            .ok_or(RepositoryError::InvalidReference)?;
        validate_reference(source)?;
        let reference = mrml_runtime::mrml_format!("refs/remotes/{remote}/{branch}");
        validate_reference(&reference)?;
        let path = join_path(&self.git_dir, &reference);
        if let Some(parent) = parent_path(&path) {
            create_dir_all(parent)?;
        }
        write_file(&path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        Ok(())
    }

    pub fn config_value(&self, section: &str, key: &str) -> Result<Option<Text>, RepositoryError> {
        validate_config_name(section)?;
        validate_config_name(key)?;
        let config = read_file_text_bounded(&join_path(&self.git_dir, "config"), 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[{section}]");
        let mut active = false;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                active = target == line;
            } else if active {
                if let Some((found, value)) = line.split_once('=') {
                    if found.trim() == key {
                        return Ok(Some(value.trim().into()));
                    }
                }
            }
        }
        Ok(None)
    }

    pub fn set_config_value(
        &self,
        section: &str,
        key: &str,
        value: &str,
    ) -> Result<(), RepositoryError> {
        validate_config_name(section)?;
        validate_config_name(key)?;
        if value.is_empty() || value.chars().any(char::is_control) {
            return Err(RepositoryError::InvalidReference);
        }
        let path = join_path(&self.git_dir, "config");
        let config = read_file_text_bounded(&path, 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[{section}]");
        let mut output = Text::new();
        let mut active = false;
        let mut found_section = false;
        let mut wrote = false;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                if active && !wrote {
                    output.push_str(&mrml_runtime::mrml_format!("\t{key} = {value}\n"));
                    wrote = true;
                }
                active = target == line;
                found_section |= active;
            }
            if active
                && line
                    .split_once('=')
                    .is_some_and(|(found, _)| found.trim() == key)
            {
                output.push_str(&mrml_runtime::mrml_format!("\t{key} = {value}\n"));
                wrote = true;
            } else {
                output.push_str(raw);
                output.push('\n');
            }
        }
        if active && !wrote {
            output.push_str(&mrml_runtime::mrml_format!("\t{key} = {value}\n"));
        }
        if !found_section {
            output.push_str(&mrml_runtime::mrml_format!(
                "\n{target}\n\t{key} = {value}\n"
            ));
        }
        write_file(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn set_upstream(&self, spec: &str) -> Result<(), RepositoryError> {
        let (remote, branch) = spec
            .split_once('/')
            .ok_or(RepositoryError::InvalidReference)?;
        validate_config_name(remote)?;
        validate_reference(&mrml_runtime::mrml_format!("refs/heads/{branch}"))?;
        self.resolve_revision(&mrml_runtime::mrml_format!(
            "refs/remotes/{remote}/{branch}"
        ))?;
        let current = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let path = join_path(&self.git_dir, "config");
        let config = read_file_text_bounded(&path, 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[branch \"{current}\"]");
        let mut output = Text::new();
        let mut active = false;
        let mut found = false;
        let mut remote_written = false;
        let mut merge_written = false;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                if active {
                    if !remote_written {
                        output.push_str(&mrml_runtime::mrml_format!("\tremote = {remote}\n"));
                    }
                    if !merge_written {
                        output.push_str(&mrml_runtime::mrml_format!(
                            "\tmerge = refs/heads/{branch}\n"
                        ));
                    }
                }
                active = target == line;
                found |= active;
                remote_written = false;
                merge_written = false;
            }
            if active
                && line
                    .split_once('=')
                    .is_some_and(|(key, _)| key.trim() == "remote")
            {
                output.push_str(&mrml_runtime::mrml_format!("\tremote = {remote}\n"));
                remote_written = true;
            } else if active
                && line
                    .split_once('=')
                    .is_some_and(|(key, _)| key.trim() == "merge")
            {
                output.push_str(&mrml_runtime::mrml_format!(
                    "\tmerge = refs/heads/{branch}\n"
                ));
                merge_written = true;
            } else {
                output.push_str(raw);
                output.push('\n');
            }
        }
        if active {
            if !remote_written {
                output.push_str(&mrml_runtime::mrml_format!("\tremote = {remote}\n"));
            }
            if !merge_written {
                output.push_str(&mrml_runtime::mrml_format!(
                    "\tmerge = refs/heads/{branch}\n"
                ));
            }
        }
        if !found {
            output.push_str(&mrml_runtime::mrml_format!(
                "\n{target}\n\tremote = {remote}\n\tmerge = refs/heads/{branch}\n"
            ));
        }
        write_file(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn upstream_status(&self) -> Result<Option<(Text, usize, usize)>, RepositoryError> {
        let branch = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let config = read_file_text_bounded(&join_path(&self.git_dir, "config"), 1024 * 1024)?;
        let target = mrml_runtime::mrml_format!("[branch \"{branch}\"]");
        let mut active = false;
        let mut remote = None;
        let mut merge: Option<Text> = None;
        for raw in config.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                active = target == line;
            } else if active {
                if let Some((key, value)) = line.split_once('=') {
                    match key.trim() {
                        "remote" => remote = Some(Text::from(value.trim())),
                        "merge" => merge = value.trim().strip_prefix("refs/heads/").map(Into::into),
                        _ => {}
                    }
                }
            }
        }
        let (Some(remote), Some(upstream_branch)) = (remote, merge) else {
            return Ok(None);
        };
        let local = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let remote_id = self.resolve_revision(&mrml_runtime::mrml_format!(
            "refs/remotes/{remote}/{upstream_branch}"
        ))?;
        let left = self.reachable_commits(local)?;
        let right = self.reachable_commits(remote_id)?;
        let ahead = left.iter().filter(|id| !right.contains(id)).count();
        let behind = right.iter().filter(|id| !left.contains(id)).count();
        Ok(Some((
            mrml_runtime::mrml_format!("{remote}/{upstream_branch}"),
            ahead,
            behind,
        )))
    }

    fn reachable_commits(&self, start: ObjectId) -> Result<Vector<ObjectId>, RepositoryError> {
        let mut pending = Vector::from([start]);
        let mut seen = Vector::new();
        while let Some(id) = pending.pop() {
            if seen.contains(&id) {
                continue;
            }
            if seen.len() >= 1_000_000 {
                return Err(RepositoryError::TooManyFiles);
            }
            let commit = self.read_commit(id)?;
            seen.push(id);
            pending.extend(commit.parents.iter().copied());
        }
        Ok(seen)
    }

    pub fn stage(&self, paths: &[Text]) -> Result<(), RepositoryError> {
        let mut index = self.index()?;
        for relative in paths {
            validate_worktree_path(relative)?;
            index.remove(relative);
            let disk_path = join_path(&self.worktree, relative);
            if !path_exists(&disk_path) {
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
        self.write_tree_from_index(&index)
    }

    fn write_tree_from_index(&self, index: &Index) -> Result<ObjectId, RepositoryError> {
        if index.entries.iter().any(|entry| entry.stage != 0) {
            return Err(RepositoryError::ConflictedIndex);
        }
        self.write_tree_prefix(index, "")
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
        self.commit_internal(message, name, email, timestamp, None)
    }

    pub fn commit_signed(
        &self,
        message: &str,
        name: &str,
        email: &str,
        timestamp: u64,
        key: &RsaPrivateKey,
    ) -> Result<ObjectId, RepositoryError> {
        self.commit_internal(message, name, email, timestamp, Some(key))
    }

    fn commit_internal(
        &self,
        message: &str,
        name: &str,
        email: &str,
        timestamp: u64,
        key: Option<&RsaPrivateKey>,
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
        let merge_head_path = join_path(&self.git_dir, "MERGE_HEAD");
        if path_is_file(&merge_head_path) {
            let value = read_file_text_bounded(&merge_head_path, 4096)?;
            let merge_parent =
                ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
            contents.push_str(&mrml_runtime::mrml_format!("parent {merge_parent}\n"));
        }
        contents.push_str(&mrml_runtime::mrml_format!(
            "author {} <{}> {} +0000\ncommitter {} <{}> {} +0000\n",
            name,
            email,
            timestamp,
            name,
            email,
            timestamp
        ));
        let mut unsigned = contents.clone();
        unsigned.push('\n');
        unsigned.push_str(message.trim());
        unsigned.push('\n');
        if let Some(key) = key {
            let armor = sign_sshsig(key, "git", unsigned.as_bytes())
                .map_err(|_| RepositoryError::Signing)?;
            for (index, line) in armor.lines().enumerate() {
                if index == 0 {
                    contents.push_str("gpgsig ");
                } else {
                    contents.push(' ');
                }
                contents.push_str(line);
                contents.push('\n');
            }
            contents.push('\n');
            contents.push_str(message.trim());
            contents.push('\n');
        } else {
            contents = unsigned;
        }
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
        if path_is_file(&merge_head_path) {
            mrml_runtime::remove_file(&merge_head_path)?;
        }
        let cherry_pick = join_path(&self.git_dir, "CHERRY_PICK_HEAD");
        if path_is_file(&cherry_pick) {
            mrml_runtime::remove_file(&cherry_pick)?;
        }
        let original = join_path(&self.git_dir, "ORIG_HEAD");
        if path_is_file(&original) {
            mrml_runtime::remove_file(&original)?;
        }
        Ok(id)
    }

    pub fn verify_commit_signature(
        &self,
        id: ObjectId,
        key: &RsaPublicKey,
    ) -> Result<(), RepositoryError> {
        let object = self.read_object(id)?;
        if object.kind != ObjectKind::Commit {
            return Err(RepositoryError::InvalidReference);
        }
        let (unsigned, armor) = split_commit_signature(&object.contents)?;
        verify_sshsig(key, "git", &unsigned, &armor).map_err(|_| RepositoryError::Signing)
    }

    fn create_commit_object(
        &self,
        tree: ObjectId,
        parents: &[ObjectId],
        message: &str,
        name: &str,
        email: &str,
        timestamp: u64,
    ) -> Result<ObjectId, RepositoryError> {
        validate_identity(name, email)?;
        let mut contents = mrml_runtime::mrml_format!("tree {tree}\n");
        for parent in parents {
            contents.push_str(&mrml_runtime::mrml_format!("parent {parent}\n"));
        }
        contents.push_str(&mrml_runtime::mrml_format!("author {name} <{email}> {timestamp} +0000\ncommitter {name} <{email}> {timestamp} +0000\n\n{}\n", message.trim()));
        self.write_object(ObjectKind::Commit, contents.as_bytes())
    }

    fn update_current_branch(&self, id: ObjectId) -> Result<(), RepositoryError> {
        let branch = self
            .current_branch()?
            .ok_or(RepositoryError::DetachedHead)?;
        let path = join_path(
            &self.git_dir,
            &mrml_runtime::mrml_format!("refs/heads/{branch}"),
        );
        let lock = mrml_runtime::mrml_format!("{path}.lock");
        if path_exists(&lock) {
            return Err(RepositoryError::AlreadyExists);
        }
        write_file(&lock, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        mrml_runtime::rename_file(&lock, &path)?;
        Ok(())
    }

    pub fn stash_push(
        &self,
        message: &str,
        name: &str,
        email: &str,
        timestamp: u64,
    ) -> Result<ObjectId, RepositoryError> {
        validate_identity(name, email)?;
        let head = self.head()?.ok_or(RepositoryError::ReferenceMissing)?;
        let baseline = self.tree_index(self.read_commit(head)?.tree)?;
        let mut snapshot = self.index()?;
        for entry in snapshot.clone().entries {
            let path = join_path(&self.worktree, &entry.path);
            if path_is_file(&path) {
                let contents = read_file_bounded(&path, MAX_WORKTREE_FILE)?;
                let id = self.write_object(ObjectKind::Blob, &contents)?;
                snapshot.upsert(IndexEntry {
                    path: entry.path.clone(),
                    id,
                    mode: entry.mode,
                    size: contents
                        .len()
                        .try_into()
                        .map_err(|_| RepositoryError::FileTooLarge)?,
                    stage: 0,
                });
            } else {
                snapshot.remove(&entry.path);
            }
        }
        if snapshot == baseline {
            return Err(RepositoryError::WorktreeDirty);
        }
        let tree = self.write_tree_from_index(&snapshot)?;
        let stash_path = join_path(&self.git_dir, "refs/stash");
        let previous = if path_is_file(&stash_path) {
            let value = read_file_text_bounded(&stash_path, 4096)?;
            Some(ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?)
        } else {
            None
        };
        let mut contents = mrml_runtime::mrml_format!("tree {tree}\n");
        if let Some(previous) = previous {
            contents.push_str(&mrml_runtime::mrml_format!("parent {previous}\n"));
        }
        contents.push_str(&mrml_runtime::mrml_format!("author {name} <{email}> {timestamp} +0000\ncommitter {name} <{email}> {timestamp} +0000\nmrml-stash-base {head}\n\n{}\n", if message.trim().is_empty() { "WIP" } else { message.trim() }));
        let id = self.write_object(ObjectKind::Commit, contents.as_bytes())?;
        if let Some(parent) = parent_path(&stash_path) {
            create_dir_all(parent)?;
        }
        write_file(&stash_path, mrml_runtime::mrml_format!("{id}\n").as_bytes())?;
        self.materialize_index(&baseline)?;
        Ok(id)
    }

    pub fn stash_list(&self, limit: usize) -> Result<Vector<(ObjectId, Commit)>, RepositoryError> {
        let path = join_path(&self.git_dir, "refs/stash");
        if !path_is_file(&path) {
            return Ok(Vector::new());
        }
        let value = read_file_text_bounded(&path, 4096)?;
        let id = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
        self.history(id, limit)
    }

    pub fn stash_pop(&self) -> Result<ObjectId, RepositoryError> {
        if !self.changes()?.is_empty() {
            return Err(RepositoryError::WorktreeDirty);
        }
        let path = join_path(&self.git_dir, "refs/stash");
        if !path_is_file(&path) {
            return Err(RepositoryError::ReferenceMissing);
        }
        let value = read_file_text_bounded(&path, 4096)?;
        let id = ObjectId::parse(value.trim()).ok_or(RepositoryError::InvalidReference)?;
        let commit = self.read_commit(id)?;
        let snapshot = self.tree_index(commit.tree)?;
        self.materialize_index(&snapshot)?;
        if let Some(previous) = commit.parents.first() {
            write_file(&path, mrml_runtime::mrml_format!("{previous}\n").as_bytes())?;
        } else {
            mrml_runtime::remove_file(&path)?;
        }
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

fn same_entry(left: Option<&IndexEntry>, right: Option<&IndexEntry>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => left.id == right.id && left.mode == right.mode,
        _ => false,
    }
}

fn validate_config_name(value: &str) -> Result<(), RepositoryError> {
    if value.is_empty()
        || value.starts_with('-')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        Err(RepositoryError::InvalidReference)
    } else {
        Ok(())
    }
}

fn split_commit_signature(contents: &[u8]) -> Result<(Vector<u8>, Text), RepositoryError> {
    let text = core::str::from_utf8(contents).map_err(|_| RepositoryError::Signing)?;
    let (headers, message) = text.split_once("\n\n").ok_or(RepositoryError::Signing)?;
    let mut unsigned = Text::new();
    let mut armor = Text::new();
    let mut signing = false;
    let mut found = false;
    for line in headers.split('\n') {
        if let Some(first) = line.strip_prefix("gpgsig ") {
            if found {
                return Err(RepositoryError::Signing);
            }
            found = true;
            signing = true;
            armor.push_str(first);
            armor.push('\n');
        } else if signing && line.starts_with(' ') {
            armor.push_str(&line[1..]);
            armor.push('\n');
        } else {
            signing = false;
            unsigned.push_str(line);
            unsigned.push('\n');
        }
    }
    if !found {
        return Err(RepositoryError::Signing);
    }
    unsigned.push('\n');
    unsigned.push_str(message);
    let mut bytes = Vector::new();
    bytes.extend(unsigned.bytes());
    Ok((bytes, armor))
}

fn remote_section(line: &str) -> Option<&str> {
    line.strip_prefix("[remote \"")?
        .strip_suffix("\"]")
        .filter(|name| validate_config_name(name).is_ok())
}

fn split_lines(text: &str) -> Vector<&str> {
    if text.is_empty() {
        Vector::new()
    } else {
        text.split_inclusive('\n').collect()
    }
}

fn line_mapping(child: &[&str], parent: &[&str]) -> Vector<Option<usize>> {
    const MAX_CELLS: usize = 4_000_000;
    let mut mapping = Vector::new();
    mapping.resize(child.len(), None);
    if child
        .len()
        .checked_mul(parent.len())
        .is_none_or(|cells| cells > MAX_CELLS)
    {
        return mapping;
    }
    let width = parent.len() + 1;
    let mut table = Vector::new();
    table.resize((child.len() + 1) * width, 0u32);
    for i in (0..child.len()).rev() {
        for j in (0..parent.len()).rev() {
            table[i * width + j] = if child[i] == parent[j] {
                table[(i + 1) * width + j + 1] + 1
            } else {
                table[(i + 1) * width + j].max(table[i * width + j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    while i < child.len() && j < parent.len() {
        if child[i] == parent[j] {
            mapping[i] = Some(j);
            i += 1;
            j += 1;
        } else if table[(i + 1) * width + j] >= table[i * width + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    mapping
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
impl From<PackError> for RepositoryError {
    fn from(value: PackError) -> Self {
        Self::Pack(value)
    }
}

impl fmt::Display for RepositoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => write!(formatter, "{error}"),
            Self::Index(error) => write!(formatter, "{error}"),
            Self::Object(error) => write!(formatter, "{error}"),
            Self::Pack(error) => write!(formatter, "{error}"),
            Self::WorktreeDirty => formatter.write_str("working tree is not clean"),
            Self::MergeRequired => {
                formatter.write_str("non-fast-forward merge requires native three-way merge")
            }
            Self::Signing => formatter.write_str("Git SSH signature is missing or invalid"),
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

    fn decode_hex(text: &str) -> Vector<u8> {
        (0..text.len() / 2)
            .map(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }
    fn signing_key() -> RsaPrivateKey {
        RsaPrivateKey {
            public: RsaPublicKey {
                modulus: decode_hex(
                    "9eff1e540991fee9de7c7ed50d5da16508d610090a52c9aa4c41bc868e93e7cc03a6cc766fb2dab78ba91e4315f6524e355fda2c8a71b372f012d43460c2c425c2ae763d96a20584bc030e3595cc9f2352f51288f8db5d398d55efc566381707b4df848444641093fc5c48ca894db8397b252d00d5d606fe377b09f3609850fb",
                ),
                exponent: Vector::from([1, 0, 1]),
            },
            private_exponent: decode_hex(
                "2187d1e08d2821e736497102035094a1d70c35d3823ed552b9c43f3aed4499e4b77c6cb0297c418de5c123a5a8330b467d111ad4bbd9a0ab839fa4eaeae108364d4ad3f439916be8a244f8071922b1918cce92b27fe5f6ed24a328b15030b3fb3e300166c651f5f457daef746c4051a7a0f035379dcacf3a164fb4aedd284a11",
            ),
        }
    }

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
        let packed = parse_pack(&repository.pack_reachable(commit).unwrap()).unwrap();
        assert!(
            packed
                .iter()
                .any(|object| object.id == commit && object.kind == ObjectKind::Commit)
        );
        assert!(packed.iter().any(|object| object.kind == ObjectKind::Tree));
        assert!(packed.iter().any(|object| object.kind == ObjectKind::Blob));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn creates_and_verifies_signed_commit() {
        let path = root("signed-commit");
        let repository = Repository::init(&path).unwrap();
        write_file(&join_path(&path, "tracked"), b"signed").unwrap();
        repository
            .stage(&Vector::from([Text::from("tracked")]))
            .unwrap();
        let key = signing_key();
        let id = repository
            .commit_signed("signed", "Signer", "signer@example.invalid", 7, &key)
            .unwrap();
        repository.verify_commit_signature(id, &key.public).unwrap();
        let object = repository.read_object(id).unwrap();
        assert!(object.contents.windows(7).any(|part| part == b"gpgsig "));
        let tag = repository
            .create_signed_tag("v1", "release", "Signer", "signer@example.invalid", 8, &key)
            .unwrap();
        repository.verify_tag_signature(tag, &key.public).unwrap();
        assert_eq!(repository.resolve_revision("v1"), Ok(tag));
        let mut wrong = key.public.clone();
        wrong.exponent = Vector::from([3]);
        assert_eq!(
            repository.verify_commit_signature(id, &wrong),
            Err(RepositoryError::Signing)
        );
        assert_eq!(
            repository.verify_tag_signature(tag, &wrong),
            Err(RepositoryError::Signing)
        );
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
        repository
            .stage(&Vector::from([Text::from("tracked")]))
            .unwrap();
        let main = repository
            .commit("main", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        write_file(&file, b"topic").unwrap();
        repository
            .stage(&Vector::from([Text::from("tracked")]))
            .unwrap();
        repository
            .commit("topic", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        assert_eq!(repository.switch_branch("main").unwrap(), main);
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"main");
        assert_eq!(
            repository.current_branch().unwrap().as_deref(),
            Some("main")
        );
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
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        write_file(&file, b"changed").unwrap();
        repository.stage(&paths).unwrap();
        repository.unstage(&paths).unwrap();
        assert_eq!(
            repository.index().unwrap().entry("tracked").unwrap().id,
            ObjectId::blob(b"base")
        );
        repository.restore(&paths).unwrap();
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"base");
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn adds_and_updates_remote_configuration_natively() {
        let path = root("remote");
        let repository = Repository::init(&path).unwrap();
        repository
            .set_remote("origin", "git@example.invalid:owner/repo.git", false)
            .unwrap();
        assert_eq!(
            repository.remotes().unwrap()[0],
            (
                Text::from("origin"),
                Text::from("git@example.invalid:owner/repo.git")
            )
        );
        repository
            .set_remote("origin", "ssh://git@example.invalid/other/repo.git", true)
            .unwrap();
        assert_eq!(
            repository.remotes().unwrap()[0].1,
            "ssh://git@example.invalid/other/repo.git"
        );
        assert!(
            repository
                .set_remote("missing", "git@example.invalid:x/y", true)
                .is_err()
        );
        let remote_id = ObjectId::blob(b"remote-tip");
        repository
            .update_remote_ref("origin", "refs/heads/main", remote_id)
            .unwrap();
        let remote_hex = remote_id.to_hex();
        assert_eq!(
            remote_hex,
            read_file_text_bounded(
                &join_path(&repository.git_dir, "refs/remotes/origin/main"),
                64
            )
            .unwrap()
            .trim()
        );
        assert!(
            repository
                .update_remote_ref("../escape", "refs/heads/main", remote_id)
                .is_err()
        );
        assert!(
            repository
                .update_remote_ref("origin", "refs/tags/not-a-branch", remote_id)
                .is_err()
        );
        repository
            .set_config_value("ssh", "privateKey", "key.pem")
            .unwrap();
        repository
            .set_config_value("ssh", "hostKey", "host.pub")
            .unwrap();
        repository
            .set_config_value("ssh", "privateKey", "new.pem")
            .unwrap();
        assert_eq!(
            repository
                .config_value("ssh", "privateKey")
                .unwrap()
                .as_deref(),
            Some("new.pem")
        );
        assert_eq!(
            repository
                .config_value("ssh", "hostKey")
                .unwrap()
                .as_deref(),
            Some("host.pub")
        );
        assert!(repository.set_config_value("ssh", "bad key", "x").is_err());
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn tracks_upstream_and_computes_native_divergence() {
        let path = root("upstream");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"base").unwrap();
        repository.stage(&paths).unwrap();
        let base = repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository
            .update_remote_ref("origin", "refs/heads/main", base)
            .unwrap();
        repository.set_upstream("origin/main").unwrap();
        assert_eq!(
            repository.upstream_status().unwrap(),
            Some((Text::from("origin/main"), 0, 0))
        );
        write_file(&file, b"local").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("local", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        assert_eq!(
            repository.upstream_status().unwrap(),
            Some((Text::from("origin/main"), 1, 0))
        );
        assert!(repository.set_upstream("../escape").is_err());
        assert!(repository.set_upstream("origin/missing").is_err());
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn computes_staged_and_worktree_diffs_natively() {
        let path = root("diff");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"base\n").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        write_file(&file, b"staged\n").unwrap();
        repository.stage(&paths).unwrap();
        assert_eq!(
            repository.diff(true, &[]).unwrap()[0].new.as_deref(),
            Some(&b"staged\n"[..])
        );
        write_file(&file, b"worktree\n").unwrap();
        assert_eq!(
            repository.diff(false, &[]).unwrap()[0].new.as_deref(),
            Some(&b"worktree\n"[..])
        );
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn filters_file_history_and_diffs_against_revision() {
        let path = root("file-history");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"one\n").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("one", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        write_file(&file, b"two\n").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("two", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        write_file(&file, b"three\n").unwrap();
        assert_eq!(repository.file_history("tracked", 10).unwrap().len(), 2);
        assert_eq!(
            repository.diff_revision("HEAD", &paths).unwrap()[0]
                .new
                .as_deref(),
            Some(&b"three\n"[..])
        );
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn attributes_lines_to_their_introducing_commits() {
        let path = root("blame");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"old\n").unwrap();
        repository.stage(&paths).unwrap();
        let old = repository
            .commit("old", "Old", "old@example.invalid", 1)
            .unwrap();
        write_file(&file, b"old\nnew\n").unwrap();
        repository.stage(&paths).unwrap();
        let new = repository
            .commit("new", "New", "new@example.invalid", 2)
            .unwrap();
        let blame = repository.blame("tracked").unwrap();
        assert_eq!(blame[0].commit, old);
        assert_eq!(blame[1].commit, new);
        assert!(blame[0].author.starts_with("Old "));
        assert!(blame[1].author.starts_with("New "));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn fast_forwards_current_branch_natively() {
        let path = root("merge-ff");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"base").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        write_file(&file, b"topic").unwrap();
        repository.stage(&paths).unwrap();
        let topic = repository
            .commit("topic", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        repository.switch_branch("main").unwrap();
        assert_eq!(
            repository.fast_forward("topic").unwrap(),
            MergeOutcome::FastForward(topic)
        );
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"topic");
        assert_eq!(repository.head().unwrap(), Some(topic));
        assert_eq!(
            repository.fast_forward("topic").unwrap(),
            MergeOutcome::UpToDate
        );
        assert_eq!(
            repository.checkout_branch_at("clone-tip", topic).unwrap(),
            topic
        );
        assert_eq!(
            repository.current_branch().unwrap().as_deref(),
            Some("clone-tip")
        );
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn pushes_lists_and_pops_stash_objects() {
        let path = root("stash");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"base").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        write_file(&file, b"saved").unwrap();
        let stash = repository
            .stash_push("checkpoint", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"base");
        assert_eq!(repository.stash_list(10).unwrap()[0].0, stash);
        assert_eq!(repository.stash_pop().unwrap(), stash);
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"saved");
        assert!(repository.stash_list(10).unwrap().is_empty());
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn creates_native_two_parent_three_way_merge() {
        let path = root("merge-three");
        let repository = Repository::init(&path).unwrap();
        let base_file = join_path(&path, "base");
        write_file(&base_file, b"base").unwrap();
        repository
            .stage(&Vector::from([Text::from("base")]))
            .unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        let topic_file = join_path(&path, "topic");
        write_file(&topic_file, b"topic").unwrap();
        repository
            .stage(&Vector::from([Text::from("topic")]))
            .unwrap();
        let topic = repository
            .commit("topic", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        repository.switch_branch("main").unwrap();
        let main_file = join_path(&path, "main");
        write_file(&main_file, b"main").unwrap();
        repository
            .stage(&Vector::from([Text::from("main")]))
            .unwrap();
        let ours = repository
            .commit("main", "MRML", "mrml@example.invalid", 3)
            .unwrap();
        let merged = match repository
            .merge("topic", "MRML", "mrml@example.invalid", 4)
            .unwrap()
        {
            MergeOutcome::Merged(id) => id,
            other => panic!("unexpected {other:?}"),
        };
        let commit = repository.read_commit(merged).unwrap();
        assert_eq!(&commit.parents[..], &[ours, topic]);
        assert!(path_is_file(&topic_file));
        assert!(path_is_file(&main_file));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn records_conflict_stages_and_aborts_merge() {
        let path = root("merge-conflict");
        let repository = Repository::init(&path).unwrap();
        let file = join_path(&path, "tracked");
        let paths = Vector::from([Text::from("tracked")]);
        write_file(&file, b"base\n").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        write_file(&file, b"theirs\n").unwrap();
        repository.stage(&paths).unwrap();
        repository
            .commit("theirs", "MRML", "mrml@example.invalid", 2)
            .unwrap();
        repository.switch_branch("main").unwrap();
        write_file(&file, b"ours\n").unwrap();
        repository.stage(&paths).unwrap();
        let ours = repository
            .commit("ours", "MRML", "mrml@example.invalid", 3)
            .unwrap();
        assert_eq!(
            repository
                .merge("topic", "MRML", "mrml@example.invalid", 4)
                .unwrap(),
            MergeOutcome::Conflicts(1)
        );
        assert_eq!(
            repository
                .index()
                .unwrap()
                .entries
                .iter()
                .filter(|entry| entry.path == "tracked")
                .count(),
            3
        );
        assert!(
            read_file_text_bounded(&file, 1024)
                .unwrap()
                .contains("<<<<<<< HEAD")
        );
        assert_eq!(repository.abort_merge().unwrap(), ours);
        assert_eq!(&*read_file_bounded(&file, 16).unwrap(), b"ours\n");
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn cherry_picks_commit_as_new_one_parent_commit() {
        let path = root("cherry");
        let repository = Repository::init(&path).unwrap();
        let base = join_path(&path, "base");
        write_file(&base, b"base").unwrap();
        repository
            .stage(&Vector::from([Text::from("base")]))
            .unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        let added = join_path(&path, "added");
        write_file(&added, b"picked").unwrap();
        repository
            .stage(&Vector::from([Text::from("added")]))
            .unwrap();
        let picked = repository
            .commit("picked", "Original", "original@example.invalid", 2)
            .unwrap();
        repository.switch_branch("main").unwrap();
        let own = join_path(&path, "own");
        write_file(&own, b"own").unwrap();
        repository
            .stage(&Vector::from([Text::from("own")]))
            .unwrap();
        let parent = repository
            .commit("own", "MRML", "mrml@example.invalid", 3)
            .unwrap();
        let id = match repository
            .cherry_pick(
                &picked.to_hex(),
                "Committer",
                "committer@example.invalid",
                4,
            )
            .unwrap()
        {
            MergeOutcome::Merged(id) => id,
            other => panic!("unexpected {other:?}"),
        };
        let commit = repository.read_commit(id).unwrap();
        assert_eq!(&commit.parents[..], &[parent]);
        assert!(commit.author.starts_with("Original "));
        assert!(path_is_file(&added));
        assert!(path_is_file(&own));
        remove_dir_all(&path).unwrap();
    }

    #[test]
    fn rebases_linear_commits_onto_target() {
        let path = root("rebase");
        let repository = Repository::init(&path).unwrap();
        let base = join_path(&path, "base");
        write_file(&base, b"base").unwrap();
        repository
            .stage(&Vector::from([Text::from("base")]))
            .unwrap();
        repository
            .commit("base", "MRML", "mrml@example.invalid", 1)
            .unwrap();
        repository.create_branch("topic", true).unwrap();
        let topic_file = join_path(&path, "topic");
        write_file(&topic_file, b"topic").unwrap();
        repository
            .stage(&Vector::from([Text::from("topic")]))
            .unwrap();
        let topic = repository
            .commit("topic", "Topic", "topic@example.invalid", 2)
            .unwrap();
        repository.switch_branch("main").unwrap();
        let main_file = join_path(&path, "main");
        write_file(&main_file, b"main").unwrap();
        repository
            .stage(&Vector::from([Text::from("main")]))
            .unwrap();
        let original = repository
            .commit("main", "Main", "main@example.invalid", 3)
            .unwrap();
        let head = match repository
            .rebase("topic", "Committer", "committer@example.invalid", 4)
            .unwrap()
        {
            RebaseOutcome::Rebased { count: 1, head } => head,
            other => panic!("unexpected {other:?}"),
        };
        assert_ne!(head, original);
        assert_eq!(repository.read_commit(head).unwrap().parents[0], topic);
        assert!(path_is_file(&topic_file));
        assert!(path_is_file(&main_file));
        remove_dir_all(&path).unwrap();
    }
}
