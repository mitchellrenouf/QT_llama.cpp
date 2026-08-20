use crate::{Text, Vector};
use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileError {
    InvalidPath,
    OpenFailed,
    ReadFailed,
    UnexpectedEnd,
    SeekFailed,
    MetadataFailed,
    InvalidUtf8,
    WriteFailed,
    DirectoryFailed,
    RemoveFailed,
    RenameFailed,
}

pub struct DirectoryEntry {
    pub name: Text,
    pub is_directory: bool,
    pub is_symlink: bool,
}

pub fn read_file(path: &str) -> Result<Vector<u8>, FileError> {
    read_file_bounded(path, usize::MAX)
}

pub fn read_file_bounded(path: &str, limit: usize) -> Result<Vector<u8>, FileError> {
    let mut file = File::open(path)?;
    let mut bytes = Vector::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        if bytes
            .len()
            .checked_add(read)
            .is_none_or(|length| length > limit)
        {
            return Err(FileError::ReadFailed);
        }
        bytes
            .try_extend_from_slice(&chunk[..read])
            .map_err(|_| FileError::ReadFailed)?;
    }
    Ok(bytes)
}

pub fn read_file_text(path: &str) -> Result<Text, FileError> {
    Text::try_from_utf8(read_file(path)?).map_err(|_| FileError::InvalidUtf8)
}

pub fn read_file_text_bounded(path: &str, limit: usize) -> Result<Text, FileError> {
    Text::try_from_utf8(read_file_bounded(path, limit)?).map_err(|_| FileError::InvalidUtf8)
}

pub fn write_file(path: &str, contents: &[u8]) -> Result<(), FileError> {
    let mut file = File::create(path)?;
    file.write_all(contents)
}

impl fmt::Display for FileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidPath => "invalid file path",
            Self::OpenFailed => "failed to open file",
            Self::ReadFailed => "failed to read file",
            Self::UnexpectedEnd => "unexpected end of file",
            Self::SeekFailed => "failed to seek file",
            Self::MetadataFailed => "failed to read file metadata",
            Self::InvalidUtf8 => "file is not valid UTF-8 text",
            Self::WriteFailed => "failed to write file",
            Self::DirectoryFailed => "failed to create directory",
            Self::RemoveFailed => "failed to remove file",
            Self::RenameFailed => "failed to rename file",
        })
    }
}

#[cfg(windows)]
fn encode_windows_path(path: &str) -> Option<Vector<u16>> {
    let mut encoded = Vector::with_capacity(path.len() + 1).ok()?;
    encoded.extend(path.encode_utf16());
    encoded.push(0);
    Some(encoded)
}

#[cfg(unix)]
fn encode_unix_path(path: &str) -> Option<Vector<u8>> {
    if path.as_bytes().contains(&0) {
        return None;
    }
    let mut encoded = Vector::with_capacity(path.len() + 1).ok()?;
    encoded.try_extend_from_slice(path.as_bytes()).ok()?;
    encoded.push(0);
    Some(encoded)
}

pub fn path_is_directory(path: &str) -> bool {
    #[cfg(windows)]
    {
        encode_windows_path(path).is_some_and(|path| mrml_windows::wide_path_is_directory(&path))
    }
    #[cfg(unix)]
    {
        encode_unix_path(path).is_some_and(|path| {
            core::ffi::CStr::from_bytes_with_nul(&path).is_ok_and(mrml_linux::path_is_directory)
        })
    }
}

pub fn path_is_file(path: &str) -> bool {
    #[cfg(windows)]
    {
        encode_windows_path(path).is_some_and(|path| mrml_windows::wide_path_is_file(&path))
    }
    #[cfg(unix)]
    {
        !path_is_directory(path) && File::open(path).is_ok()
    }
}

pub fn path_exists(path: &str) -> bool {
    path_is_directory(path) || path_is_file(path)
}

pub fn path_is_absolute(path: &str) -> bool {
    if cfg!(windows) {
        path.starts_with(['/', '\\'])
            || (path.as_bytes().get(1) == Some(&b':')
                && path
                    .as_bytes()
                    .get(2)
                    .is_some_and(|byte| matches!(byte, b'/' | b'\\')))
    } else {
        path.starts_with('/')
    }
}

pub fn join_path(base: &str, child: &str) -> Text {
    if path_is_absolute(child) || base.is_empty() {
        return child.into();
    }
    let mut joined = Text::from(base.trim_end_matches(['/', '\\']));
    if !joined.is_empty() {
        joined.push(if cfg!(windows) { '\\' } else { '/' });
    }
    joined.push_str(child.trim_start_matches(['/', '\\']));
    joined
}

pub fn parent_path(path: &str) -> Option<&str> {
    let trimmed = path.trim_end_matches(['/', '\\']);
    let split = trimmed.rfind(['/', '\\'])?;
    if split == 0 {
        Some(&trimmed[..1])
    } else {
        Some(&trimmed[..split])
    }
}

pub fn canonical_path(path: &str) -> Result<Text, FileError> {
    #[cfg(windows)]
    {
        let encoded = encode_windows_path(path).ok_or(FileError::InvalidPath)?;
        const MAX_WINDOWS_PATH_UNITS: usize = 32_768;
        let mut output =
            Vector::with_capacity(MAX_WINDOWS_PATH_UNITS).map_err(|_| FileError::MetadataFailed)?;
        output.resize(MAX_WINDOWS_PATH_UNITS, 0);
        let length =
            mrml_windows::full_path_wide(&encoded, &mut output).ok_or(FileError::MetadataFailed)?;
        let mut result = Text::new();
        for character in core::char::decode_utf16(output[..length].iter().copied()) {
            result.push(character.map_err(|_| FileError::InvalidUtf8)?);
        }
        Ok(result)
    }
    #[cfg(unix)]
    {
        let encoded = encode_unix_path(path).ok_or(FileError::InvalidPath)?;
        let encoded =
            core::ffi::CStr::from_bytes_with_nul(&encoded).map_err(|_| FileError::InvalidPath)?;
        let mut output = [0u8; 4096];
        let path =
            mrml_linux::canonical_path(encoded, &mut output).ok_or(FileError::MetadataFailed)?;
        Text::try_from_str(core::str::from_utf8(path).map_err(|_| FileError::InvalidUtf8)?)
            .map_err(|_| FileError::MetadataFailed)
    }
}

pub fn create_dir_all(path: &str) -> Result<(), FileError> {
    let path = path.trim_end_matches(['/', '\\']);
    if path.is_empty() || path == "/" || path.ends_with(':') || path_is_directory(path) {
        return Ok(());
    }
    if let Some(split) = path.rfind(['/', '\\']) {
        let parent = &path[..split];
        if !parent.is_empty() && parent != path {
            create_dir_all(parent)?;
        }
    }
    #[cfg(windows)]
    let created =
        encode_windows_path(path).is_some_and(|path| mrml_windows::create_directory_wide(&path));
    #[cfg(unix)]
    let created = encode_unix_path(path).is_some_and(|path| {
        core::ffi::CStr::from_bytes_with_nul(&path).is_ok_and(mrml_linux::create_directory)
    });
    created.then_some(()).ok_or(FileError::DirectoryFailed)
}

pub fn remove_file(path: &str) -> Result<(), FileError> {
    #[cfg(windows)]
    let removed =
        encode_windows_path(path).is_some_and(|path| mrml_windows::delete_file_wide(&path));
    #[cfg(unix)]
    let removed = encode_unix_path(path).is_some_and(|path| {
        core::ffi::CStr::from_bytes_with_nul(&path).is_ok_and(mrml_linux::delete_file)
    });
    removed.then_some(()).ok_or(FileError::RemoveFailed)
}

fn remove_directory(path: &str) -> Result<(), FileError> {
    #[cfg(windows)]
    let removed =
        encode_windows_path(path).is_some_and(|path| mrml_windows::remove_directory_wide(&path));
    #[cfg(unix)]
    let removed = encode_unix_path(path).is_some_and(|path| {
        core::ffi::CStr::from_bytes_with_nul(&path).is_ok_and(mrml_linux::remove_directory)
    });
    removed.then_some(()).ok_or(FileError::RemoveFailed)
}

pub fn remove_dir_all(path: &str) -> Result<(), FileError> {
    for entry in read_directory(path)? {
        let mut child = Text::from(path.trim_end_matches(['/', '\\']));
        if !child.is_empty() && !child.ends_with(['/', '\\']) {
            child.push(if cfg!(windows) { '\\' } else { '/' });
        }
        child.push_str(&entry.name);
        if entry.is_directory && !entry.is_symlink {
            remove_dir_all(&child)?;
        } else if entry.is_directory {
            remove_directory(&child)?;
        } else {
            remove_file(&child)?;
        }
    }
    remove_directory(path)
}

pub fn rename_file(existing: &str, replacement: &str) -> Result<(), FileError> {
    #[cfg(windows)]
    let renamed = encode_windows_path(existing)
        .zip(encode_windows_path(replacement))
        .is_some_and(|(existing, replacement)| {
            mrml_windows::rename_file_wide(&existing, &replacement)
        });
    #[cfg(unix)]
    let renamed = encode_unix_path(existing)
        .zip(encode_unix_path(replacement))
        .is_some_and(|(existing, replacement)| {
            core::ffi::CStr::from_bytes_with_nul(&existing).is_ok_and(|existing| {
                core::ffi::CStr::from_bytes_with_nul(&replacement)
                    .is_ok_and(|replacement| mrml_linux::rename_file(existing, replacement))
            })
        });
    renamed.then_some(()).ok_or(FileError::RenameFailed)
}

pub fn read_directory(path: &str) -> Result<Vector<DirectoryEntry>, FileError> {
    let mut entries = Vector::new();
    #[cfg(windows)]
    {
        let mut pattern = encode_windows_path(path).ok_or(FileError::InvalidPath)?;
        pattern.pop();
        if !path.ends_with(['/', '\\']) {
            pattern.push('\\' as u16);
        }
        pattern.push('*' as u16);
        pattern.push(0);
        let mut directory =
            mrml_windows::NativeDirectory::open(&pattern).ok_or(FileError::DirectoryFailed)?;
        let mut wide_name = [0u16; 260];
        while let Some((length, is_directory, is_symlink)) = directory.next(&mut wide_name) {
            let mut name = Text::new();
            for character in core::char::decode_utf16(wide_name[..length].iter().copied()) {
                name.push(character.map_err(|_| FileError::InvalidUtf8)?);
            }
            if name != "." && name != ".." {
                entries.push(DirectoryEntry {
                    name,
                    is_directory,
                    is_symlink,
                });
            }
        }
    }
    #[cfg(unix)]
    {
        let encoded = encode_unix_path(path).ok_or(FileError::InvalidPath)?;
        let path =
            core::ffi::CStr::from_bytes_with_nul(&encoded).map_err(|_| FileError::InvalidPath)?;
        let mut directory =
            mrml_linux::NativeDirectory::open(path).ok_or(FileError::DirectoryFailed)?;
        let mut name_buffer = [0u8; 256];
        while let Some((name, is_directory, is_symlink)) = directory.next(&mut name_buffer) {
            let name = core::str::from_utf8(name).map_err(|_| FileError::InvalidUtf8)?;
            if name != "." && name != ".." {
                entries.push(DirectoryEntry {
                    name: name.into(),
                    is_directory,
                    is_symlink,
                });
            }
        }
    }
    Ok(entries)
}
impl core::error::Error for FileError {}

pub struct File {
    #[cfg(windows)]
    inner: mrml_windows::NativeFile,
    #[cfg(unix)]
    inner: mrml_linux::NativeFile,
    position: u64,
}

impl File {
    pub fn open(path: &str) -> Result<Self, FileError> {
        #[cfg(windows)]
        {
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            for unit in path.encode_utf16() {
                encoded.push(unit);
            }
            encoded.push(0);
            let inner =
                mrml_windows::NativeFile::open_read(&encoded).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
        #[cfg(unix)]
        {
            if path.as_bytes().contains(&0) {
                return Err(FileError::InvalidPath);
            }
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            encoded
                .try_extend_from_slice(path.as_bytes())
                .map_err(|_| FileError::OpenFailed)?;
            encoded.push(0);
            let path = core::ffi::CStr::from_bytes_with_nul(&encoded)
                .map_err(|_| FileError::InvalidPath)?;
            let inner = mrml_linux::NativeFile::open_read(path).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
    }

    pub fn create(path: &str) -> Result<Self, FileError> {
        #[cfg(windows)]
        {
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            encoded.extend(path.encode_utf16());
            encoded.push(0);
            let inner =
                mrml_windows::NativeFile::create_write(&encoded).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
        #[cfg(unix)]
        {
            if path.as_bytes().contains(&0) {
                return Err(FileError::InvalidPath);
            }
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            encoded
                .try_extend_from_slice(path.as_bytes())
                .map_err(|_| FileError::OpenFailed)?;
            encoded.push(0);
            let path = core::ffi::CStr::from_bytes_with_nul(&encoded)
                .map_err(|_| FileError::InvalidPath)?;
            let inner = mrml_linux::NativeFile::create_write(path).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
    }

    pub fn open_write(path: &str) -> Result<Self, FileError> {
        #[cfg(windows)]
        {
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            encoded.extend(path.encode_utf16());
            encoded.push(0);
            let inner =
                mrml_windows::NativeFile::open_write(&encoded).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
        #[cfg(unix)]
        {
            if path.as_bytes().contains(&0) {
                return Err(FileError::InvalidPath);
            }
            let mut encoded =
                Vector::with_capacity(path.len() + 1).map_err(|_| FileError::OpenFailed)?;
            encoded
                .try_extend_from_slice(path.as_bytes())
                .map_err(|_| FileError::OpenFailed)?;
            encoded.push(0);
            let path = core::ffi::CStr::from_bytes_with_nul(&encoded)
                .map_err(|_| FileError::InvalidPath)?;
            let inner = mrml_linux::NativeFile::open_write(path).ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
    }

    pub fn read_exact(&mut self, mut buffer: &mut [u8]) -> Result<(), FileError> {
        while !buffer.is_empty() {
            let read = self.inner.read(buffer).ok_or(FileError::ReadFailed)?;
            if read == 0 {
                return Err(FileError::UnexpectedEnd);
            }
            self.position = self.position.saturating_add(read as u64);
            buffer = &mut buffer[read..];
        }
        Ok(())
    }

    pub fn read(&mut self, buffer: &mut [u8]) -> Result<usize, FileError> {
        let read = self.inner.read(buffer).ok_or(FileError::ReadFailed)?;
        self.position = self.position.saturating_add(read as u64);
        Ok(read)
    }

    pub fn write_all(&mut self, mut buffer: &[u8]) -> Result<(), FileError> {
        while !buffer.is_empty() {
            let written = self.inner.write(buffer).ok_or(FileError::WriteFailed)?;
            if written == 0 {
                return Err(FileError::WriteFailed);
            }
            self.position = self.position.saturating_add(written as u64);
            buffer = &buffer[written..];
        }
        Ok(())
    }

    pub fn seek(&mut self, position: u64) -> Result<(), FileError> {
        self.inner
            .seek_absolute(position)
            .then_some(())
            .ok_or(FileError::SeekFailed)?;
        self.position = position;
        Ok(())
    }

    pub fn len(&self) -> Result<u64, FileError> {
        self.inner.len().ok_or(FileError::MetadataFailed)
    }

    pub const fn position(&self) -> u64 {
        self.position
    }

    #[cfg(windows)]
    pub fn raw_handle(&self) -> *mut core::ffi::c_void {
        self.inner.raw_handle()
    }
    #[cfg(unix)]
    pub fn raw_fd(&self) -> core::ffi::c_int {
        self.inner.raw_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> Text {
        crate::join_path(
            &crate::temporary_directory(),
            &crate::mrml_format!("{name}-{}", crate::process_id()),
        )
    }

    #[test]
    fn native_file_reads_seeks_and_reports_length() {
        let path = test_path("mrml-file.bin");
        write_file(&path, b"native file").unwrap();

        let mut file = File::open(&path).unwrap();
        assert_eq!(file.len().unwrap(), 11);
        file.seek(7).unwrap();
        let mut suffix = [0; 4];
        file.read_exact(&mut suffix).unwrap();
        assert_eq!(&suffix, b"file");
        drop(file);
        remove_file(&path).unwrap();
    }
    #[test]
    fn writes_truncates_and_reads_utf8_text_through_native_file() {
        let path = test_path("mrml-text.txt");
        write_file(&path, b"a longer discarded value").unwrap();
        write_file(&path, "observatory λ".as_bytes()).unwrap();
        assert_eq!(read_file_text(&path).unwrap(), "observatory λ");
        remove_file(&path).unwrap();
    }

    #[test]
    fn bounded_reads_reject_oversized_files() {
        let path = test_path("mrml-bounded.bin");
        write_file(&path, b"12345").unwrap();
        assert_eq!(&read_file_bounded(&path, 5).unwrap()[..], b"12345");
        assert_eq!(read_file_bounded(&path, 4), Err(FileError::ReadFailed));
        remove_file(&path).unwrap();
    }

    #[test]
    fn renames_replaces_and_removes_files_natively() {
        let root = test_path("mrml-rename");
        create_dir_all(&root).unwrap();
        let source = join_path(&root, "source-λ.txt");
        let target = join_path(&root, "target-λ.txt");
        write_file(&source, b"new").unwrap();
        write_file(&target, b"old").unwrap();
        rename_file(&source, &target).unwrap();
        assert_eq!(&*read_file(&target).unwrap(), b"new");
        assert!(!path_exists(&source));
        remove_file(&target).unwrap();
        assert!(!path_exists(&target));
        remove_dir_all(&root).unwrap();
    }

    #[test]
    fn enumerates_unicode_files_and_directories_natively() {
        let root = test_path("mrml-enumerate");
        let child = join_path(&root, "folder-星");
        create_dir_all(&child).unwrap();
        let file = join_path(&root, "file-λ.txt");
        write_file(&file, b"entry").unwrap();
        let entries = read_directory(&root).unwrap();
        assert!(
            entries
                .iter()
                .any(|entry| entry.name == "folder-星" && entry.is_directory)
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.name == "file-λ.txt" && !entry.is_directory)
        );
        remove_dir_all(&root).unwrap();
        assert!(!path_exists(&root));
    }

    #[test]
    fn resolves_existing_paths_natively() {
        let root = test_path("mrml-canonical");
        create_dir_all(&root).unwrap();
        let resolved = canonical_path(&root).unwrap();
        assert!(path_is_directory(&resolved));
        assert!(resolved.contains("mrml-canonical-"));
        remove_dir_all(&root).unwrap();
    }

    #[test]
    fn recursively_creates_unicode_directories_and_rejects_file_conflicts() {
        let root = test_path("mrml-dir");
        let nested = join_path(&join_path(&root, "observatory-λ"), "deep");
        create_dir_all(&nested).unwrap();
        assert!(path_is_directory(&nested));
        let file = join_path(&nested, "conflict");
        write_file(&file, b"not a directory").unwrap();
        assert!(create_dir_all(&file).is_err());
        remove_dir_all(&root).unwrap();
        assert!(!path_exists(&root));
    }
}
