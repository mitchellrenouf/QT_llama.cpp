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
}

pub fn read_file(path: &str) -> Result<Vector<u8>, FileError> {
    let mut file = File::open(path)?;
    let mut bytes = Vector::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
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
        encode_windows_path(path)
            .is_some_and(|path| mrml_windows::wide_path_is_directory(&path))
    }
    #[cfg(unix)]
    {
        encode_unix_path(path).is_some_and(|path| {
            core::ffi::CStr::from_bytes_with_nul(&path)
                .is_ok_and(mrml_linux::path_is_directory)
        })
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
    let created = encode_windows_path(path)
        .is_some_and(|path| mrml_windows::create_directory_wide(&path));
    #[cfg(unix)]
    let created = encode_unix_path(path).is_some_and(|path| {
        core::ffi::CStr::from_bytes_with_nul(&path)
            .is_ok_and(mrml_linux::create_directory)
    });
    created.then_some(()).ok_or(FileError::DirectoryFailed)
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
            let mut encoded = Vector::with_capacity(path.len() + 1)
                .map_err(|_| FileError::OpenFailed)?;
            encoded.extend(path.encode_utf16());
            encoded.push(0);
            let inner = mrml_windows::NativeFile::create_write(&encoded)
                .ok_or(FileError::OpenFailed)?;
            Ok(Self { inner, position: 0 })
        }
        #[cfg(unix)]
        {
            if path.as_bytes().contains(&0) {
                return Err(FileError::InvalidPath);
            }
            let mut encoded = Vector::with_capacity(path.len() + 1)
                .map_err(|_| FileError::OpenFailed)?;
            encoded
                .try_extend_from_slice(path.as_bytes())
                .map_err(|_| FileError::OpenFailed)?;
            encoded.push(0);
            let path = core::ffi::CStr::from_bytes_with_nul(&encoded)
                .map_err(|_| FileError::InvalidPath)?;
            let inner = mrml_linux::NativeFile::create_write(path)
                .ok_or(FileError::OpenFailed)?;
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

    #[test]
    fn native_file_reads_seeks_and_reports_length() {
        let path = std::env::temp_dir().join(std::format!("mrml-file-{}.bin", std::process::id()));
        write_file(path.to_str().unwrap(), b"native file").unwrap();

        let mut file = File::open(path.to_str().unwrap()).unwrap();
        assert_eq!(file.len().unwrap(), 11);
        file.seek(7).unwrap();
        let mut suffix = [0; 4];
        file.read_exact(&mut suffix).unwrap();
        assert_eq!(&suffix, b"file");
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn writes_truncates_and_reads_utf8_text_through_native_file() {
        let path = std::env::temp_dir().join(std::format!("mrml-text-{}.txt", std::process::id()));
        write_file(path.to_str().unwrap(), b"a longer discarded value").unwrap();
        write_file(path.to_str().unwrap(), "observatory λ".as_bytes()).unwrap();
        assert_eq!(read_file_text(path.to_str().unwrap()).unwrap(), "observatory λ");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn recursively_creates_unicode_directories_and_rejects_file_conflicts() {
        let root = std::env::temp_dir().join(std::format!("mrml-dir-{}", std::process::id()));
        let nested = root.join("observatory-λ").join("deep");
        create_dir_all(nested.to_str().unwrap()).unwrap();
        assert!(path_is_directory(nested.to_str().unwrap()));
        let file = nested.join("conflict");
        write_file(file.to_str().unwrap(), b"not a directory").unwrap();
        assert!(create_dir_all(file.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
