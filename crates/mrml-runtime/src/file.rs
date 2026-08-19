use crate::Vector;
use core::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileError {
    InvalidPath,
    OpenFailed,
    ReadFailed,
    UnexpectedEnd,
    SeekFailed,
    MetadataFailed,
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
        })
    }
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
    use std::io::Write;

    #[test]
    fn native_file_reads_seeks_and_reports_length() {
        let path = std::env::temp_dir().join(std::format!("mrml-file-{}.bin", std::process::id()));
        let mut created = std::fs::File::create(&path).unwrap();
        created.write_all(b"native file").unwrap();
        drop(created);

        let mut file = File::open(path.to_str().unwrap()).unwrap();
        assert_eq!(file.len().unwrap(), 11);
        file.seek(7).unwrap();
        let mut suffix = [0; 4];
        file.read_exact(&mut suffix).unwrap();
        assert_eq!(&suffix, b"file");
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
