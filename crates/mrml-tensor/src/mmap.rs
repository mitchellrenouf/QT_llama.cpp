use core::fmt;
use core::ops::Deref;
use core::ptr::NonNull;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapError {
    Empty,
    Platform,
}

impl fmt::Display for MapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "cannot map an empty file",
            Self::Platform => "native read-only file mapping failed",
        })
    }
}

impl core::error::Error for MapError {}

/// Read-only native file mapping for zero-copy GGUF tensor access.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
}
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// # Safety
    /// `file` must be a valid open file handle/descriptor containing at least
    /// `len` bytes. The mapping owns its view but does not own the file handle.
    #[cfg(windows)]
    pub unsafe fn map_raw(file: *mut core::ffi::c_void, len: usize) -> Result<Self, MapError> {
        if len == 0 {
            return Err(MapError::Empty);
        }
        unsafe { mrml_windows::map_file_read_only(file, len) }
            .map(|ptr| Self { ptr, len })
            .ok_or(MapError::Platform)
    }

    /// # Safety
    /// `file` must be a valid open file descriptor containing at least `len`
    /// bytes. The mapping owns its view but does not own the descriptor.
    #[cfg(unix)]
    pub unsafe fn map_raw(file: i32, len: usize) -> Result<Self, MapError> {
        if len == 0 {
            return Err(MapError::Empty);
        }
        unsafe { mrml_linux::map_file_read_only(file, len) }
            .map(|ptr| Self { ptr, len })
            .ok_or(MapError::Platform)
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for Mmap {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(windows)]
        let _ = unsafe { mrml_windows::unmap_file(self.ptr, 0) };
        #[cfg(unix)]
        let _ = unsafe { mrml_linux::unmap_file(self.ptr, self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::Mmap;
    use mrml_runtime::{File, join_path, mrml_format as format, process_id, temporary_directory};

    #[test]
    fn maps_complete_file_read_only() {
        let path = join_path(&temporary_directory(), &format!("mrml-mmap-{}.bin", process_id()));
        mrml_runtime::write_file(&path, b"MRML native mapping").unwrap();
        let file = File::open(&path).unwrap();
        #[cfg(windows)]
        let map = unsafe { Mmap::map_raw(file.raw_handle(), 19) }.unwrap();
        #[cfg(unix)]
        let map = unsafe { Mmap::map_raw(file.raw_fd(), 19) }.unwrap();
        assert_eq!(&*map, b"MRML native mapping");
        drop(map);
        drop(file);
        mrml_runtime::remove_file(&path).unwrap();
    }
}
