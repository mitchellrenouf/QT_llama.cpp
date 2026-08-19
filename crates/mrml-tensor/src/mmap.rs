use std::{fs::File, io, ops::Deref, ptr::NonNull};

/// Read-only native file mapping for zero-copy GGUF tensor access.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
}
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn map(file: &File) -> io::Result<Self> {
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "file is too large to map"))?;
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cannot map an empty file",
            ));
        }
        platform::map(file, len).map(|ptr| Self { ptr, len })
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
        // SAFETY: this read-only view remains mapped for self's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}
impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe { platform::unmap(self.ptr, self.len) }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::os::windows::io::AsRawHandle;
    pub fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        unsafe { mrml_windows::map_file_read_only(file.as_raw_handle().cast(), len) }
            .ok_or_else(io::Error::last_os_error)
    }
    pub unsafe fn unmap(ptr: NonNull<u8>, _: usize) {
        let _ = unsafe { mrml_windows::unmap_file(ptr, 0) };
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::os::fd::AsRawFd;
    pub fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        unsafe { mrml_linux::map_file_read_only(file.as_raw_fd(), len) }
            .ok_or_else(io::Error::last_os_error)
    }
    pub unsafe fn unmap(ptr: NonNull<u8>, len: usize) {
        let _ = unsafe { mrml_linux::unmap_file(ptr, len) };
    }
}

#[cfg(test)]
mod tests {
    use super::Mmap;
    use std::{fs::File, io::Write};
    #[test]
    fn maps_complete_file_read_only() {
        let path = std::env::temp_dir().join(format!("mrml-mmap-{}.bin", std::process::id()));
        let mut file = File::create(&path).unwrap();
        file.write_all(b"MRML native mapping").unwrap();
        drop(file);
        let file = File::open(&path).unwrap();
        let map = Mmap::map(&file).unwrap();
        assert_eq!(&*map, b"MRML native mapping");
        drop(map);
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
