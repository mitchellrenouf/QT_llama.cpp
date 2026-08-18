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
    use std::{ffi::c_void, os::windows::io::AsRawHandle};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            f: *mut c_void,
            a: *const c_void,
            p: u32,
            h: u32,
            l: u32,
            n: *const u16,
        ) -> *mut c_void;
        fn MapViewOfFile(m: *mut c_void, a: u32, h: u32, l: u32, b: usize) -> *mut c_void;
        fn UnmapViewOfFile(p: *const c_void) -> i32;
        fn CloseHandle(h: *mut c_void) -> i32;
    }
    pub fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        unsafe {
            let handle = CreateFileMappingW(
                file.as_raw_handle().cast(),
                std::ptr::null(),
                2,
                0,
                0,
                std::ptr::null(),
            );
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let view = MapViewOfFile(handle, 4, 0, 0, len);
            let error = view.is_null().then(io::Error::last_os_error);
            CloseHandle(handle);
            if let Some(error) = error {
                return Err(error);
            }
            Ok(NonNull::new_unchecked(view.cast()))
        }
    }
    pub unsafe fn unmap(ptr: NonNull<u8>, _: usize) {
        unsafe {
            UnmapViewOfFile(ptr.as_ptr().cast());
        }
    }
}

#[cfg(unix)]
mod platform {
    use super::*;
    use std::{ffi::c_void, os::fd::AsRawFd};
    unsafe extern "C" {
        fn mmap(a: *mut c_void, l: usize, p: i32, f: i32, fd: i32, o: isize) -> *mut c_void;
        fn munmap(a: *mut c_void, l: usize) -> i32;
    }
    pub fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        unsafe {
            let view = mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0);
            if view as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            NonNull::new(view.cast()).ok_or_else(io::Error::last_os_error)
        }
    }
    pub unsafe fn unmap(ptr: NonNull<u8>, len: usize) {
        unsafe {
            munmap(ptr.as_ptr().cast(), len);
        }
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
