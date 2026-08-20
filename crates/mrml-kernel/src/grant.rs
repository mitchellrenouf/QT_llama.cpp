pub const MAX_HOST_PATH: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrantMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrantError {
    Empty,
    TooLong,
    NotAbsolute,
    InvalidCharacter,
    Traversal,
}

/// A host directory grant copied into kernel-owned fixed storage. Host VMMs
/// must canonicalize the path and open a directory handle before constructing
/// this value; pathname checks alone are not an authorization boundary.
#[derive(Clone, Eq, PartialEq)]
pub struct DirectoryGrant {
    path: [u8; MAX_HOST_PATH],
    length: u16,
    mode: GrantMode,
}

impl DirectoryGrant {
    pub fn new(canonical_path: &str, mode: GrantMode) -> Result<Self, GrantError> {
        let bytes = canonical_path.as_bytes();
        if bytes.is_empty() {
            return Err(GrantError::Empty);
        }
        if bytes.len() > MAX_HOST_PATH {
            return Err(GrantError::TooLong);
        }
        let windows_absolute = bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\');
        if bytes[0] != b'/' && !windows_absolute {
            return Err(GrantError::NotAbsolute);
        }
        if bytes.iter().any(|byte| *byte == 0 || *byte < 0x20) {
            return Err(GrantError::InvalidCharacter);
        }
        if canonical_path
            .split(['/', '\\'])
            .any(|component| component == "..")
        {
            return Err(GrantError::Traversal);
        }
        let mut path = [0; MAX_HOST_PATH];
        path[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            path,
            length: bytes.len() as u16,
            mode,
        })
    }

    pub fn path(&self) -> &str {
        core::str::from_utf8(&self.path[..self.length as usize])
            .expect("DirectoryGrant is constructed from UTF-8")
    }

    pub const fn mode(&self) -> GrantMode {
        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_explicit_absolute_canonical_paths() {
        assert!(DirectoryGrant::new("/models", GrantMode::ReadOnly).is_ok());
        assert!(DirectoryGrant::new("C:\\models", GrantMode::ReadWrite).is_ok());
        assert!(DirectoryGrant::new("models", GrantMode::ReadOnly).is_err());
        assert!(DirectoryGrant::new("/models/../secrets", GrantMode::ReadOnly).is_err());
        assert!(DirectoryGrant::new("/models\0hidden", GrantMode::ReadOnly).is_err());
    }
}
