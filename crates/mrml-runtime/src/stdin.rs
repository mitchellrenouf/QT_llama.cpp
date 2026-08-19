use crate::{Text, Vector};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StdinError {
    ReadFailed,
    InvalidUtf8,
}

fn read_native(buffer: &mut [u8]) -> Option<usize> {
    #[cfg(windows)]
    {
        mrml_windows::read_stdin(buffer)
    }
    #[cfg(unix)]
    {
        mrml_linux::read_stdin(buffer)
    }
}

pub fn read_stdin_line() -> Result<Option<Text>, StdinError> {
    let mut bytes = Vector::new();
    let mut byte = [0u8; 1];
    loop {
        match read_native(&mut byte).ok_or(StdinError::ReadFailed)? {
            0 if bytes.is_empty() => return Ok(None),
            0 => break,
            _ if byte[0] == b'\n' => break,
            _ => bytes.push(byte[0]),
        }
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    Text::try_from_utf8(bytes)
        .map(Some)
        .map_err(|_| StdinError::InvalidUtf8)
}

pub fn read_stdin_to_end() -> Result<Text, StdinError> {
    let mut bytes = Vector::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = read_native(&mut chunk).ok_or(StdinError::ReadFailed)?;
        if read == 0 {
            break;
        }
        bytes
            .try_extend_from_slice(&chunk[..read])
            .map_err(|_| StdinError::ReadFailed)?;
    }
    Text::try_from_utf8(bytes).map_err(|_| StdinError::InvalidUtf8)
}
