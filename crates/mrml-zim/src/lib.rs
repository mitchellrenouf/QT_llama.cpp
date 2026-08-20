#![no_std]

use core::fmt;
use mrml_runtime::{File, Text, Vector};

pub const ZIM_MAGIC: u32 = 0x044d_495a;
const HEADER_BYTES: usize = 80;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Error {
    File(mrml_runtime::FileError),
    InvalidMagic(u32),
    UnsupportedVersion(u16, u16),
    InvalidOffset,
    InvalidDirectoryEntry,
    InvalidCluster,
    UnsupportedCompression(u8),
    InvalidUtf8,
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => error.fmt(output),
            Self::InvalidMagic(magic) => write!(output, "invalid ZIM magic 0x{magic:08x}"),
            Self::UnsupportedVersion(major, minor) => write!(output, "unsupported ZIM version {major}.{minor}"),
            Self::InvalidOffset => output.write_str("invalid ZIM offset table"),
            Self::InvalidDirectoryEntry => output.write_str("invalid ZIM directory entry"),
            Self::InvalidCluster => output.write_str("invalid ZIM cluster"),
            Self::UnsupportedCompression(kind) => write!(output, "unsupported ZIM cluster compression {kind}"),
            Self::InvalidUtf8 => output.write_str("ZIM text is not valid UTF-8"),
        }
    }
}

impl core::error::Error for Error {}
impl From<mrml_runtime::FileError> for Error {
    fn from(error: mrml_runtime::FileError) -> Self { Self::File(error) }
}
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Header {
    pub major_version: u16,
    pub minor_version: u16,
    pub uuid: [u8; 16],
    pub entry_count: u32,
    pub cluster_count: u32,
    pub url_pointer_position: u64,
    pub title_pointer_position: u64,
    pub cluster_pointer_position: u64,
    pub mime_list_position: u64,
    pub main_page: u32,
    pub layout_page: u32,
    pub checksum_position: u64,
}

impl Header {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_BYTES { return Err(Error::InvalidOffset); }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != ZIM_MAGIC { return Err(Error::InvalidMagic(magic)); }
        let major_version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let minor_version = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        if major_version != 6 { return Err(Error::UnsupportedVersion(major_version, minor_version)); }
        Ok(Self {
            major_version, minor_version, uuid: bytes[8..24].try_into().unwrap(),
            entry_count: le_u32(bytes, 24), cluster_count: le_u32(bytes, 28),
            url_pointer_position: le_u64(bytes, 32), title_pointer_position: le_u64(bytes, 40),
            cluster_pointer_position: le_u64(bytes, 48), mime_list_position: le_u64(bytes, 56),
            main_page: le_u32(bytes, 64), layout_page: le_u32(bytes, 68),
            checksum_position: le_u64(bytes, 72),
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EntryLocation { Redirect(u32), Blob { cluster: u32, blob: u32 } }

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DirectoryEntry {
    pub mime_type: u16,
    pub namespace: u8,
    pub revision: u32,
    pub path: Text,
    pub title: Text,
    pub location: EntryLocation,
}

pub struct Archive { file: File, header: Header, cluster_offsets: Vector<u64> }

impl Archive {
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = [0; HEADER_BYTES];
        file.read_exact(&mut bytes)?;
        let header = Header::parse(&bytes)?;
        file.seek(header.cluster_pointer_position)?;
        let mut cluster_offsets = Vector::with_capacity(header.cluster_count as usize + 1).map_err(|_| Error::InvalidOffset)?;
        for _ in 0..header.cluster_count { cluster_offsets.push(read_u64(&mut file)?); }
        cluster_offsets.push(header.checksum_position);
        if cluster_offsets.windows(2).any(|pair| pair[0] >= pair[1]) { return Err(Error::InvalidOffset); }
        Ok(Self { file, header, cluster_offsets })
    }

    pub fn header(&self) -> &Header { &self.header }

    pub fn entry(&mut self, index: u32) -> Result<DirectoryEntry> {
        if index >= self.header.entry_count { return Err(Error::InvalidOffset); }
        self.file.seek(self.header.url_pointer_position + index as u64 * 8)?;
        let offset = read_u64(&mut self.file)?;
        self.file.seek(offset)?;
        let mime_type = read_u16(&mut self.file)?;
        let parameter_len = read_u8(&mut self.file)? as usize;
        let namespace = read_u8(&mut self.file)?;
        let revision = read_u32(&mut self.file)?;
        let location = if mime_type == u16::MAX {
            EntryLocation::Redirect(read_u32(&mut self.file)?)
        } else {
            EntryLocation::Blob { cluster: read_u32(&mut self.file)?, blob: read_u32(&mut self.file)? }
        };
        let path = read_zero_text(&mut self.file)?;
        let title = read_zero_text(&mut self.file)?;
        let position = self.file.position();
        self.file.seek(position.checked_add(parameter_len as u64).ok_or(Error::InvalidOffset)?)?;
        Ok(DirectoryEntry { mime_type, namespace, revision, path, title, location })
    }

    /// Reads a blob from an uncompressed cluster. Compressed clusters are
    /// reported explicitly so a decompressor can be layered without changing
    /// directory parsing or dataset streaming APIs.
    pub fn read_blob(&mut self, cluster: u32, blob: u32) -> Result<Vector<u8>> {
        let start = *self.cluster_offsets.get(cluster as usize).ok_or(Error::InvalidOffset)?;
        let end = *self.cluster_offsets.get(cluster as usize + 1).ok_or(Error::InvalidOffset)?;
        self.file.seek(start)?;
        let descriptor = read_u8(&mut self.file)?;
        let compression = descriptor & 0x0f;
        if compression != 0 { return Err(Error::UnsupportedCompression(compression)); }
        let wide = descriptor & 0x10 != 0;
        let first = if wide { read_u64(&mut self.file)? } else { read_u32(&mut self.file)? as u64 };
        let width: u64 = if wide { 8 } else { 4 };
        if first < width || first % width != 0 { return Err(Error::InvalidCluster); }
        let count = (first / width) as usize - 1;
        if blob as usize >= count { return Err(Error::InvalidCluster); }
        self.file.seek(start + 1 + blob as u64 * width as u64)?;
        let blob_start = if wide { read_u64(&mut self.file)? } else { read_u32(&mut self.file)? as u64 };
        let blob_end = if wide { read_u64(&mut self.file)? } else { read_u32(&mut self.file)? as u64 };
        if blob_start > blob_end || start + 1 + blob_end > end { return Err(Error::InvalidCluster); }
        self.file.seek(start + 1 + blob_start)?;
        let length = usize::try_from(blob_end - blob_start).map_err(|_| Error::InvalidCluster)?;
        let mut output = Vector::with_capacity(length).map_err(|_| Error::InvalidCluster)?;
        output.resize(length, 0);
        self.file.read_exact(&mut output)?;
        Ok(output)
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 { u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) }
fn le_u64(bytes: &[u8], offset: usize) -> u64 { u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) }
fn read_u8(file: &mut File) -> Result<u8> { let mut b=[0]; file.read_exact(&mut b)?; Ok(b[0]) }
fn read_u16(file: &mut File) -> Result<u16> { let mut b=[0;2]; file.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn read_u32(file: &mut File) -> Result<u32> { let mut b=[0;4]; file.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn read_u64(file: &mut File) -> Result<u64> { let mut b=[0;8]; file.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn read_zero_text(file: &mut File) -> Result<Text> {
    let mut bytes = Vector::new();
    loop { let byte=read_u8(file)?; if byte == 0 { break; } bytes.push(byte); }
    core::str::from_utf8(&bytes).map(Text::from).map_err(|_| Error::InvalidUtf8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_version_six_header() {
        let mut bytes = [0u8; 80];
        bytes[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&6u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&1u16.to_le_bytes());
        bytes[24..28].copy_from_slice(&7u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&3u32.to_le_bytes());
        bytes[72..80].copy_from_slice(&4096u64.to_le_bytes());
        let header = Header::parse(&bytes).unwrap();
        assert_eq!(header.entry_count, 7);
        assert_eq!(header.cluster_count, 3);
        assert_eq!(header.checksum_position, 4096);
    }

    #[test]
    fn rejects_other_files_and_versions() {
        assert!(matches!(Header::parse(&[0; 80]), Err(Error::InvalidMagic(0))));
        let mut bytes = [0u8; 80];
        bytes[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&5u16.to_le_bytes());
        assert!(matches!(Header::parse(&bytes), Err(Error::UnsupportedVersion(5, 0))));
    }
}
