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
    Decompression,
    InvalidUtf8,
    Allocation,
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => error.fmt(output),
            Self::InvalidMagic(magic) => write!(output, "invalid ZIM magic 0x{magic:08x}"),
            Self::UnsupportedVersion(major, minor) => {
                write!(output, "unsupported ZIM version {major}.{minor}")
            }
            Self::InvalidOffset => output.write_str("invalid ZIM offset table"),
            Self::InvalidDirectoryEntry => output.write_str("invalid ZIM directory entry"),
            Self::InvalidCluster => output.write_str("invalid ZIM cluster"),
            Self::UnsupportedCompression(kind) => {
                write!(output, "unsupported ZIM cluster compression {kind}")
            }
            Self::Decompression => output.write_str("failed to decompress ZIM cluster"),
            Self::InvalidUtf8 => output.write_str("ZIM text is not valid UTF-8"),
            Self::Allocation => output.write_str("not enough memory for ZIM data"),
        }
    }
}

impl core::error::Error for Error {}
impl From<mrml_runtime::FileError> for Error {
    fn from(error: mrml_runtime::FileError) -> Self {
        Self::File(error)
    }
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
        if bytes.len() < HEADER_BYTES {
            return Err(Error::InvalidOffset);
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != ZIM_MAGIC {
            return Err(Error::InvalidMagic(magic));
        }
        let major_version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        let minor_version = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        if major_version != 6 {
            return Err(Error::UnsupportedVersion(major_version, minor_version));
        }
        Ok(Self {
            major_version,
            minor_version,
            uuid: bytes[8..24].try_into().unwrap(),
            entry_count: le_u32(bytes, 24),
            cluster_count: le_u32(bytes, 28),
            url_pointer_position: le_u64(bytes, 32),
            title_pointer_position: le_u64(bytes, 40),
            cluster_pointer_position: le_u64(bytes, 48),
            mime_list_position: le_u64(bytes, 56),
            main_page: le_u32(bytes, 64),
            layout_page: le_u32(bytes, 68),
            checksum_position: le_u64(bytes, 72),
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EntryLocation {
    Redirect(u32),
    Blob { cluster: u32, blob: u32 },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DirectoryEntry {
    pub mime_type: u16,
    pub namespace: u8,
    pub revision: u32,
    pub path: Text,
    pub title: Text,
    pub location: EntryLocation,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Compression {
    None,
    Zstd,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ClusterInfo {
    pub offset: u64,
    pub compressed_bytes: u64,
    pub compression: Compression,
    pub extended_offsets: bool,
}

/// Compression is deliberately supplied by a separate crate so the ZIM
/// container remains small and testable independently of codec internals.
pub trait ClusterDecoder {
    fn decode_zstd(&mut self, compressed: &[u8]) -> core::result::Result<Vector<u8>, ()>;
}

pub struct Archive {
    file: File,
    header: Header,
    cluster_offsets: Vector<u64>,
    mime_types: Vector<Text>,
}

impl Archive {
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = [0; HEADER_BYTES];
        file.read_exact(&mut bytes)?;
        let header = Header::parse(&bytes)?;
        let length = file.len()?;
        if header.checksum_position > length
            || header.mime_list_position >= header.checksum_position
            || header.cluster_pointer_position >= header.checksum_position
        {
            return Err(Error::InvalidOffset);
        }
        file.seek(header.mime_list_position)?;
        let mut mime_types = Vector::new();
        loop {
            let mime = read_zero_text(&mut file)?;
            if mime.is_empty() {
                break;
            }
            mime_types.push(mime);
        }
        file.seek(header.cluster_pointer_position)?;
        let mut cluster_offsets = Vector::with_capacity(header.cluster_count as usize + 1)
            .map_err(|_| Error::InvalidOffset)?;
        for _ in 0..header.cluster_count {
            cluster_offsets.push(read_u64(&mut file)?);
        }
        cluster_offsets.push(header.checksum_position);
        if cluster_offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidOffset);
        }
        Ok(Self {
            file,
            header,
            cluster_offsets,
            mime_types,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }
    pub fn mime_types(&self) -> &[Text] {
        &self.mime_types
    }
    pub fn mime_type(&self, index: u16) -> Option<&str> {
        self.mime_types.get(index as usize).map(Text::as_str)
    }

    pub fn cluster_info(&mut self, cluster: u32) -> Result<ClusterInfo> {
        let offset = *self
            .cluster_offsets
            .get(cluster as usize)
            .ok_or(Error::InvalidOffset)?;
        let end = *self
            .cluster_offsets
            .get(cluster as usize + 1)
            .ok_or(Error::InvalidOffset)?;
        self.file.seek(offset)?;
        let descriptor = read_u8(&mut self.file)?;
        let compression = match descriptor & 0x0f {
            0 | 1 => Compression::None,
            5 => Compression::Zstd,
            value => Compression::Unknown(value),
        };
        Ok(ClusterInfo {
            offset,
            compressed_bytes: end.checked_sub(offset).ok_or(Error::InvalidCluster)?,
            compression,
            extended_offsets: descriptor & 0x10 != 0,
        })
    }

    /// Copies the cluster payload exactly as stored, excluding its descriptor.
    /// This allows the native Zstandard decoder to consume clusters without
    /// loading the complete archive or relying on temporary files.
    pub fn read_cluster_payload(&mut self, cluster: u32) -> Result<Vector<u8>> {
        let info = self.cluster_info(cluster)?;
        let length = usize::try_from(
            info.compressed_bytes
                .checked_sub(1)
                .ok_or(Error::InvalidCluster)?,
        )
        .map_err(|_| Error::InvalidCluster)?;
        let mut output = Vector::with_capacity(length).map_err(|_| Error::Allocation)?;
        output.resize(length, 0);
        self.file.read_exact(&mut output)?;
        Ok(output)
    }

    pub fn entry(&mut self, index: u32) -> Result<DirectoryEntry> {
        if index >= self.header.entry_count {
            return Err(Error::InvalidOffset);
        }
        self.file
            .seek(self.header.url_pointer_position + index as u64 * 8)?;
        let offset = read_u64(&mut self.file)?;
        self.file.seek(offset)?;
        let mime_type = read_u16(&mut self.file)?;
        let parameter_len = read_u8(&mut self.file)? as usize;
        let namespace = read_u8(&mut self.file)?;
        let revision = read_u32(&mut self.file)?;
        let location = if mime_type == u16::MAX {
            EntryLocation::Redirect(read_u32(&mut self.file)?)
        } else {
            EntryLocation::Blob {
                cluster: read_u32(&mut self.file)?,
                blob: read_u32(&mut self.file)?,
            }
        };
        let path = read_zero_text(&mut self.file)?;
        let title = read_zero_text(&mut self.file)?;
        let position = self.file.position();
        self.file.seek(
            position
                .checked_add(parameter_len as u64)
                .ok_or(Error::InvalidOffset)?,
        )?;
        Ok(DirectoryEntry {
            mime_type,
            namespace,
            revision,
            path,
            title,
            location,
        })
    }

    /// Reads a blob from an uncompressed cluster. Compressed clusters are
    /// reported explicitly so a decompressor can be layered without changing
    /// directory parsing or dataset streaming APIs.
    pub fn read_blob(&mut self, cluster: u32, blob: u32) -> Result<Vector<u8>> {
        let start = *self
            .cluster_offsets
            .get(cluster as usize)
            .ok_or(Error::InvalidOffset)?;
        let end = *self
            .cluster_offsets
            .get(cluster as usize + 1)
            .ok_or(Error::InvalidOffset)?;
        self.file.seek(start)?;
        let descriptor = read_u8(&mut self.file)?;
        let compression = descriptor & 0x0f;
        if compression != 0 && compression != 1 {
            return Err(Error::UnsupportedCompression(compression));
        }
        let wide = descriptor & 0x10 != 0;
        let length = usize::try_from(end - start - 1).map_err(|_| Error::InvalidCluster)?;
        let mut bytes = Vector::with_capacity(length).map_err(|_| Error::Allocation)?;
        bytes.resize(length, 0);
        self.file.read_exact(&mut bytes)?;
        blob_from_cluster(&bytes, wide, blob)
    }

    pub fn read_blob_with<D: ClusterDecoder>(
        &mut self,
        cluster: u32,
        blob: u32,
        decoder: &mut D,
    ) -> Result<Vector<u8>> {
        let info = self.cluster_info(cluster)?;
        match info.compression {
            Compression::None => self.read_blob(cluster, blob),
            Compression::Zstd => {
                let compressed = self.read_cluster_payload(cluster)?;
                let decompressed = decoder
                    .decode_zstd(&compressed)
                    .map_err(|_| Error::Decompression)?;
                blob_from_cluster(&decompressed, info.extended_offsets, blob)
            }
            Compression::Unknown(kind) => Err(Error::UnsupportedCompression(kind)),
        }
    }
}

fn blob_from_cluster(bytes: &[u8], wide: bool, blob: u32) -> Result<Vector<u8>> {
    let width: usize = if wide { 8 } else { 4 };
    if bytes.len() < width {
        return Err(Error::InvalidCluster);
    }
    let first = if wide {
        u64::from_le_bytes(bytes[..8].try_into().unwrap())
    } else {
        u32::from_le_bytes(bytes[..4].try_into().unwrap()) as u64
    };
    let width: u64 = if wide { 8 } else { 4 };
    if first < width || first % width != 0 {
        return Err(Error::InvalidCluster);
    }
    let count = (first / width) as usize - 1;
    if blob as usize >= count {
        return Err(Error::InvalidCluster);
    }
    let offset = blob as usize * width as usize;
    let blob_start = read_offset(bytes, offset, wide)?;
    let blob_end = read_offset(bytes, offset + width as usize, wide)?;
    if blob_start > blob_end || blob_end > bytes.len() as u64 {
        return Err(Error::InvalidCluster);
    }
    let length = usize::try_from(blob_end - blob_start).map_err(|_| Error::InvalidCluster)?;
    let mut output = Vector::with_capacity(length).map_err(|_| Error::InvalidCluster)?;
    output
        .try_extend_from_slice(&bytes[blob_start as usize..blob_end as usize])
        .map_err(|_| Error::Allocation)?;
    Ok(output)
}

fn read_offset(bytes: &[u8], offset: usize, wide: bool) -> Result<u64> {
    if wide {
        bytes
            .get(offset..offset + 8)
            .map(|value| u64::from_le_bytes(value.try_into().unwrap()))
            .ok_or(Error::InvalidCluster)
    } else {
        bytes
            .get(offset..offset + 4)
            .map(|value| u32::from_le_bytes(value.try_into().unwrap()) as u64)
            .ok_or(Error::InvalidCluster)
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn le_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
fn read_u8(file: &mut File) -> Result<u8> {
    let mut b = [0];
    file.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u16(file: &mut File) -> Result<u16> {
    let mut b = [0; 2];
    file.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(file: &mut File) -> Result<u32> {
    let mut b = [0; 4];
    file.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(file: &mut File) -> Result<u64> {
    let mut b = [0; 8];
    file.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_zero_text(file: &mut File) -> Result<Text> {
    let mut bytes = Vector::new();
    loop {
        let byte = read_u8(file)?;
        if byte == 0 {
            break;
        }
        bytes.push(byte);
    }
    core::str::from_utf8(&bytes)
        .map(Text::from)
        .map_err(|_| Error::InvalidUtf8)
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
        assert!(matches!(
            Header::parse(&[0; 80]),
            Err(Error::InvalidMagic(0))
        ));
        let mut bytes = [0u8; 80];
        bytes[0..4].copy_from_slice(&ZIM_MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&5u16.to_le_bytes());
        assert!(matches!(
            Header::parse(&bytes),
            Err(Error::UnsupportedVersion(5, 0))
        ));
    }

    #[test]
    fn opens_external_archive_when_configured() {
        let Some(path) = mrml_runtime::environment_variable("MRML_TEST_ZIM") else {
            return;
        };
        let mut archive = Archive::open(&path).expect("open configured ZIM archive");
        assert!(archive.header().entry_count > 0);
        assert!(archive.header().cluster_count > 0);
        assert!(archive.mime_types().iter().any(|mime| mime == "text/html"));
        assert!(matches!(
            archive.cluster_info(0).unwrap().compression,
            Compression::Zstd
        ));
        let payload = archive.read_cluster_payload(0).unwrap();
        assert_eq!(payload.get(..4), Some(&[0x28, 0xb5, 0x2f, 0xfd][..]));
        let entry = archive.entry(0).expect("read first directory entry");
        assert!(!entry.path.is_empty());
    }

    #[test]
    fn extracts_blob_from_decoded_cluster() {
        let mut cluster = Vector::new();
        cluster
            .try_extend_from_slice(&[12, 0, 0, 0, 15, 0, 0, 0, 19, 0, 0, 0])
            .unwrap();
        cluster.try_extend_from_slice(b"onefour").unwrap();
        let first = blob_from_cluster(&cluster, false, 0).unwrap();
        let second = blob_from_cluster(&cluster, false, 1).unwrap();
        assert_eq!(&*first, b"one");
        assert_eq!(&*second, b"four");
        assert!(matches!(
            blob_from_cluster(&cluster, false, 2),
            Err(Error::InvalidCluster)
        ));
    }
}
