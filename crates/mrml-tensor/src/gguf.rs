use crate::types::{DType, Shape};
use crate::anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little endian

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Uint64(v) => Some(*v as u32),
            GgufValue::Int64(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint8(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            GgufValue::Uint32(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub offset: u64,
    pub size_bytes: usize,
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub data_offset: u64,
    pub path: PathBuf,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p_buf = path.as_ref().to_path_buf();
        let mut file = File::open(&p_buf)?;

        let magic = read_u32_le(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic number: 0x{:08X}", magic));
        }

        let version = read_u32_le(&mut file)?;
        if version < 2 || version > 3 {
            return Err(anyhow!("Unsupported GGUF version: {}", version));
        }

        let tensor_count = read_u64_le(&mut file)?;
        let kv_count = read_u64_le(&mut file)?;

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut file)?;
            let val_type = read_u32_le(&mut file)?;
            let value = read_gguf_value(&mut file, val_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file)?;
            let n_dims = read_u32_le(&mut file)? as usize;

            let mut dims = [1; 4];
            for d in 0..n_dims.min(4) {
                dims[d] = read_u64_le(&mut file)? as usize;
            }
            // Skip any extra dimensions
            for _ in 4..n_dims {
                let _ = read_u64_le(&mut file)?;
            }

            let type_id = read_u32_le(&mut file)?;
            let dtype = match type_id {
                0 => DType::F32,
                1 => DType::F16,
                2 => DType::Q4_0,
                3 => DType::Q4_1,
                8 => DType::Q8_0,
                12 => DType::Q4_K,
                13 => DType::Q5_K,
                14 => DType::Q6_K,
                16 => DType::IQ4_XS,
                30 => DType::BF16,
                _ => return Err(anyhow!("Unsupported GGUF dtype ID: {}", type_id)),
            };

            let offset = read_u64_le(&mut file)?;
            let shape = Shape {
                dims,
                n_dims: n_dims.min(4),
            };

            let numel = shape.numel();
            let size_bytes = (numel / dtype.block_size()) * dtype.type_size();

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    shape,
                    dtype,
                    offset,
                    size_bytes,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;

        let cur_pos = file.stream_position()?;
        let data_offset = if cur_pos % alignment == 0 {
            cur_pos
        } else {
            cur_pos + (alignment - (cur_pos % alignment))
        };

        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            path: p_buf,
        })
    }

    /// Read raw tensor bytes directly from file on-demand
    pub fn read_tensor_bytes(&self, tensor: &GgufTensorInfo) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)?;
        let start = self.data_offset + tensor.offset;
        file.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; tensor.size_bytes];
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    pub fn get_meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
}

fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64_le(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_gguf_value<R: Read>(reader: &mut R, val_type: u32) -> Result<GgufValue> {
    match val_type {
        0 => Ok(GgufValue::Uint8(read_u8(reader)?)),
        1 => Ok(GgufValue::Int8(read_i8(reader)?)),
        2 => Ok(GgufValue::Uint16(read_u16_le(reader)?)),
        3 => Ok(GgufValue::Int16(read_i16_le(reader)?)),
        4 => Ok(GgufValue::Uint32(read_u32_le(reader)?)),
        5 => Ok(GgufValue::Int32(read_i32_le(reader)?)),
        6 => Ok(GgufValue::Float32(read_f32_le(reader)?)),
        7 => Ok(GgufValue::Bool(read_u8(reader)? != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(reader)?)),
        9 => {
            let item_type = read_u32_le(reader)?;
            let len = read_u64_le(reader)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_gguf_value(reader, item_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64_le(reader)?)),
        11 => Ok(GgufValue::Int64(read_i64_le(reader)?)),
        12 => Ok(GgufValue::Float64(read_f64_le(reader)?)),
        _ => Err(anyhow!("Unknown GGUF value type ID: {}", val_type)),
    }
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    reader.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    let mut b = [0u8; 1];
    reader.read_exact(&mut b)?;
    Ok(b[0] as i8)
}

fn read_u16_le<R: Read>(reader: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    reader.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16_le<R: Read>(reader: &mut R) -> Result<i16> {
    let mut b = [0u8; 2];
    reader.read_exact(&mut b)?;
    Ok(i16::from_le_bytes(b))
}

fn read_u32_le<R: Read>(reader: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    reader.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32_le<R: Read>(reader: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    reader.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32_le<R: Read>(reader: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    reader.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(reader: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64_le<R: Read>(reader: &mut R) -> Result<i64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_f64_le<R: Read>(reader: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
