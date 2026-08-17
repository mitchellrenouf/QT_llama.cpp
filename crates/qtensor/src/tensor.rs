use crate::types::{DType, Shape, Strides};
use anyhow::{anyhow, Result};

pub enum TensorStorage<'a> {
    OwnedF32(Vec<f32>),
    OwnedBytes(Vec<u8>),
    Borrowed(&'a [u8]),
}

pub struct Tensor<'a> {
    pub shape: Shape,
    pub strides: Strides,
    pub dtype: DType,
    pub storage: TensorStorage<'a>,
}

impl<'a> Tensor<'a> {
    pub fn zeros(shape: Shape) -> Self {
        let numel = shape.numel();
        let strides = shape.default_strides(DType::F32);
        Self {
            shape,
            strides,
            dtype: DType::F32,
            storage: TensorStorage::OwnedF32(vec![0.0f32; numel]),
        }
    }

    pub fn from_f32_vec(shape: Shape, data: Vec<f32>) -> Self {
        assert_eq!(shape.numel(), data.len());
        let strides = shape.default_strides(DType::F32);
        Self {
            shape,
            strides,
            dtype: DType::F32,
            storage: TensorStorage::OwnedF32(data),
        }
    }

    pub fn from_borrowed_bytes(shape: Shape, dtype: DType, bytes: &'a [u8]) -> Self {
        let strides = shape.default_strides(dtype);
        Self {
            shape,
            strides,
            dtype,
            storage: TensorStorage::Borrowed(bytes),
        }
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match &self.storage {
            TensorStorage::OwnedF32(vec) => Ok(vec.as_slice()),
            _ => Err(anyhow!("Tensor storage is not owned F32")),
        }
    }

    pub fn as_f32_mut_slice(&mut self) -> Result<&mut [f32]> {
        match &mut self.storage {
            TensorStorage::OwnedF32(vec) => Ok(vec.as_mut_slice()),
            _ => Err(anyhow!("Tensor storage is not mutable F32")),
        }
    }

    pub fn as_raw_bytes(&self) -> &[u8] {
        match &self.storage {
            TensorStorage::OwnedBytes(vec) => vec.as_slice(),
            TensorStorage::Borrowed(slice) => slice,
            TensorStorage::OwnedF32(vec) => {
                let ptr = vec.as_ptr() as *const u8;
                let len = vec.len() * std::mem::size_of::<f32>();
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
        }
    }

    pub fn reshape(&mut self, new_shape: Shape) -> Result<()> {
        if self.shape.numel() != new_shape.numel() {
            return Err(anyhow!(
                "Cannot reshape tensor with {} elements to {} elements",
                self.shape.numel(),
                new_shape.numel()
            ));
        }
        self.strides = new_shape.default_strides(self.dtype);
        self.shape = new_shape;
        Ok(())
    }
}
