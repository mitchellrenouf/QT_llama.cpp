use crate::anyhow::{Result, anyhow};
use crate::types::{DType, Shape, Strides};
use mrml_runtime::Vector;

pub enum TensorStorage<'a> {
    OwnedF32(Vector<f32>),
    OwnedBytes(Vector<u8>),
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
            storage: TensorStorage::OwnedF32({
                let mut values = Vector::new();
                values.resize(numel, 0.0f32);
                values
            }),
        }
    }

    pub fn from_f32_vec(shape: Shape, data: Vector<f32>) -> Self {
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
            TensorStorage::OwnedF32(vec) => Ok(&vec[..]),
            _ => Err(anyhow!("Tensor storage is not owned F32")),
        }
    }

    pub fn as_f32_mut_slice(&mut self) -> Result<&mut [f32]> {
        match &mut self.storage {
            TensorStorage::OwnedF32(vec) => Ok(&mut vec[..]),
            _ => Err(anyhow!("Tensor storage is not mutable F32")),
        }
    }

    pub fn as_raw_bytes(&self) -> &[u8] {
        match &self.storage {
            TensorStorage::OwnedBytes(vec) => &vec[..],
            TensorStorage::Borrowed(slice) => slice,
            TensorStorage::OwnedF32(vec) => {
                let ptr = vec.as_ptr() as *const u8;
                let len = vec.len() * core::mem::size_of::<f32>();
                unsafe { core::slice::from_raw_parts(ptr, len) }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owns_and_reshapes_runtime_storage() {
        let mut tensor = Tensor::zeros(Shape::new_2d(3, 2));
        tensor.as_f32_mut_slice().unwrap()[5] = 4.5;
        tensor.reshape(Shape::new_1d(6)).unwrap();

        assert_eq!(
            tensor.as_f32_slice().unwrap(),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 4.5]
        );
        assert_eq!(tensor.as_raw_bytes().len(), 6 * core::mem::size_of::<f32>());
    }
}
