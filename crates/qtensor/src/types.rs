use std::fmt;

/// Supported GGML Tensor Data Types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    IQ4_NL = 21,
    BF16 = 30,
}

impl DType {
    /// Number of elements in a single quantization block
    #[inline]
    pub fn block_size(&self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 | DType::F64 => 1,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 | DType::Q8_1 | DType::IQ4_NL => 32,
        }
    }

    /// Size in bytes of a single block of elements
    #[inline]
    pub fn type_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Q4_0 => 18,  // 2 bytes (fp16 scale) + 16 bytes (32 nibbles)
            DType::Q4_1 => 20,  // 2 bytes scale + 2 bytes min + 16 bytes
            DType::Q5_0 => 22,
            DType::Q5_1 => 24,
            DType::Q8_0 => 34,  // 2 bytes (fp16 scale) + 32 bytes (32 int8)
            DType::Q8_1 => 36,
            DType::IQ4_NL => 18,
        }
    }

    /// Check if this data type is quantized
    #[inline]
    pub fn is_quantized(&self) -> bool {
        self.block_size() > 1
    }

    /// Convert from GGUF integer ID to DType
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            3 => Some(DType::Q4_1),
            6 => Some(DType::Q5_0),
            7 => Some(DType::Q5_1),
            8 => Some(DType::Q8_0),
            9 => Some(DType::Q8_1),
            16 => Some(DType::I8),
            17 => Some(DType::I16),
            18 => Some(DType::I32),
            19 => Some(DType::I64),
            20 => Some(DType::F64),
            21 => Some(DType::IQ4_NL),
            30 => Some(DType::BF16),
            _ => None,
        }
    }

    /// Convert to GGUF integer ID
    pub fn to_u32(&self) -> u32 {
        *self as u32
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// 4D Tensor Shape compatible with GGML dimension layout (ne0, ne1, ne2, ne3)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub dims: [usize; 4],
    pub n_dims: usize,
}

impl Shape {
    pub fn new_1d(ne0: usize) -> Self {
        Self {
            dims: [ne0, 1, 1, 1],
            n_dims: 1,
        }
    }

    pub fn new_2d(ne0: usize, ne1: usize) -> Self {
        Self {
            dims: [ne0, ne1, 1, 1],
            n_dims: 2,
        }
    }

    pub fn new_3d(ne0: usize, ne1: usize, ne2: usize) -> Self {
        Self {
            dims: [ne0, ne1, ne2, 1],
            n_dims: 3,
        }
    }

    pub fn new_4d(ne0: usize, ne1: usize, ne2: usize, ne3: usize) -> Self {
        Self {
            dims: [ne0, ne1, ne2, ne3],
            n_dims: 4,
        }
    }

    pub fn from_slice(slice: &[usize]) -> Self {
        let mut dims = [1; 4];
        let n = slice.len().min(4);
        for i in 0..n {
            dims[i] = slice[i];
        }
        Self { dims, n_dims: n }
    }

    #[inline]
    pub fn ne0(&self) -> usize {
        self.dims[0]
    }

    #[inline]
    pub fn ne1(&self) -> usize {
        self.dims[1]
    }

    #[inline]
    pub fn ne2(&self) -> usize {
        self.dims[2]
    }

    #[inline]
    pub fn ne3(&self) -> usize {
        self.dims[3]
    }

    /// Total number of elements in the tensor
    #[inline]
    pub fn numel(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    /// Calculate default contiguous strides (in bytes) for a given DType
    pub fn default_strides(&self, dtype: DType) -> Strides {
        let bs = dtype.block_size();
        let ts = dtype.type_size();

        let nb0 = ts;
        let nb1 = (self.dims[0] / bs) * ts;
        let nb2 = nb1 * self.dims[1];
        let nb3 = nb2 * self.dims[2];

        Strides {
            strides: [nb0, nb1, nb2, nb3],
        }
    }
}

/// Tensor Strides in bytes across dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides {
    pub strides: [usize; 4],
}

impl Strides {
    #[inline]
    pub fn nb0(&self) -> usize {
        self.strides[0]
    }

    #[inline]
    pub fn nb1(&self) -> usize {
        self.strides[1]
    }

    #[inline]
    pub fn nb2(&self) -> usize {
        self.strides[2]
    }

    #[inline]
    pub fn nb3(&self) -> usize {
        self.strides[3]
    }

    /// Compute raw byte offset for given 4D index
    #[inline]
    pub fn offset(&self, i0: usize, i1: usize, i2: usize, i3: usize) -> usize {
        i0 * self.strides[0] + i1 * self.strides[1] + i2 * self.strides[2] + i3 * self.strides[3]
    }
}
