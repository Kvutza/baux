pub use super::fbt_math::{GaussianShape, Mhc4Point, NeighborPoint, block_flags, half, half_bits};

pub const VOCAB: u32 = baux::semantics::mhc4::VOCABULARY as u32;
pub use baux::semantics::mhc4::{
    EncodeShape as Mhc4EncodeShape, ObjectiveStats, PairShape, ShardStats, WaveShape,
};

pub const FBT_THREADS: u32 = 256;
pub const TILE: u32 = 16;
pub const PISA_SCORES: usize = 512;
pub const PISA_WIDTH: usize = 64;
pub const PISA_CANDIDATES: usize = 16;
pub const PISA_SELECTED: usize = 8;
pub const PISA_GROUP_HEADS: usize = 4;
pub const PISA_HEAD_GROUPS: u32 = 8 / PISA_GROUP_HEADS as u32;
pub const PISA_HEAD_SCORES: usize = PISA_SCORES * PISA_GROUP_HEADS;
pub const PISA_QUERIES: usize = PISA_WIDTH * PISA_GROUP_HEADS;
pub const PISA_VALUES: usize = 256 * PISA_GROUP_HEADS;
pub const ROUTE_THREADS: u32 = 256;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FbtShape {
    pub rows: u32,
    pub width: u32,
    pub vocab: u32,
    pub epsilon_bits: u32,
}

// SAFETY: FbtShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for FbtShape {}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ScoreStats {
    pub reward: f32,
    pub mean_nll: f32,
    pub maximum_nll: f32,
    pub positional_accuracy: f32,
    pub tokens: u32,
    pub mismatches: u32,
    pub patch_nll_octiles: [f32; 8],
}

// SAFETY: ScoreStats is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for ScoreStats {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveShape {
    pub losses: u32,
    pub tokens: u32,
    pub patch: u32,
}

// SAFETY: ObjectiveShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for ObjectiveShape {}

impl FbtShape {
    pub fn new(rows: u32, width: u32, vocab: u32, epsilon: f32) -> Result<Self, &'static str> {
        if rows == 0 || width == 0 || vocab == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("FBT CUDA shape requires positive finite dimensions and epsilon");
        }
        Ok(Self {
            rows,
            width,
            vocab,
            epsilon_bits: epsilon.to_bits(),
        })
    }

    #[inline(always)]
    pub fn epsilon(self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulShape {
    pub rows: u32,
    pub columns: u32,
    pub inner: u32,
    pub input_stride: u32,
    pub weight_stride: u32,
    pub output_stride: u32,
    pub input_offset: u64,
    pub weight_offset: u64,
    pub output_offset: u64,
}

// SAFETY: MatmulShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MatmulShape {}

impl MatmulShape {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.columns == 0 || self.inner == 0 {
            return Err("FBT CUDA matmul dimensions must be positive");
        }
        if self.input_stride < self.inner
            || self.weight_stride < self.columns
            || self.output_stride < self.columns
        {
            return Err("FBT CUDA matmul strides do not cover their logical rows");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PisaShape {
    pub rows: u32,
    pub context: u32,
    pub heads: u32,
    pub head_width: u32,
    pub selected: u32,
    pub block_tokens: u32,
    pub query_start: u32,
    pub query_stride: u32,
    pub kv_stride: u32,
    pub key_offset: u32,
    pub visibility: u32,
}

// SAFETY: PisaShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for PisaShape {}

impl PisaShape {
    pub fn fbt(rows: u32, context: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            context,
            heads: 8,
            head_width: 64,
            selected: 8,
            block_tokens: 64,
            query_start: 0,
            query_stride: 640,
            kv_stride: 640,
            key_offset: 512,
            visibility: 0,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0
            || self.context == 0
            || self.heads == 0
            || self.head_width == 0
            || self.selected == 0
            || self.block_tokens == 0
        {
            return Err("FBT CUDA PISA shape has zero dimensions");
        }
        let leaves = self.context / self.block_tokens;
        let cached = self.query_stride == 512 && self.kv_stride == 128 && self.key_offset == 0;
        let packed = self.query_stride == 640 && self.kv_stride == 640 && self.key_offset == 512;
        if (!cached && !packed)
            || (packed && (self.rows % self.context != 0 || self.query_start != 0))
            || (cached
                && (self.rows > 4096
                    || self
                        .query_start
                        .checked_add(self.rows)
                        .is_none_or(|end| end > self.context)))
            || self.head_width as usize != PISA_WIDTH
            || self.heads != 8
            || self.block_tokens != 64
            || self.selected as usize != PISA_SELECTED
            || self.context % self.block_tokens != 0
            || !leaves.is_power_of_two()
            || !(64..=32768).contains(&leaves)
            || (self.visibility != 0
                && (!self.visibility.is_power_of_two() || !(128..=4096).contains(&self.visibility)))
            || self.selected as usize * self.block_tokens as usize > PISA_SCORES
        {
            return Err("FBT CUDA PISA shape is invalid");
        }
        Ok(())
    }

    pub fn cached(rows: u32, context: u32, start: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            context,
            heads: 8,
            head_width: 64,
            selected: 8,
            block_tokens: 64,
            query_start: start,
            query_stride: 512,
            kv_stride: 128,
            key_offset: 0,
            visibility: 0,
        };
        shape.validate()?;
        Ok(shape)
    }

    #[inline(always)]
    pub fn query_width(self) -> u32 {
        self.heads * self.head_width
    }

    #[inline(always)]
    pub fn qkv_width(self) -> u32 {
        self.query_width() + 2 * self.head_width
    }

    #[inline(always)]
    pub fn leaves(self) -> u32 {
        self.context / self.block_tokens
    }

    #[inline(always)]
    pub fn nodes(self) -> u32 {
        2 * self.leaves() - 1
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteShape {
    pub rows: u32,
    pub experts: u32,
    pub top_k: u32,
}

// SAFETY: RouteShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for RouteShape {}

impl RouteShape {
    pub fn fbt(rows: u32) -> Result<Self, &'static str> {
        if rows == 0 {
            return Err("FBT CUDA routing requires at least one row");
        }
        Ok(Self {
            rows,
            experts: 128,
            top_k: 3,
        })
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.rows == 0 || self.experts < 4 || self.experts > 625 || self.top_k != 3 {
            return Err("FBT CUDA routing shape is invalid");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeShape {
    pub width: u32,
    pub expert_width: u32,
    pub activation_stride: u32,
    pub row_tiles: u32,
}

// SAFETY: MoeShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for MoeShape {}

impl MoeShape {
    pub fn fbt() -> Self {
        Self {
            width: 512,
            expert_width: 216,
            activation_stride: 224,
            row_tiles: 8,
        }
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.width == 0
            || self.expert_width == 0
            || self.activation_stride < self.expert_width
            || self.row_tiles == 0
        {
            return Err("FBT CUDA MoE shape is invalid");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedTile {
    pub expert: u32,
    pub first_row: u32,
    pub valid_rows: u32,
    pub reserved: u32,
}

// SAFETY: RoutedTile is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for RoutedTile {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackShape {
    pub rows: u32,
    pub width: u32,
    pub context: u32,
    pub token_epsilon_bits: u32,
    pub fused_epsilon_bits: u32,
    pub token_unit_rms: u32,
    pub fused_unit_rms: u32,
}

// SAFETY: FeedbackShape is repr(C) and contains only DeviceCopy scalars.
unsafe impl cuda_core::DeviceCopy for FeedbackShape {}

impl FeedbackShape {
    pub fn fbt(rows: u32, context: u32) -> Result<Self, &'static str> {
        let shape = Self {
            rows,
            width: 512,
            context,
            token_epsilon_bits: 1.0e-5_f32.to_bits(),
            fused_epsilon_bits: 1.0e-5_f32.to_bits(),
            token_unit_rms: 1,
            fused_unit_rms: 1,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<(), &'static str> {
        let token = f32::from_bits(self.token_epsilon_bits);
        let fused = f32::from_bits(self.fused_epsilon_bits);
        if self.rows == 0
            || self.width == 0
            || self.context == 0
            || self.rows % self.context != 0
            || !token.is_finite()
            || token <= 0.0
            || !fused.is_finite()
            || fused <= 0.0
            || self.token_unit_rms > 1
            || self.fused_unit_rms > 1
        {
            return Err("FBT CUDA feedback shape is invalid");
        }
        Ok(())
    }

    #[inline(always)]
    pub fn token_epsilon(self) -> f32 {
        f32::from_bits(self.token_epsilon_bits)
    }

    #[inline(always)]
    pub fn fused_epsilon(self) -> f32 {
        f32::from_bits(self.fused_epsilon_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(PisaShape::fbt(8192, 4096).unwrap().qkv_width(), 640);
        assert!(PisaShape::fbt(8191, 4096).is_err());
        assert!(FbtShape::new(8192, 512, VOCAB, 1.0e-5).is_ok());
        assert!(MoeShape::fbt().validate().is_ok());
    }

    #[test]
    fn matmul_shape() {
        let shape = MatmulShape {
            rows: 8192,
            columns: 640,
            inner: 512,
            input_stride: 512,
            weight_stride: 640,
            output_stride: 640,
            input_offset: 0,
            weight_offset: 0,
            output_offset: 0,
        };
        assert!(shape.validate().is_ok());
    }
}
