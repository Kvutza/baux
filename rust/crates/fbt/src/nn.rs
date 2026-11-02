//! Neural network layers built on [`crate::tensor`] views.
//!
//! A layer owns no buffers. It is given typed views of its parameters, its
//! state, and its inputs/outputs, and records kernel dispatches into a caller
//! encoder. The same layer serves prefill chunks and single-token decode; the
//! row geometry ([`Rows`]) selects kernel grids, never a separate code copy.

pub(crate) mod arena;
pub(crate) mod attention;
pub(crate) mod block;
pub(crate) mod hyper;
pub(crate) mod linear;
pub(crate) mod loss;
pub(crate) mod moe;
pub(crate) mod norm;

pub(crate) const WIDTH: u32 = baux::semantics::mhc4::WIDTH as u32;
pub(crate) const ROUTED_EXPERTS: u32 = baux::semantics::shape::ROUTED_COUNT as u32;
pub(crate) const EXPERT_WIDTH: u32 = baux::semantics::shape::EXPERT_WIDTH as u32;
pub(crate) const QKV_WIDTH: u32 = baux::semantics::shape::QKV_WIDTH as u32;
pub(crate) const ROPE_PAIRS: u32 = (WIDTH / 8) / 2;
pub(crate) const ROPE_HEADS: u32 = 9;
pub(crate) const PISA_BLOCK: u32 = 64;
pub(crate) const MODEL_LAYERS: u32 = baux::semantics::shape::MHC4_LAYERS as u32;

/// Row geometry of one layer invocation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rows {
    /// A tile of `n` consecutive positions (prefill and scoring).
    Chunk(u32),
    /// One position (incremental decode).
    Step,
}

impl Rows {
    #[inline(always)]
    pub(crate) fn count(self) -> u32 {
        if let Self::Chunk(rows) = self {
            rows
        } else {
            1
        }
    }

    /// Chunks fence exactly what they write; single steps fence every bound
    /// resource, which is how the decoder has always encoded.
    pub(crate) fn fence(self) -> crate::tensor::Fence {
        match self {
            Self::Chunk(_) => crate::tensor::Fence::Writes,
            Self::Step => crate::tensor::Fence::Bound,
        }
    }
}
