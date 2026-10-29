//! Readout projection and cross-entropy loss reduction sublayer.
//!
//! Encodes:
//! - Vocabulary logit projection from hidden state (`Rows::Chunk` or
//!   `Rows::Step`)
//! - Chunked cross-entropy loss computation against target token IDs
//! - Scalar loss reduction across all tokens in the forward chunk
//! - Direct connection to [`EvalReceipt`] in optimizer transaction lifecycle

use metal::{ComputeCommandEncoderRef, ComputePipelineState};

use baux::semantics::optimizer::{EvalReceipt, WorldId};

use crate::tensor::{Dispatch, F16, F32, Fence, Out, U32, View};

const VOCAB: u32 = baux::semantics::mhc4::VOCABULARY as u32;
const WIDTH: u32 = baux::semantics::mhc4::WIDTH as u32;

/// Readout projection weights: vocabulary embedding matrix `[WIDTH, VOCAB]`.
#[derive(Clone, Copy)]
pub(crate) struct ReadoutWeights<'a> {
    pub(crate) embedding: View<'a, F16>,
}

impl<'a> ReadoutWeights<'a> {
    #[inline(always)]
    pub(crate) const fn new(embedding: View<'a, F16>) -> Self {
        Self { embedding }
    }

    /// Readout weights directly from parameter buffer at byte offset.
    #[inline(always)]
    pub(crate) fn from_buffer(buffer: &'a metal::BufferRef, offset: u64) -> Self {
        Self::new(View::new(buffer, offset, WIDTH, VOCAB))
    }

    /// Underlying embedding view.
    #[inline(always)]
    pub(crate) const fn embedding(self) -> View<'a, F16> {
        self.embedding
    }
}

/// Chunked readout and cross-entropy loss reduction sublayer.
#[derive(Clone, Copy)]
pub(crate) struct ChunkLoss<'a> {
    pub(crate) loss_tiles: &'a ComputePipelineState,
    pub(crate) loss_reduce: &'a ComputePipelineState,
    pub(crate) weights: ReadoutWeights<'a>,
    pub(crate) rows: u32,
}

impl<'a> ChunkLoss<'a> {
    pub(crate) fn new(
        loss_tiles: &'a ComputePipelineState,
        loss_reduce: &'a ComputePipelineState,
        weights: ReadoutWeights<'a>,
        rows: u32,
    ) -> Self {
        Self {
            loss_tiles,
            loss_reduce,
            weights,
            rows,
        }
    }

    /// Copy of this sublayer configured for `rows` active tokens.
    #[inline(always)]
    pub(crate) fn with_rows(self, rows: u32) -> Self {
        Self { rows, ..self }
    }

    /// Number of rows scored in one chunk.
    #[inline(always)]
    pub(crate) const fn rows(&self) -> u32 {
        self.rows
    }

    /// Encode vocabulary projection and target loss tile scoring into
    /// `loss_partials`.
    ///
    /// Grid configuration:
    /// - Width: `VOCAB / 64` threadgroups (one per 64-column vocabulary tile).
    /// - Height: `rows.div_ceil(128)` threadgroups (128 rows per threadgroup).
    /// - Threadgroup size: 128 threads.
    pub(crate) fn score_tiles(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: View<'_, F16>,
        targets: View<'_, U32>,
        loss_partials: Out<'_, F32>,
    ) {
        let rows = self.rows;
        let shape = [rows, VOCAB];
        let tile_cols = VOCAB / 64;
        Dispatch::new(encoder, self.loss_tiles, Fence::Writes)
            .read(hidden)
            .read(self.weights.embedding)
            .read(targets)
            .write(loss_partials)
            .bytes(&shape)
            .groups_2d([tile_cols, rows.div_ceil(128)], 128);
    }

    /// Reduce partial cross-entropy tiles into per-token scalar losses.
    ///
    /// Grid configuration:
    /// - `rows` threadgroups of 128 threads.
    /// - Constant tiles parameter `VOCAB / 64` at buffer slot 2.
    pub(crate) fn reduce_loss(
        &self,
        encoder: &ComputeCommandEncoderRef,
        loss_partials: View<'_, F32>,
        output_losses: Out<'_, F32>,
    ) {
        let rows = self.rows;
        let tiles = VOCAB / 64;
        Dispatch::new(encoder, self.loss_reduce, Fence::Writes)
            .read(loss_partials)
            .write(output_losses)
            .bytes(&tiles)
            .groups(rows, 128);
    }

    /// Complete chunk scoring pass: project vocabulary, compute partial tiles,
    /// and reduce to per-token cross-entropy losses.
    pub(crate) fn score_chunk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: View<'_, F16>,
        targets: View<'_, U32>,
        loss_partials: Out<'_, F32>,
        output_losses: Out<'_, F32>,
    ) {
        let partials = loss_partials.view();
        self.score_tiles(encoder, hidden, targets, loss_partials);
        self.reduce_loss(encoder, partials, output_losses);
    }

    /// Compute an [`EvalReceipt`] directly from token loss slices across one or
    /// more chunks.
    pub fn eval_receipt(
        world: WorldId,
        candidate: &[f32],
        baseline: Option<&[f32]>,
    ) -> EvalReceipt {
        match baseline {
            Some(base) => LossAccumulator::from_paired(candidate, base).eval_receipt(world),
            None => LossAccumulator::from_chunk(candidate).eval_receipt(world),
        }
    }
}

/// Online loss accumulator tracking cross-entropy statistics across chunks.
///
/// Computes sample mean, variance of the sample mean, and optional paired
/// improvement across streaming forward evaluation chunks, connecting directly
/// to [`EvalReceipt`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LossAccumulator {
    tokens: usize,
    loss_sum: f64,
    loss_sq: f64,
    pair_sum: f64,
    pair_sq: f64,
    pair_tokens: usize,
}

impl LossAccumulator {
    /// Create an empty loss accumulator.
    pub const fn new() -> Self {
        Self {
            tokens: 0,
            loss_sum: 0.0,
            loss_sq: 0.0,
            pair_sum: 0.0,
            pair_sq: 0.0,
            pair_tokens: 0,
        }
    }

    /// Accumulate one token loss.
    #[inline]
    pub fn push_token(&mut self, loss: f32) {
        let val = f64::from(loss);
        self.loss_sum += val;
        self.loss_sq += val * val;
        self.tokens += 1;
    }

    /// Accumulate a chunk of token losses.
    pub fn push_chunk(&mut self, losses: &[f32]) {
        for &loss in losses {
            self.push_token(loss);
        }
    }

    /// Accumulate one paired token loss evaluation against an incumbent.
    #[inline]
    pub fn push_pair(&mut self, candidate: f32, baseline: f32) {
        self.push_token(candidate);
        let gain = f64::from(baseline - candidate);
        self.pair_sum += gain;
        self.pair_sq += gain * gain;
        self.pair_tokens += 1;
    }

    /// Accumulate a chunk of paired token evaluations.
    pub fn pair_chunk(&mut self, candidate: &[f32], baseline: &[f32]) {
        let count = candidate.len().min(baseline.len());
        for i in 0..count {
            self.push_pair(candidate[i], baseline[i]);
        }
    }

    /// Number of evaluated tokens.
    #[inline(always)]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Measured mean cross-entropy loss.
    pub fn mean(&self) -> f64 {
        if self.tokens == 0 {
            return 0.0;
        }
        self.loss_sum / self.tokens as f64
    }

    /// Estimated variance of the candidate sample mean loss.
    pub fn variance(&self) -> f64 {
        if self.tokens <= 1 {
            return 0.0;
        }
        let n = self.tokens as f64;
        let mean = self.loss_sum / n;
        ((self.loss_sq - self.loss_sum * mean).max(0.0)) / (n * (n - 1.0))
    }

    /// Measured paired improvement gain over incumbent, if paired.
    pub fn paired_gain(&self) -> Option<f64> {
        if self.pair_tokens == 0 {
            return None;
        }
        Some(self.pair_sum / self.pair_tokens as f64)
    }

    /// Estimated variance of the paired improvement mean, if paired.
    pub fn paired_variance(&self) -> Option<f64> {
        if self.pair_tokens == 0 {
            return None;
        }
        if self.pair_tokens == 1 {
            return Some(0.0);
        }
        let n = self.pair_tokens as f64;
        let mean = self.pair_sum / n;
        let var = ((self.pair_sq - self.pair_sum * mean).max(0.0)) / (n * (n - 1.0));
        Some(var)
    }

    /// Build an [`EvalReceipt`] connecting this evaluation to the optimizer
    /// transaction lifecycle.
    pub fn eval_receipt(&self, world: WorldId) -> EvalReceipt {
        let variance = self.paired_variance().unwrap_or_else(|| self.variance());
        EvalReceipt::new(
            world,
            self.tokens,
            self.mean(),
            variance,
            self.paired_gain(),
        )
    }

    /// Construct directly from a slice of token losses.
    pub fn from_chunk(losses: &[f32]) -> Self {
        let mut acc = Self::new();
        acc.push_chunk(losses);
        acc
    }

    /// Construct directly from paired candidate and baseline loss slices.
    pub fn from_paired(candidate: &[f32], baseline: &[f32]) -> Self {
        let mut acc = Self::new();
        acc.pair_chunk(candidate, baseline);
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let acc = LossAccumulator::new();
        assert_eq!(acc.tokens(), 0);
        assert_eq!(acc.mean(), 0.0);
        assert_eq!(acc.variance(), 0.0);
        assert_eq!(acc.paired_gain(), None);
    }

    #[test]
    fn test_unpaired() {
        let mut acc = LossAccumulator::new();
        acc.push_chunk(&[1.0, 3.0]);
        assert_eq!(acc.tokens(), 2);
        assert_eq!(acc.mean(), 2.0);
        assert!((acc.variance() - 1.0).abs() < 1e-6);
        assert_eq!(acc.paired_gain(), None);

        let world = WorldId::FixedCorpus {
            seed: 42,
            tokens: 2,
        };
        let receipt = acc.eval_receipt(world);
        assert_eq!(receipt.tokens, 2);
        assert_eq!(receipt.candidate_loss, 2.0);
        assert!((receipt.candidate_variance - 1.0).abs() < 1e-6);
        assert_eq!(receipt.paired_gain, None);
    }

    #[test]
    fn test_paired() {
        let mut acc = LossAccumulator::new();
        acc.pair_chunk(&[1.5, 2.5], &[2.0, 4.0]);
        assert_eq!(acc.tokens(), 2);
        assert_eq!(acc.mean(), 2.0);
        assert_eq!(acc.paired_gain(), Some(1.0));
        assert!((acc.paired_variance().unwrap() - 0.25).abs() < 1e-6);

        let world = WorldId::FixedCorpus {
            seed: 99,
            tokens: 2,
        };
        let receipt = acc.eval_receipt(world);
        assert_eq!(receipt.tokens, 2);
        assert_eq!(receipt.candidate_loss, 2.0);
        assert!((receipt.candidate_variance - 0.25).abs() < 1e-6);
        assert_eq!(receipt.paired_gain, Some(1.0));
    }

    #[test]
    fn test_receipt() {
        let world = WorldId::FixedCorpus {
            seed: 123,
            tokens: 4,
        };
        let losses = [2.0, 2.0, 2.0, 2.0];
        let receipt = ChunkLoss::eval_receipt(world, &losses, None);
        assert_eq!(receipt.tokens, 4);
        assert_eq!(receipt.candidate_loss, 2.0);
        assert_eq!(receipt.candidate_variance, 0.0);
        assert_eq!(receipt.paired_gain, None);
    }
}
