//! Shared host/device ABI for Muon and AdamW updates.
//!
//! Imported as a crate dependency by both host and device implementations.

#[path = "muon/reference.rs"]
pub mod reference;

// SAFETY: scalar repr(C) records with no ownership or references.
#[cfg(feature = "cuda")]
unsafe impl cuda_core::DeviceCopy for Shape {}
#[cfg(feature = "cuda")]
unsafe impl cuda_core::DeviceCopy for Step {}
#[cfg(feature = "cuda")]
unsafe impl cuda_core::DeviceCopy for Polynomial {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub rows: u32,
    pub cols: u32,
    pub batch: u32,
}

impl Shape {
    pub fn new(rows: u32, cols: u32, batch: u32) -> Result<Self, String> {
        let elements = u64::from(rows)
            .checked_mul(u64::from(cols))
            .and_then(|n| n.checked_mul(u64::from(batch)));
        if rows == 0
            || cols == 0
            || batch == 0
            || batch > 65535
            || elements.is_none_or(|n| n > u64::from(u32::MAX) - 8192)
        {
            return Err("optimizer matrix dimensions must be nonzero and fit u32".into());
        }
        Ok(Self { rows, cols, batch })
    }

    pub fn elements(self) -> usize {
        self.rows as usize * self.cols as usize * self.batch as usize
    }

    pub fn wide(self) -> Self {
        Self {
            rows: self.rows.min(self.cols),
            cols: self.rows.max(self.cols),
            batch: self.batch,
        }
    }

    pub fn chunks(self) -> u32 {
        (self.rows * self.cols).div_ceil(8192)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub rate: f32,
    pub momentum: f32,
    pub decay: f32,
    pub epsilon: f32,
    pub beta2: f32,
    pub bias1: f32,
    pub bias2: f32,
    pub scale: f32,
}

impl Default for Step {
    fn default() -> Self {
        Self {
            rate: 0.02,
            momentum: 0.95,
            decay: 0.0,
            epsilon: 1.0e-8,
            beta2: 0.95,
            bias1: 1.0,
            bias2: 1.0,
            scale: 1.0,
        }
    }
}

impl Step {
    pub fn muon() -> Self {
        Self::default()
    }

    /// AdamW settings used by the PISA from-scratch runs. Muon is reserved
    /// for matrix projections; embeddings, norms, biases, and controls retain
    /// the paper's smaller AdamW step.
    pub fn adamw() -> Self {
        Self {
            rate: 3.0e-4,
            momentum: 0.9,
            decay: 0.1,
            epsilon: 1.0e-8,
            beta2: 0.95,
            bias1: 1.0,
            bias2: 1.0,
            scale: 1.0,
        }
    }

    pub fn validate(self) -> Result<(), String> {
        if [self.rate, self.decay, self.scale]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.momentum.is_finite()
            || !(0.0..1.0).contains(&self.momentum)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || [self.bias1, self.bias2]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self.rate * self.decay > 1.0
        {
            return Err("invalid optimizer step".into());
        }
        Ok(())
    }

    /// AdamW bias correction uses the actual number of updates to this tensor.
    /// The caller supplies the learning-rate schedule; no hidden schedule
    /// exists.
    pub fn adam(mut self, updates: u32) -> Result<Self, String> {
        if updates == 0 {
            return Err("AdamW update count starts at one".into());
        }
        self.bias1 = 1.0 - self.momentum.powf(updates as f32);
        self.bias2 = 1.0 - self.beta2.powf(updates as f32);
        self.validate()?;
        Ok(self)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Polynomial {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

/// Polar Express, Algorithm 1, Amsel et al., arXiv:2505.16932v2.
/// Apply the paper's coefficient safety factor and initial 1.01 Frobenius
/// margin. This is the five-step Muon variant, not modded-nanoGPT's tuned ANVIL
/// variant.
pub const POLAR: [Polynomial; 5] = [
    Polynomial {
        a: 8.28721201814563_f64 as f32 / 1.01,
        b: -23.595886519098837_f64 as f32 / (1.01 * 1.01 * 1.01),
        c: 17.300387312530933_f64 as f32 / (1.01 * 1.01 * 1.01 * 1.01 * 1.01),
    },
    Polynomial {
        a: 4.107059111542203_f64 as f32 / 1.01,
        b: -2.9478499167379106_f64 as f32 / (1.01 * 1.01 * 1.01),
        c: 0.5448431082926601_f64 as f32 / (1.01 * 1.01 * 1.01 * 1.01 * 1.01),
    },
    Polynomial {
        a: 3.9486908534822946_f64 as f32 / 1.01,
        b: -2.908902115962949_f64 as f32 / (1.01 * 1.01 * 1.01),
        c: 0.5518191394370137_f64 as f32 / (1.01 * 1.01 * 1.01 * 1.01 * 1.01),
    },
    Polynomial {
        a: 3.3184196573706015_f64 as f32 / 1.01,
        b: -2.488488024314874_f64 as f32 / (1.01 * 1.01 * 1.01),
        c: 0.51004894012372_f64 as f32 / (1.01 * 1.01 * 1.01 * 1.01 * 1.01),
    },
    Polynomial {
        a: 2.300652019954817_f64 as f32 / 1.01,
        b: -1.6689039845747493_f64 as f32 / (1.01 * 1.01 * 1.01),
        c: 0.4188073119525673_f64 as f32 / (1.01 * 1.01 * 1.01 * 1.01 * 1.01),
    },
];
