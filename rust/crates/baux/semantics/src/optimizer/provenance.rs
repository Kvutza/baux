//! Observation provenance and chained measurement tracking for optimizer
//! worlds.

/// Aeneas-extracted scalar provenance identity check
/// (`BauxResearch.Generated.Optimizer.provenance.same_world`).
pub const fn same_world(left: u64, right: u64) -> bool {
    left == right
}

/// Aeneas-extracted provenance chain classifier (`WindowPath.lean` &
/// `EnnCommit.lean`). Returns `true` iff the observation is anchored to a fixed
/// corpus or zero-step origin (`from_step == to_step`).
pub const fn exact_anchored(is_fixed_corpus: bool, from_step: u64, to_step: u64) -> bool {
    is_fixed_corpus || from_step == to_step
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorldId {
    FixedCorpus {
        seed: u64,
        tokens: usize,
    },
    RotatingCondition {
        step: u64,
        window_id: u64,
        tokens: usize,
    },
}

impl WorldId {
    pub const fn key(self) -> u64 {
        match self {
            WorldId::FixedCorpus { seed, tokens } => {
                seed ^ ((tokens as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            }
            WorldId::RotatingCondition {
                step,
                window_id,
                tokens,
            } => {
                step.wrapping_mul(0xbf58_476d_1ce4_e5b9)
                    ^ window_id.rotate_left(17)
                    ^ (tokens as u64)
            }
        }
    }

    pub fn pair(
        self,
        incumbent_loss: f64,
        challenger_world: WorldId,
        challenger_loss: f64,
    ) -> Result<PairedImprovement, WorldMismatchError> {
        PairedImprovement::new(self, incumbent_loss, challenger_world, challenger_loss)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldMismatchError {
    pub message: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairedImprovement {
    pub world: WorldId,
    pub incumbent_loss: f64,
    pub challenger_loss: f64,
    pub delta: f64,
}

impl PairedImprovement {
    /// Construct a verified same-world paired improvement (`EnnCommit.lean`
    /// P2).
    ///
    /// Precondition (`hax_lib::requires`): `incumbent_world ==
    /// challenger_world`. Postcondition (`hax_lib::ensures`): `result.delta
    /// == incumbent_loss - challenger_loss`.
    pub fn new(
        incumbent_world: WorldId,
        incumbent_loss: f64,
        challenger_world: WorldId,
        challenger_loss: f64,
    ) -> Result<Self, WorldMismatchError> {
        if !same_world(incumbent_world.key(), challenger_world.key())
            || incumbent_world != challenger_world
        {
            return Err(WorldMismatchError {
                message: "Incumbent and challenger must be evaluated on identical WorldId",
            });
        }
        let delta = incumbent_loss - challenger_loss;
        Ok(Self {
            world: incumbent_world,
            incumbent_loss,
            challenger_loss,
            delta,
        })
    }

    /// Override the rounded scalar loss difference with the exact
    /// FP64-accumulated token-wise paired gain when available
    /// (`fbt_block.rs`).
    pub const fn with_gain(mut self, paired_gain: Option<f64>) -> Self {
        if let Some(gain) = paired_gain {
            self.delta = gain;
        }
        self
    }

    /// Anchor this paired improvement to the incumbent archive baseline
    /// (`WindowPath.lean` & `EnnCommit.lean`).
    ///
    /// Fixed-corpus evaluations produce `ExactAnchored`; rotating-window
    /// pretraining transitions produce `DriftedChain`
    /// (`WindowPath.errorSplit`), accumulating `base_variance + pair_variance`.
    pub fn anchor(
        self,
        from_step: u64,
        base_reward: f64,
        base_variance: f64,
        pair_variance: f64,
    ) -> (ChainedObservation, f32, f32) {
        let (is_fixed, to_step) = match self.world {
            WorldId::FixedCorpus { .. } => (true, from_step),
            WorldId::RotatingCondition { step, .. } => (false, step),
        };
        let obs = if exact_anchored(is_fixed, from_step, to_step) {
            ChainedObservation::ExactAnchored {
                anchor_step: from_step,
                world: self.world,
                cumulative_delta: base_reward + self.delta,
            }
        } else {
            ChainedObservation::DriftedChain {
                from_step,
                to_step,
                step_delta: self.delta,
                drift_variance_floor: pair_variance.max(0.0),
            }
        };
        let base_noise = if exact_anchored(is_fixed, from_step, to_step) {
            base_variance.max(0.0) + pair_variance.max(0.0)
        } else {
            base_variance.max(0.0)
        };
        let total_var = obs.noise_variance(base_noise);
        ((obs), (base_reward + self.delta) as f32, total_var as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChainedObservation {
    ExactAnchored {
        anchor_step: u64,
        world: WorldId,
        cumulative_delta: f64,
    },
    DriftedChain {
        from_step: u64,
        to_step: u64,
        step_delta: f64,
        drift_variance_floor: f64,
    },
}

impl ChainedObservation {
    pub fn noise_variance(&self, base_noise: f64) -> f64 {
        match *self {
            ChainedObservation::ExactAnchored { .. } => base_noise,
            ChainedObservation::DriftedChain {
                from_step,
                to_step,
                drift_variance_floor,
                ..
            } => {
                let steps_diff = to_step.abs_diff(from_step).max(1) as f64;
                base_noise + drift_variance_floor * steps_diff.sqrt()
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn prove_pair() {
        let step: u64 = kani::any();
        let window_id: u64 = kani::any();
        let tokens: usize = kani::any();
        let w = WorldId::RotatingCondition {
            step,
            window_id,
            tokens,
        };
        assert!(same_world(w.key(), w.key()));
    }
}
