//! Transaction typestates for derivative-free optimization rounds.
//!
//! Inspired by `burn-backend::TransactionOps` and `burn-ir::GraphBoundary`,
//! this module encapsulates the complete Bayesian Optimization transaction
//! lifecycle:
//!
//! ```text
//! ┌────────────────┐          ┌──────────────────┐          ┌─────────────────┐
//! │  AskPlan       │ ───────> │  EvalReceipt     │ ───────> │  TellDecision   │
//! │  (Candidate)   │          │  (Scored Tokens) │          │  (Commit State) │
//! └────────────────┘          └──────────────────┘          └─────────────────┘
//! ```
//!
//! The modeled single-candidate runner binds a receipt to a pending proposal,
//! checks its world/counts/numerical validity, then uses `commit_round` to
//! select a retained identity. Receipt producers and physical buffer commits
//! remain trusted execution boundaries. This does not certify statistical error
//! bounds.

use super::provenance::{PairedImprovement, WorldId, WorldMismatchError};

/// Identity of one pending proposal and the incumbent it was prepared against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoundId {
    pub owner: u64,
    pub proposal: u64,
    pub incumbent: i64,
}

/// Executable commit contract. Reject mismatched evaluations and candidate IDs
/// not newer than the incumbent. The backend allocates observation IDs; this
/// does not certify scores.
pub fn commit_round(
    active: RoundId,
    evaluated: RoundId,
    candidate: i64,
    accepted: bool,
) -> Option<i64> {
    if active.owner != evaluated.owner
        || active.proposal != evaluated.proposal
        || active.incumbent != evaluated.incumbent
        || candidate <= active.incumbent
    {
        None
    } else if accepted {
        Some(candidate)
    } else {
        Some(active.incumbent)
    }
}

/// Step 1: The proposal plan produced by an optimizer `ask`.
#[derive(Debug)]
pub struct AskPlan {
    round: RoundId,
    world: WorldId,
}

impl AskPlan {
    pub const fn new(round: RoundId, world: WorldId) -> Self {
        Self { round, world }
    }

    /// Bind a producer's observations to a plan. Counts and finite values are
    /// checked; hardware execution and statistical accuracy remain obligations
    /// of the producer. The resulting receipt is consumed by the optimizer.
    pub fn record(self, receipt: EvalReceipt) -> Result<Evaluated, &'static str> {
        let planned = match self.world {
            WorldId::FixedCorpus { tokens, .. } | WorldId::RotatingCondition { tokens, .. } => {
                tokens
            }
        };
        if receipt.world != self.world {
            return Err("evaluation world differs from the plan");
        }
        if planned == 0 || receipt.tokens != planned {
            return Err("evaluation does not cover the planned scored tokens");
        }
        if !receipt.candidate_loss.is_finite()
            || !receipt.candidate_variance.is_finite()
            || receipt.candidate_variance < 0.0
            || receipt.paired_gain.is_some_and(|gain| !gain.is_finite())
        {
            return Err("evaluation contains invalid numerical evidence");
        }
        Ok(Evaluated {
            plan: self,
            receipt,
        })
    }
}

/// A checked plan/receipt association, not a proof of measurement accuracy.
#[derive(Debug)]
pub struct Evaluated {
    plan: AskPlan,
    receipt: EvalReceipt,
}

impl Evaluated {
    pub const fn round(&self) -> RoundId {
        self.plan.round
    }

    pub const fn receipt(&self) -> &EvalReceipt {
        &self.receipt
    }
}

/// Step 2: Hardware evaluation receipt returned after forward pass scoring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalReceipt {
    /// Evaluation world matching the planned condition.
    pub world: WorldId,
    /// Number of tokens evaluated in the forward pass.
    pub tokens: usize,
    /// Measured mean token loss for the candidate.
    pub candidate_loss: f64,
    /// Measured token loss variance across the window.
    pub candidate_variance: f64,
    /// Exact paired token loss gain against incumbent, if evaluated in same
    /// buffer.
    pub paired_gain: Option<f64>,
}

impl EvalReceipt {
    pub const fn new(
        world: WorldId,
        tokens: usize,
        candidate_loss: f64,
        candidate_variance: f64,
        paired_gain: Option<f64>,
    ) -> Self {
        Self {
            world,
            tokens,
            candidate_loss,
            candidate_variance,
            paired_gain,
        }
    }

    /// Pair this evaluation receipt against an incumbent score on the same
    /// world.
    pub fn pair_with(
        self,
        incumbent_world: WorldId,
        incumbent_loss: f64,
    ) -> Result<PairedImprovement, WorldMismatchError> {
        incumbent_world
            .pair(incumbent_loss, self.world, self.candidate_loss)
            .map(|paired| paired.with_gain(self.paired_gain))
    }
}

/// Step 3: Final optimizer decision produced by `tell`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TellDecision {
    retained: i64,
    accepted: bool,
}

impl TellDecision {
    pub fn new(
        active: RoundId,
        evaluated: RoundId,
        candidate: i64,
        accepted: bool,
    ) -> Option<Self> {
        commit_round(active, evaluated, candidate, accepted)
            .map(|retained| Self { retained, accepted })
    }

    pub const fn retained(self) -> i64 {
        self.retained
    }

    pub const fn accepted(self) -> bool {
        self.accepted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(step: u64) -> WorldId {
        WorldId::RotatingCondition {
            step,
            window_id: 11,
            tokens: 16,
        }
    }

    fn round() -> RoundId {
        RoundId {
            owner: 2,
            proposal: 3,
            incumbent: 4,
        }
    }

    #[test]
    fn receipt_rejection() {
        for receipt in [
            EvalReceipt::new(world(1), 16, 2.0, 0.0, None),
            EvalReceipt::new(world(0), 15, 2.0, 0.0, None),
            EvalReceipt::new(world(0), 16, f64::NAN, 0.0, None),
            EvalReceipt::new(world(0), 16, 2.0, -1.0, None),
            EvalReceipt::new(world(0), 16, 2.0, f64::INFINITY, None),
            EvalReceipt::new(world(0), 16, 2.0, 0.0, Some(f64::NAN)),
        ] {
            assert!(AskPlan::new(round(), world(0)).record(receipt).is_err());
        }
        let receipt = EvalReceipt::new(world(0), 16, 2.0, 0.0, None);
        assert!(receipt.pair_with(world(1), 3.0).is_err());
        assert_eq!(receipt.pair_with(world(0), 3.0).unwrap().delta, 1.0);
        let evaluation = AskPlan::new(round(), world(0)).record(receipt).unwrap();
        assert_eq!(evaluation.round(), round());
    }

    #[test]
    fn stale_commit() {
        let active = round();
        for accepted in [false, true] {
            for evaluated in [
                RoundId { owner: 7, ..active },
                RoundId {
                    proposal: 7,
                    ..active
                },
                RoundId {
                    incumbent: 7,
                    ..active
                },
            ] {
                assert_eq!(commit_round(active, evaluated, 8, accepted), None);
            }
            for candidate in [i64::MIN, 3, 4] {
                assert_eq!(commit_round(active, active, candidate, accepted), None);
            }
            for candidate in [5, i64::MAX] {
                assert_eq!(
                    commit_round(active, active, candidate, accepted),
                    Some(if accepted {
                        candidate
                    } else {
                        active.incumbent
                    })
                );
            }
        }
    }
}
