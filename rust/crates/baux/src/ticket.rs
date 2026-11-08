//! Affine proposal ticket for compile-time single-resolution enforcement.

use ndarray::{Array2, ArrayView2};
use rand::RngCore;

use crate::coordinate::{Coords, DomainSpace, OwnedCoords};
use crate::error::ENNError;
use crate::optimizer::{BayesOptimizer, EvaluationId, TrustRegionState};

/// Affine proposal token representing an in-flight batch of candidate arms.
///
/// Intentionally neither `Copy` nor `Clone`: the Rust compiler enforces at
/// compile time that each issued ticket is resolved at most once via
/// [`BayesOptimizer::tell_ticket`] or [`BayesOptimizer::cancel_ticket`].
#[must_use = "an issued ProposalTicket must be consumed by tell_ticket or cancel_ticket"]
#[derive(Debug)]
pub struct ProposalTicket {
    x: OwnedCoords<DomainSpace>,
    ids: Option<Vec<EvaluationId>>,
}

impl ProposalTicket {
    #[inline(always)]
    pub fn coords(&self) -> Coords<'_, DomainSpace> {
        self.x.borrow()
    }

    #[inline(always)]
    pub fn as_array(&self) -> &Array2<f64> {
        self.x.as_array()
    }

    #[inline(always)]
    pub fn ids(&self) -> Option<&[EvaluationId]> {
        self.ids.as_deref()
    }

    #[inline(always)]
    pub fn num_arms(&self) -> usize {
        self.x.as_array().nrows()
    }
}

impl BayesOptimizer {
    /// Issue an affine [`ProposalTicket`] in natural domain coordinates.
    pub fn ask_ticket(
        &mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<ProposalTicket, ENNError> {
        let natural = self.ask_natural(num_arms, rng)?;
        let ids = match &self.regions {
            TrustRegionState::Multi {
                pending: Some(prop),
                ..
            } => {
                let n = natural.nrows();
                let total = prop.ids.len();
                Some(prop.ids[total.saturating_sub(n)..].to_vec())
            }
            _ => None,
        };
        Ok(ProposalTicket {
            x: OwnedCoords::new(natural),
            ids,
        })
    }

    /// Consume an affine [`ProposalTicket`] by value and record its measured
    /// outcomes.
    pub fn tell_ticket(
        &mut self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let ProposalTicket { x, ids } = ticket;
        if let Some(eval_ids) = ids {
            if self.regions.is_multi() && y.ncols() == 1 {
                return self.complete_ids(&eval_ids, y, yvar, rng);
            }
        }
        self.tell_natural(&x.as_array().view(), y, yvar, rng)
    }

    /// Consume an affine [`ProposalTicket`] by value and retire its pending
    /// evaluations without controller penalty.
    pub fn cancel_ticket(
        &mut self,
        ticket: ProposalTicket,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let ProposalTicket { ids, .. } = ticket;
        if let Some(eval_ids) = ids {
            if self.regions.is_multi() {
                return self.cancel_ids(&eval_ids, rng);
            }
        }
        Ok(())
    }
}
