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
#[diagnostic::on_move(
    message = "`ProposalTicket` was already resolved or cancelled",
    label = "each `ProposalTicket` is an affine token and may be consumed at most once",
    note = "issue a fresh ticket via `ask_ticket`, `ask_pure`, or `ask_step` for the next round"
)]
#[derive(Debug)]
pub struct ProposalTicket {
    x: OwnedCoords<DomainSpace>,
    ids: Option<Vec<EvaluationId>>,
    origin_digest: Option<[u8; 32]>,
}

impl !Clone for ProposalTicket {}
impl !Copy for ProposalTicket {}

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

    #[inline(always)]
    pub fn origin_digest(&self) -> Option<[u8; 32]> {
        self.origin_digest
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
        let mut raw = Vec::with_capacity(natural.len() * 8);
        for &v in natural.iter() {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        self.proposals.ticket_tag = Some(crate::artifact::blake3::hash(&raw));
        let origin_digest = Some(self.digest());
        Ok(ProposalTicket {
            x: OwnedCoords::new(natural),
            ids,
            origin_digest,
        })
    }

    fn verify_ticket(&mut self, origin_digest: Option<[u8; 32]>) -> Result<(), ENNError> {
        if let Some(expected) = origin_digest {
            if self.digest() != expected {
                return Err(ENNError::InvalidParameter(
                    "ProposalTicket state digest mismatch: ticket belongs to a different optimizer branch or epoch"
                        .to_string(),
                ));
            }
        }
        self.proposals.ticket_tag = None;
        Ok(())
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
        let ProposalTicket {
            x,
            ids,
            origin_digest,
        } = ticket;
        self.verify_ticket(origin_digest)?;
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
        let ProposalTicket {
            ids, origin_digest, ..
        } = ticket;
        self.verify_ticket(origin_digest)?;
        if let Some(eval_ids) = ids {
            if self.regions.is_multi() {
                return self.cancel_ids(&eval_ids, rng);
            }
        }
        Ok(())
    }
}
