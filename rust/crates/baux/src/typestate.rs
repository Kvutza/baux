//! Compile-time typestate surrogate and optimizer protocols.

use core::marker::PhantomData;
use std::borrow::Cow;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;
use crate::model::ENN;
use crate::optimizer::BayesOptimizer;
use crate::params::ENNParams;
use crate::sense::{BetterScore, Maximize, ObjectiveSense};
use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
use crate::surrogate_api::{ObservationView, Predictor};
pub use crate::ticket::ProposalTicket;

#[path = "typestate/flow.rs"]
mod flow;
pub use flow::{Pending, Ready, StepTrace, pmap_branches, scan_step};

/// Typestate marker for an unfitted surrogate model.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unfitted;

/// Typestate payload for a fitted ENN surrogate model.
#[derive(Clone)]
pub struct Fitted {
    inner: ENNSurrogate,
}

/// Compile-time typestate ENN surrogate preventing `predict` before `fit`.
#[diagnostic::on_type_error(
    note = "surrogate typestate mismatch (`{Expected}` vs `{Found}`): call `.fit(...)` on `TypedSurrogate<Unfitted>` before `.predict(...)`"
)]
#[derive(Clone)]
pub struct TypedSurrogate<State = Unfitted> {
    config: ENNSurrogateConfig,
    state: State,
}

impl TypedSurrogate<Unfitted> {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config,
            state: Unfitted,
        }
    }

    /// Fit the surrogate on initial observations and transition typestate to
    /// [`TypedSurrogate<Fitted>`].
    pub fn fit(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<TypedSurrogate<Fitted>, ENNError> {
        let inner = ENNSurrogate::new(self.config.clone());
        let inner = inner.fit(x, y, yvar, rng)?;
        Ok(TypedSurrogate {
            config: self.config,
            state: Fitted { inner },
        })
    }
}

impl TypedSurrogate<Fitted> {
    #[inline(always)]
    pub fn model(&self) -> &ENN {
        self.state
            .inner
            .model()
            .expect("TypedSurrogate<Fitted> typestate guarantees fitted model")
    }

    #[inline(always)]
    pub fn params(&self) -> &ENNParams {
        self.state
            .inner
            .params()
            .expect("TypedSurrogate<Fitted> typestate guarantees fitted params")
    }

    /// Posterior prediction on a guaranteed-fitted surrogate.
    pub fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        self.state.inner.predict(x)
    }

    /// Append new observations to the fitted surrogate, returning the updated
    /// fitted surrogate with pure value semantics (consumed `self`).
    pub fn append(
        mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.state.inner = self.state.inner.fit_append(x_new, y_new, yvar_new, rng)?;
        Ok(self)
    }
}

impl Predictor for TypedSurrogate<Fitted> {
    #[inline]
    fn predict_batch(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        self.predict(x)
    }
}

impl ObservationView for TypedSurrogate<Fitted> {
    type Row<'a> = Cow<'a, [f64]>;

    #[inline]
    fn obs_len(&self) -> usize {
        self.state.inner.observation_count().unwrap_or(0)
    }

    #[inline]
    fn view_x<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError> {
        self.state.inner.x_slice(idx)
    }

    #[inline]
    fn view_y<'a>(&'a self, idx: usize) -> Result<Self::Row<'a>, ENNError> {
        self.state.inner.y_slice(idx)
    }
}

/// Typestate marker: optimizer requires initial warm-up observations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Warmup;

/// Typestate marker: optimizer has completed warm-up and is ready for active
/// search.
#[derive(Clone, Copy, Debug, Default)]
pub struct Active;

/// Sense-parametric typestate Bayesian optimizer.
pub struct TypedOptimizer<O: ObjectiveSense = Maximize, Phase = Warmup> {
    inner: BayesOptimizer,
    sense: O,
    phase: PhantomData<fn() -> Phase>,
}

impl<O: ObjectiveSense, Phase> Clone for TypedOptimizer<O, Phase> {
    fn clone(&self) -> Self {
        self.fork().expect("TypedOptimizer fork failed")
    }
}

impl<O: ObjectiveSense, Phase> TypedOptimizer<O, Phase> {
    #[inline(always)]
    pub fn inner(&self) -> &BayesOptimizer {
        &self.inner
    }

    #[inline(always)]
    pub fn sense(&self) -> O {
        self.sense
    }

    /// Fork this typestate optimizer into an independent value branch.
    pub fn fork(&self) -> Result<Self, ENNError> {
        Ok(Self {
            inner: self.inner.fork()?,
            sense: self.sense,
            phase: PhantomData,
        })
    }

    #[inline(always)]
    pub fn digest(&self) -> [u8; 32] {
        self.inner.digest()
    }

    #[inline(always)]
    pub fn digest_hex(&self) -> String {
        self.inner.digest_hex()
    }

    fn canonicalize_y(&self, y: &ArrayView2<f64>) -> Array2<f64> {
        let sense = self.sense;
        y.mapv(|val| sense.score(val).raw())
    }

    /// Current incumbent point in natural domain coordinates.
    pub fn best_x(&self) -> Option<Array1<f64>> {
        self.inner.incumbent_x()
    }

    /// Current incumbent objective value in natural (untransformed) units.
    pub fn best_y(&self) -> Option<f64> {
        let raw = self.inner.y_scalar()?.first().copied()?;
        Some(self.sense.invert(BetterScore(raw)))
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Warmup> {
    pub fn new(inner: BayesOptimizer, sense: O) -> Self {
        Self {
            inner,
            sense,
            phase: PhantomData,
        }
    }

    /// Issue warm-up design points and evaluate/seed them to transition into
    /// the [`Active`] typestate. Consumes `self` and returns `(design_points,
    /// next_opt)`.
    pub fn warmup_design(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (next, x) = self.inner.ask_natural(num_arms, rng)?;
        self.inner = next;
        Ok((x, self))
    }

    /// Ingest warm-up observations and transition to [`TypedOptimizer<O,
    /// Active>`].
    pub fn activate(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<TypedOptimizer<O, Active>, ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.inner = self.inner.tell_natural(x, &y_canon.view(), None, rng)?;
        Ok(TypedOptimizer {
            inner: self.inner,
            sense: self.sense,
            phase: PhantomData,
        })
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Active> {
    /// Propose an affine [`ProposalTicket`] in the active search phase.
    /// Consumes `self`.
    pub fn ask_ticket(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(ProposalTicket, Self), ENNError> {
        let (ticket, next) = self.inner.ask_ticket(num_arms, rng)?;
        self.inner = next;
        Ok((ticket, self))
    }

    /// Consume an affine [`ProposalTicket`] and record observations under sense
    /// `O`. Consumes `self`.
    pub fn tell_ticket(
        mut self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.inner = self.inner.tell_ticket(ticket, &y_canon.view(), None, rng)?;
        Ok(self)
    }

    /// Retire an affine [`ProposalTicket`] without controller penalty under
    /// sense `O`. Consumes `self`.
    pub fn cancel_ticket(
        mut self,
        ticket: ProposalTicket,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.inner = self.inner.cancel_ticket(ticket, rng)?;
        Ok(self)
    }

    /// Propose candidate points in natural coordinates with consumed `self`.
    pub fn ask_natural(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (next, x) = self.inner.ask_natural(num_arms, rng)?;
        self.inner = next;
        Ok((x, self))
    }

    /// Ingest observations in natural coordinates with consumed `self`.
    pub fn tell_natural(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.inner = self.inner.tell_natural(x, &y_canon.view(), yvar, rng)?;
        Ok(self)
    }

    /// Propose candidate points in unit coordinates with consumed `self`.
    pub fn ask(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (next, x) = self.inner.ask(num_arms, rng)?;
        self.inner = next;
        Ok((x, self))
    }

    /// Ingest observations in unit coordinates with consumed `self`.
    pub fn tell(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.inner = self.inner.tell(x, &y_canon.view(), yvar, rng)?;
        Ok(self)
    }

    /// Pure functional candidate proposal: forks `self` and returns `(next,
    /// ticket)`.
    pub fn ask_pure(
        &self,
        num_arms: usize,
        key: crate::hash::Key,
    ) -> Result<(Self, ProposalTicket), ENNError> {
        let next = self.fork()?;
        let mut rng = key.prng();
        let (ticket, next) = next.ask_ticket(num_arms, &mut rng)?;
        Ok((next, ticket))
    }

    /// Pure functional observation update: forks `self`, consumes `ticket`, and
    /// returns `next`.
    pub fn tell_pure(
        &self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        key: crate::hash::Key,
    ) -> Result<Self, ENNError> {
        let next = self.fork()?;
        let mut rng = key.prng();
        next.tell_ticket(ticket, y, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{AcqStage, InitStage, ModelStage, PerturbStage};
    use crate::sense::Minimize;
    use ndarray::{Axis, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn surrogate_append() {
        let mut rng = StdRng::seed_from_u64(42);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![[0.0], [1.0], [4.0]];

        let unfitted = TypedSurrogate::new(ENNSurrogateConfig {
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        });

        // fit consumes unfitted -> TypedSurrogate<Fitted>
        let fitted = unfitted.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        assert_eq!(fitted.obs_len(), 3);

        let pred0 = fitted.predict(&array![[0.5, 0.5]].view()).unwrap();
        assert_eq!(pred0.mu.shape(), &[1, 1]);

        // Pure clone
        let branch = fitted.clone();
        assert_eq!(branch.obs_len(), 3);

        // append consumes self by value -> Result<Self, ENNError>
        let x_new = array![[3.0, 3.0]];
        let y_new = array![[9.0]];
        let appended = fitted
            .append(&x_new.view(), &y_new.view(), None, &mut rng)
            .unwrap();
        assert_eq!(appended.obs_len(), 4);

        // Verify original cloned branch was unaffected (pure value semantics)
        assert_eq!(branch.obs_len(), 3);
    }

    #[test]
    fn optimizer_flow() {
        let mut rng = StdRng::seed_from_u64(123);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.5, Minimize);

        let warmup = pipe.build(bounds, &mut rng).unwrap();

        // warmup_design consumes self -> (Array2<f64>, Self)
        let (x0, warmup) = warmup.warmup_design(4, &mut rng).unwrap();
        assert_eq!(x0.nrows(), 4);

        let y0 = x0
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));

        // activate consumes warmup -> TypedOptimizer<Minimize, Active>
        let active = warmup.activate(&x0.view(), &y0.view(), &mut rng).unwrap();

        // Clone active optimizer to test value independence
        let branch_active = active.clone();

        // ask_ticket consumes self -> (ProposalTicket, Self)
        let (ticket, active_after_ask) = active.ask_ticket(2, &mut rng).unwrap();
        assert_eq!(ticket.num_arms(), 2);

        let y_step = ticket
            .as_array()
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));

        // tell_ticket consumes self -> Self
        let active_after_tell = active_after_ask
            .tell_ticket(ticket, &y_step.view(), &mut rng)
            .unwrap();
        assert!(active_after_tell.best_y().is_some());

        // branch_active is unchanged and independent
        assert_ne!(active_after_tell.digest(), branch_active.digest());
    }
}
