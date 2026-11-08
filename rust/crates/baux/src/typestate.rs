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
        let mut inner = ENNSurrogate::new(self.config.clone());
        inner.fit(x, y, yvar, rng)?;
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

    pub fn append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.state.inner.fit_append(x_new, y_new, yvar_new, rng)
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
    /// the [`Active`] typestate.
    pub fn warmup_design(
        &mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array2<f64>, ENNError> {
        self.inner.ask_natural(num_arms, rng)
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
        self.inner.tell_natural(x, &y_canon.view(), None, rng)?;
        Ok(TypedOptimizer {
            inner: self.inner,
            sense: self.sense,
            phase: PhantomData,
        })
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Active> {
    /// Propose an affine [`ProposalTicket`] in the active search phase.
    pub fn ask_ticket(
        &mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<ProposalTicket, ENNError> {
        self.inner.ask_ticket(num_arms, rng)
    }

    /// Consume an affine [`ProposalTicket`] and record observations under sense
    /// `O`.
    pub fn tell_ticket(
        &mut self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.inner.tell_ticket(ticket, &y_canon.view(), None, rng)
    }

    /// Pure functional candidate proposal: forks `self` and returns `(next,
    /// ticket)`.
    pub fn ask_pure(
        &self,
        num_arms: usize,
        key: crate::hash::Key,
    ) -> Result<(Self, ProposalTicket), ENNError> {
        let mut next = self.fork()?;
        let mut rng = key.prng();
        let ticket = next.ask_ticket(num_arms, &mut rng)?;
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
        let mut next = self.fork()?;
        let mut rng = key.prng();
        next.tell_ticket(ticket, y, &mut rng)?;
        Ok(next)
    }
}
