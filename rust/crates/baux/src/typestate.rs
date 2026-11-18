//! Compile-time typestate surrogate and optimizer protocols.

use core::marker::PhantomData;
use std::borrow::Cow;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::config::{ObjectivePolicy, OptimizerConfig, OptimizerSpec};
use crate::coordinate::DomainFrame;
use crate::error::ENNError;
use crate::hash::AffineKey;
use crate::model::ENN;
use crate::optimizer::{BayesOptimizer, EvaluationId};
use crate::params::ENNParams;
pub use crate::scratch::ImLog;
use crate::sense::{BetterScore, Maximize, ObjectiveSense};
use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, SurrogatePrediction};
use crate::surrogate_api::{ObservationView, Predictor};
pub use crate::ticket::ProposalTicket;
use crate::unified_opt::{CpuBackend, OptBackend, UnifiedEngine};

#[path = "typestate/flow.rs"]
mod flow;
pub use flow::{Pending, Ready, StepTrace, pmap_branches, scan_step};

/// Typestate marker for an unfitted surrogate model.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unfitted;

/// Typestate payload for a fitted ENN surrogate model with $O(1)$ `Arc`
/// sharing.
#[derive(Clone)]
pub struct Fitted {
    inner: Arc<ENNSurrogate>,
}

/// Compile-time typestate ENN surrogate preventing `predict` before `fit`.
#[diagnostic::on_type_error(
    note = "surrogate typestate mismatch (`{Expected}` vs `{Found}`): call `.fit(...)` on `TypedSurrogate<Unfitted>` before `.predict(...)`"
)]
#[derive(Clone)]
pub struct TypedSurrogate<State = Unfitted> {
    config: Arc<ENNSurrogateConfig>,
    state: State,
}

impl TypedSurrogate<Unfitted> {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config: Arc::new(config),
            state: Unfitted,
        }
    }

    /// Pure functional surrogate fit consuming an [`AffineKey`] and returning
    /// `(fitted, next_key)`.
    pub fn fit_key(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        key: AffineKey,
    ) -> Result<(TypedSurrogate<Fitted>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut prng = draw_key.prng();
        let inner = ENNSurrogate::new((*self.config).clone());
        let inner = inner.fit(x, y, yvar, &mut prng)?;
        Ok((
            TypedSurrogate {
                config: self.config,
                state: Fitted {
                    inner: Arc::new(inner),
                },
            },
            next_key,
        ))
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
        let (fitted, _) = self.fit_key(x, y, yvar, AffineKey::new(rng.next_u64()))?;
        Ok(fitted)
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

    /// Pure functional observation append consuming `self` and an [`AffineKey`]
    /// and returning `(updated, next_key)` with FBIP in-place mutation when
    /// uniquely owned.
    pub fn append_key(
        mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        key: AffineKey,
    ) -> Result<(Self, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut prng = draw_key.prng();
        let inner = Arc::try_unwrap(self.state.inner).unwrap_or_else(|arc| (*arc).clone());
        self.state.inner = Arc::new(inner.fit_append(x_new, y_new, yvar_new, &mut prng)?);
        Ok((self, next_key))
    }

    /// Append new observations to the fitted surrogate, returning the updated
    /// fitted surrogate with pure value semantics (consumed `self` and FBIP
    /// in-place mutation when uniquely owned).
    pub fn append(
        self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let (next, _) = self.append_key(x_new, y_new, yvar_new, AffineKey::new(rng.next_u64()))?;
        Ok(next)
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

/// Sense- and backend-parametric typestate Bayesian optimizer.
pub struct TypedOptimizer<O: ObjectiveSense = Maximize, Phase = Warmup, B: OptBackend = CpuBackend>
{
    engine: UnifiedEngine,
    sense: O,
    backend: B,
    phase: PhantomData<fn() -> Phase>,
}

impl<O: ObjectiveSense, Phase, B: OptBackend> Clone for TypedOptimizer<O, Phase, B> {
    fn clone(&self) -> Self {
        self.fork().expect("TypedOptimizer fork failed")
    }
}

impl<O: ObjectiveSense, Phase> TypedOptimizer<O, Phase, CpuBackend> {
    #[inline(always)]
    pub fn inner(&self) -> &BayesOptimizer {
        self.engine
            .as_cpu()
            .expect("CpuBackend TypedOptimizer holds Cpu UnifiedEngine")
    }
}

impl<O: ObjectiveSense, Phase, B: OptBackend> TypedOptimizer<O, Phase, B> {
    #[inline(always)]
    pub fn engine(&self) -> &UnifiedEngine {
        &self.engine
    }

    #[inline(always)]
    pub fn into_engine(self) -> UnifiedEngine {
        self.engine
    }

    #[inline(always)]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline(always)]
    pub fn device_name(&self) -> &'static str {
        self.engine.device_name()
    }

    #[inline(always)]
    pub fn sense(&self) -> O {
        self.sense
    }

    /// Fork this typestate optimizer into an independent value branch in $O(1)$
    /// via `Arc` sharing.
    pub fn fork(&self) -> Result<Self, ENNError> {
        Ok(Self {
            engine: self.engine.fork()?,
            sense: self.sense,
            backend: self.backend.clone(),
            phase: PhantomData,
        })
    }

    /// True when `self` and `other` share identical `Arc`-backed storage.
    #[inline(always)]
    pub fn shares_storage(&self, other: &Self) -> bool {
        self.engine.shares_storage(&other.engine)
    }

    #[inline(always)]
    pub fn digest(&self) -> [u8; 32] {
        self.engine.digest()
    }

    #[inline(always)]
    pub fn digest_hex(&self) -> String {
        self.engine.digest_hex()
    }

    fn canonicalize_y(&self, y: &ArrayView2<f64>) -> Array2<f64> {
        let sense = self.sense;
        y.mapv(|val| sense.score(val).raw())
    }

    /// Current incumbent point in natural domain coordinates.
    pub fn best_x(&self) -> Option<Array1<f64>> {
        self.engine.best_x()
    }

    /// Current incumbent objective value in natural (untransformed) units.
    pub fn best_y(&self) -> Option<f64> {
        let raw = self.engine.best_y()?;
        Some(self.sense.invert(BetterScore(raw)))
    }

    /// Number of retained observations across the optimizer.
    #[inline(always)]
    pub fn num_obs(&self) -> usize {
        self.engine.num_obs()
    }

    /// Number of configured trust-region slots.
    #[inline(always)]
    pub fn num_regions(&self) -> usize {
        self.engine.num_regions()
    }

    /// Current trust-region length.
    #[inline(always)]
    pub fn tr_length(&self) -> f64 {
        self.engine.tr_length()
    }

    /// Warm-up `(completed, budget)` progress when in the initialization phase.
    #[inline(always)]
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        self.engine.init_progress()
    }

    /// Outstanding multi-region evaluation identities in ask order.
    #[inline(always)]
    pub fn pending_ids(&self) -> Option<Vec<EvaluationId>> {
        self.engine.pending_ids()
    }

    /// Serialize canonical optimizer state with a BLAKE3 integrity tag.
    pub fn freeze(&self) -> Result<Vec<u8>, ENNError> {
        self.engine.freeze()
    }

    /// Restore an equivalent optimizer state from a BLAKE3-verified checkpoint.
    pub fn thaw(&self, bytes: &[u8]) -> Result<Self, ENNError> {
        Ok(Self {
            engine: self.engine.thaw(bytes)?,
            sense: self.sense,
            backend: self.backend.clone(),
            phase: PhantomData,
        })
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Warmup, CpuBackend> {
    pub fn new(inner: BayesOptimizer, sense: O) -> Self {
        Self::from_engine(UnifiedEngine::from_cpu(inner), sense, CpuBackend)
    }
}

impl<O: ObjectiveSense, B: OptBackend> TypedOptimizer<O, Warmup, B> {
    pub fn from_engine(engine: UnifiedEngine, sense: O, backend: B) -> Self {
        Self {
            engine,
            sense,
            backend,
            phase: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_spec(
        domain: DomainFrame,
        config: OptimizerConfig,
        spec: &OptimizerSpec,
        policy: Option<ObjectivePolicy>,
        num_init: usize,
        sense: O,
        backend: B,
        key: AffineKey,
    ) -> Result<(Self, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let engine = backend.from_spec(domain, config, spec, policy, num_init, draw_key)?;
        Ok((Self::from_engine(engine, sense, backend), next_key))
    }

    /// Pure functional warm-up design step consuming `self` and an
    /// [`AffineKey`] and returning `(design_points, next_opt, next_key)`.
    pub fn design_step(
        mut self,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(Array2<f64>, Self, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut prng = draw_key.prng();
        let (next, x) = self.engine.ask_natural(num_arms, &mut prng)?;
        self.engine = next;
        Ok((x, self, next_key))
    }

    /// Pure functional warm-up activation consuming `self` and an [`AffineKey`]
    /// and transitioning to [`TypedOptimizer<O, Active, B>`].
    pub fn activate_step(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Active, B>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let y_canon = self.canonicalize_y(y);
        let mut prng = draw_key.prng();
        self.engine = self
            .engine
            .tell_natural(x, &y_canon.view(), None, &mut prng)?;
        Ok((
            TypedOptimizer {
                engine: self.engine,
                sense: self.sense,
                backend: self.backend,
                phase: PhantomData,
            },
            next_key,
        ))
    }

    /// Pure functional warm-up activation transitioning directly into
    /// [`TypedOptimizer<O, Ready, B>`] for `ask_step` / `scan_step` /
    /// `pmap_branches`.
    pub fn activate_ready(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Ready, B>, AffineKey), ENNError> {
        let (active, next_key) = self.activate_step(x, y, key)?;
        Ok((active.into_ready(), next_key))
    }

    /// Issue warm-up design points and evaluate/seed them to transition into
    /// the [`Active`] typestate. Consumes `self` and returns `(design_points,
    /// next_opt)`.
    pub fn warmup_design(
        self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (x, next, _) = self.design_step(num_arms, AffineKey::new(rng.next_u64()))?;
        Ok((x, next))
    }

    /// Ingest warm-up observations and transition to [`TypedOptimizer<O,
    /// Active, B>`].
    pub fn activate(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<TypedOptimizer<O, Active, B>, ENNError> {
        let (active, _) = self.activate_step(x, y, AffineKey::new(rng.next_u64()))?;
        Ok(active)
    }
}

impl<O: ObjectiveSense, B: OptBackend> TypedOptimizer<O, Active, B> {
    /// Propose an affine [`ProposalTicket`] in the active search phase.
    /// Consumes `self`.
    pub fn ask_ticket(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(ProposalTicket, Self), ENNError> {
        let (ticket, next) = self.engine.ask_ticket(num_arms, rng)?;
        self.engine = next;
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
        self.engine = self
            .engine
            .tell_ticket(ticket, &y_canon.view(), None, rng)?;
        Ok(self)
    }

    /// Retire an affine [`ProposalTicket`] without controller penalty under
    /// sense `O`. Consumes `self`.
    pub fn cancel_ticket(
        mut self,
        ticket: ProposalTicket,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.engine = self.engine.cancel_ticket(ticket, rng)?;
        Ok(self)
    }

    /// Propose candidate points in natural coordinates with consumed `self`.
    pub fn ask_natural(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (next, x) = self.engine.ask_natural(num_arms, rng)?;
        self.engine = next;
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
        self.engine = self.engine.tell_natural(x, &y_canon.view(), yvar, rng)?;
        Ok(self)
    }

    /// Propose candidate points in unit coordinates with consumed `self`.
    pub fn ask(
        mut self,
        num_arms: usize,
        rng: &mut dyn RngCore,
    ) -> Result<(Array2<f64>, Self), ENNError> {
        let (next, x) = self.engine.ask(num_arms, rng)?;
        self.engine = next;
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
        self.engine = self.engine.tell(x, &y_canon.view(), yvar, rng)?;
        Ok(self)
    }

    /// Complete a subset of pending regional evaluations by identity under
    /// sense `O`.
    pub fn complete_ids(
        mut self,
        ids: &[EvaluationId],
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let y_canon = self.canonicalize_y(y);
        self.engine = self.engine.complete_ids(ids, &y_canon.view(), yvar, rng)?;
        Ok(self)
    }

    /// Retire unevaluated pending regional rows without controller penalty.
    pub fn cancel_ids(
        mut self,
        ids: &[EvaluationId],
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self.engine = self.engine.cancel_ids(ids, rng)?;
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
    use crate::unified_opt::{AnyBackend, CudaBackend};
    use ndarray::{Axis, array};

    #[test]
    fn surrogate_append() {
        let key = AffineKey::new(42);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![[0.0], [1.0], [4.0]];

        let unfitted = TypedSurrogate::new(ENNSurrogateConfig {
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        });

        let (fitted, key) = unfitted.fit_key(&x.view(), &y.view(), None, key).unwrap();
        assert_eq!(fitted.obs_len(), 3);

        let pred = fitted.predict(&array![[0.5, 0.5]].view()).unwrap();
        assert_eq!(pred.mu.shape(), &[1, 1]);

        let branch = fitted.clone();
        assert_eq!(branch.obs_len(), 3);

        let x_new = array![[3.0, 3.0]];
        let y_new = array![[9.0]];
        let (appended, key) = fitted
            .append_key(&x_new.view(), &y_new.view(), None, key)
            .unwrap();
        let _ = key.into_key();
        assert_eq!(appended.obs_len(), 4);
        assert_eq!(branch.obs_len(), 3);
    }

    #[test]
    fn optimizer_flow() {
        let key = AffineKey::new(123);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);

        let (warmup, key) = pipe.build_key(bounds, key).unwrap();
        let (x_init, warmup, key) = warmup.design_step(4, key).unwrap();
        assert_eq!(x_init.nrows(), 4);

        let y_init = x_init
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));

        let (ready, key) = warmup
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap();
        let branch_ready = ready.clone();
        assert!(ready.shares_storage(&branch_ready));

        let (pending, ticket, key) = ready.ask_step(2, key).unwrap();
        assert_eq!(ticket.num_arms(), 2);

        let y_step = ticket
            .as_array()
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));

        let (ready_next, key) = pending.tell_step(ticket, &y_step.view(), key).unwrap();
        let _ = key.into_key();
        assert!(ready_next.best_y().is_some());
        assert_eq!(ready_next.num_obs(), 6);
        assert_ne!(ready_next.digest(), branch_ready.digest());
    }

    #[test]
    fn any_backend() {
        let key = AffineKey::new(77);
        let bounds = array![[-3.0, 3.0], [-3.0, 3.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);

        let (warmup, key) = pipe.with_backend(bounds, AnyBackend::Cpu, key).unwrap();
        assert_eq!(warmup.device_name(), "cpu");
        let (x_init, warmup, key) = warmup.design_step(4, key).unwrap();
        let y_init = x_init
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));
        let (ready, key) = warmup
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap();
        let (ready, key, traces) = scan_step(ready, key, 2, 2, |_, x| {
            Ok(x.map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
                .insert_axis(Axis(1)))
        })
        .unwrap();
        let _ = key.into_key();
        assert_eq!(traces.len(), 2);
        assert_eq!(ready.num_obs(), 8);
    }

    #[cfg(not(all(target_os = "linux", target_arch = "x86_64", feature = "cuda")))]
    #[test]
    fn cuda_uncompiled() {
        let bounds = array![[-2.0, 2.0], [-2.0, 2.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);
        let res = pipe.with_backend(bounds, CudaBackend, AffineKey::new(9));
        assert!(res.is_err());
    }

    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    #[test]
    fn metal_backend() {
        use crate::unified_opt::MetalBackend;

        let key = AffineKey::new(2026);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(64)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);

        let (warmup, key) = pipe
            .with_backend(bounds.clone(), MetalBackend, key)
            .unwrap();
        assert_eq!(warmup.device_name(), "metal");
        assert_eq!(warmup.init_progress(), Some((0, 4)));

        let (x_init, warmup, key) = warmup.design_step(4, key).unwrap();
        assert_eq!(x_init.shape(), &[4, 2]);
        let y_init = x_init
            .map_axis(Axis(1), |r| (r[0] - 1.25).powi(2) + (r[1] + 0.75).powi(2))
            .insert_axis(Axis(1));

        let (ready, key) = warmup
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap();
        assert_eq!(ready.num_obs(), 4);
        assert_eq!(ready.init_progress(), None);

        let parent = ready.clone();
        assert!(ready.shares_storage(&parent));
        let parent_digest = parent.digest();

        // Multi-arm (num_arms = 4) and candidates = 64 on live Metal GPU
        let (ready_after, key, traces) = scan_step(ready, key, 3, 4, |_, x| {
            Ok(
                x.map_axis(Axis(1), |r| (r[0] - 1.25).powi(2) + (r[1] + 0.75).powi(2))
                    .insert_axis(Axis(1)),
            )
        })
        .unwrap();
        assert_eq!(traces.len(), 3);
        assert_eq!(ready_after.num_obs(), 16);
        assert!(ready_after.best_y().unwrap().is_finite());
        assert_eq!(parent.digest(), parent_digest);

        // Posterior mean on live Metal GPU
        let mu = ready_after
            .engine()
            .posterior_mu(&array![[1.25, -0.75], [0.0, 0.0]].view())
            .unwrap();
        assert_eq!(mu.shape(), &[2, 1]);

        // Parallel Rayon branching over shared Metal GPU buffers
        let branch_digests = pmap_branches(parent.clone(), key, 3, |idx, opt, bkey| {
            let shift = idx as f64 * 0.5;
            let (fin, bkey, _) = scan_step(opt, bkey, 2, 2, |_, x| {
                Ok(
                    x.map_axis(Axis(1), |r| (r[0] - shift).powi(2) + r[1].powi(2))
                        .insert_axis(Axis(1)),
                )
            })?;
            let _ = bkey.into_key();
            Ok(fin.digest())
        })
        .unwrap();
        assert_eq!(branch_digests.len(), 3);
        assert_ne!(branch_digests[0], branch_digests[1]);
        assert_eq!(parent.digest(), parent_digest);

        // Freeze/thaw roundtrip on MetalBackend
        let frozen = ready_after.freeze().unwrap();
        let thawed = parent.thaw(&frozen).unwrap();
        assert_eq!(thawed.num_obs(), ready_after.num_obs());

        // Runtime AnyBackend::Metal selection
        let (any_opt, _) = pipe
            .with_backend(bounds, AnyBackend::Metal, AffineKey::new(99))
            .unwrap();
        assert_eq!(any_opt.device_name(), "metal");
    }

    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
    #[test]
    fn metal_morbo() {
        use crate::config::{InitStrategy, turbo_enn};
        use crate::morbo::{MorboTRSettings, Rescalarize};
        use crate::optimizer::Strategy;
        use crate::trust_region::{TRLengthConfig, TrustRegionConfig};
        use crate::unified_opt::MetalBackend;

        // 1. Multi-region Metal portfolio with pending_ids + cancel_step +
        //    tell_step
        let bounds = array![[-4.0, 4.0], [-4.0, 4.0]];
        let reg_cfg = turbo_enn();
        let reg_strat =
            Strategy::experimental(2, 3, 4, &mut crate::hash::Key::new(314).prng()).unwrap();
        let reg_eng = UnifiedEngine::new_strategy(
            bounds.clone(),
            reg_cfg,
            reg_strat,
            AnyBackend::Metal,
            crate::hash::Key::new(314),
        )
        .unwrap();
        let reg_warm = TypedOptimizer::from_engine(reg_eng, Minimize, MetalBackend);
        assert_eq!(reg_warm.num_regions(), 3);
        let key = AffineKey::new(314);
        let (x_init, reg_warm, key) = reg_warm.design_step(4, key).unwrap();
        let y_init = x_init
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));
        let (reg_ready, key) = reg_warm
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap();
        let (reg_pend, ticket, key) = reg_ready.ask_step(2, key).unwrap();
        assert_eq!(reg_pend.pending_ids().unwrap().len(), 2);
        let (reg_ready, key) = reg_pend.cancel_step(ticket, key).unwrap();
        assert!(reg_ready.pending_ids().is_none());
        let (reg_pend, ticket, key) = reg_ready.ask_step(2, key).unwrap();
        let y_step = ticket
            .as_array()
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));
        let (reg_done, key) = reg_pend.tell_step(ticket, &y_step.view(), key).unwrap();
        let _ = key.into_key();
        assert_eq!(reg_done.num_obs(), 6);

        // 2. MORBO multi-objective on Metal
        let mut morbo_cfg = turbo_enn();
        morbo_cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        });
        let morbo_eng = UnifiedEngine::new_strategy(
            bounds,
            morbo_cfg,
            Strategy::hybrid(InitStrategy::LHD, 4),
            AnyBackend::Metal,
            crate::hash::Key::new(2718),
        )
        .unwrap();
        let morbo_warm = TypedOptimizer::from_engine(morbo_eng, Maximize, MetalBackend);
        let mkey = AffineKey::new(2718);
        let (mx, morbo_warm, mkey) = morbo_warm.design_step(4, mkey).unwrap();
        let my = Array2::from_shape_fn((4, 2), |(i, j)| {
            if j == 0 {
                -(mx[[i, 0]].powi(2) + mx[[i, 1]].powi(2))
            } else {
                -((mx[[i, 0]] - 1.0).powi(2) + (mx[[i, 1]] - 1.0).powi(2))
            }
        });
        let (morbo_ready, mkey) = morbo_warm
            .activate_ready(&mx.view(), &my.view(), mkey)
            .unwrap();
        let (morbo_pend, mticket, mkey) = morbo_ready.ask_step(2, mkey).unwrap();
        let my_step = Array2::from_shape_fn((2, 2), |(i, j)| {
            let r = mticket.as_array();
            if j == 0 {
                -(r[[i, 0]].powi(2) + r[[i, 1]].powi(2))
            } else {
                -((r[[i, 0]] - 1.0).powi(2) + (r[[i, 1]] - 1.0).powi(2))
            }
        });
        let (morbo_done, mkey) = morbo_pend
            .tell_step(mticket, &my_step.view(), mkey)
            .unwrap();
        let _ = mkey.into_key();
        assert_eq!(morbo_done.num_obs(), 6);
        assert!(morbo_done.engine().best_halves().is_some());

        // 3. Multi-arm regional restart via ask_halves/tell_halves (arms = 4,
        //    fail_tolerance = 1)
        let mut rst_strat =
            Strategy::experimental(2, 2, 4, &mut crate::hash::Key::new(777).prng()).unwrap();
        if let Strategy::Experimental(ref mut st) = rst_strat {
            st.multi_tr.config.fail_tolerance = 1;
        }
        let mut rst_cfg = turbo_enn();
        rst_cfg.trust_region = TrustRegionConfig::Turbo(TRLengthConfig::new(0.8, 0.75, 1.6));
        let mut rst_eng = UnifiedEngine::new_strategy(
            array![[-4.0, 4.0], [-4.0, 4.0]],
            rst_cfg,
            rst_strat,
            AnyBackend::Metal,
            crate::hash::Key::new(777),
        )
        .unwrap();
        let mut rng = crate::hash::Key::new(777).prng();
        let (next_eng, w_rows) = rst_eng.ask_halves(4, &mut rng).unwrap();
        let w_y = array![[10.0], [9.0], [8.0], [7.0]];
        rst_eng = next_eng
            .tell_halves(&w_rows, &w_y.view(), None, &mut rng)
            .unwrap();
        for round in 0..4 {
            let (next_eng, rows) = rst_eng.ask_halves(4, &mut rng).unwrap();
            assert!(!rows.is_empty());
            let bad_y = Array2::from_elem((rows.len(), 1), -100.0 - (round as f64));
            rst_eng = next_eng
                .tell_halves(&rows, &bad_y.view(), None, &mut rng)
                .unwrap();
        }
        assert!(rst_eng.obs_count() >= 8);
    }
}
