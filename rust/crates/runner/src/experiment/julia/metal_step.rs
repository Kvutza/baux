//! Unified cross-backend step-by-step and multi-arm batch execution over
//! `UnifiedEngine` / `TypedOptimizer`.

use baux::artifact::inc::{EvalOutcome, StepRecord};
use baux::config::{ObjectivePolicy, OptimizerSpec};
use baux::optimizer::DomainFrame;
use baux::{
    AffineKey, AnyBackend, Maximize, OptBackend, OptimizerConfig, TypedOptimizer, UnifiedEngine,
    Warmup,
};
use ndarray::Array2;

use super::metal::{lhd_batch, round_seed};
use crate::experiment::tokamak::label::round_rng;

pub fn unit_domain(num_dim: usize) -> Result<DomainFrame, String> {
    if num_dim == 0 {
        return Err("Invalid parameter: dimension must be > 0, got 0".into());
    }
    let bounds = Array2::from_shape_fn((num_dim, 2), |(_, j)| if j == 0 { 0.0 } else { 1.0 });
    DomainFrame::new(bounds).map_err(|e| e.to_string())
}

pub fn typed_warmup<B: OptBackend>(
    domain: DomainFrame,
    spec: &OptimizerSpec,
    cfg_opt: OptimizerConfig,
    policy: Option<ObjectivePolicy>,
    backend: B,
    key: AffineKey,
) -> Result<(TypedOptimizer<Maximize, Warmup, B>, AffineKey), String> {
    TypedOptimizer::from_spec(domain, cfg_opt, spec, policy, 0, Maximize, backend, key)
        .map_err(|e| e.to_string())
}

#[derive(Clone)]
pub struct MetalJobSpec {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub n_obj: usize,
    pub rounds: usize,
    pub arms: usize,
    pub prototype: UnifiedEngine,
}

impl MetalJobSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn with_backend(
        rep: u32,
        seed: u64,
        domain: DomainFrame,
        n_obj: usize,
        rounds: usize,
        spec: &OptimizerSpec,
        cfg_opt: OptimizerConfig,
        policy: Option<ObjectivePolicy>,
        backend: AnyBackend,
    ) -> Result<Self, String> {
        let num_dim = domain.num_dim();
        let arms = spec
            .candidate_generation()
            .map_or(1, |cg| cg.arms.max(1) as usize);
        let (opt, k_rem) =
            typed_warmup(domain, spec, cfg_opt, policy, backend, AffineKey::new(seed))?;
        let _ = k_rem.into_key();
        Ok(Self {
            rep,
            seed,
            num_dim,
            n_obj: n_obj.max(1),
            rounds,
            arms,
            prototype: opt.into_engine(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        rep: u32,
        seed: u64,
        num_dim: usize,
        n_obj: usize,
        rounds: usize,
        spec: &OptimizerSpec,
        cfg_opt: OptimizerConfig,
        policy: Option<ObjectivePolicy>,
    ) -> Result<Self, String> {
        let domain = unit_domain(num_dim)?;
        Self::with_backend(
            rep,
            seed,
            domain,
            n_obj,
            rounds,
            spec,
            cfg_opt,
            policy,
            AnyBackend::Metal,
        )
    }

    #[inline]
    pub fn arms(&self) -> usize {
        self.arms.max(1)
    }
}

pub struct MetalFiber {
    pub(crate) spec: MetalJobSpec,
    pub(crate) engine: Option<UnifiedEngine>,
    pub(crate) best_row: Vec<u16>,
    pub(crate) best_unit: Vec<f64>,
    pub(crate) round: usize,
    pub(crate) best_score: f64,
    pub(crate) best_viol: f64,
    pub(crate) initial_error: f64,
    pub(crate) total_accepts: u32,
    pub(crate) worker_s: f64,
    pending_rows: Option<Vec<Vec<u16>>>,
    pending_unit: Option<Array2<f64>>,
}

impl MetalFiber {
    pub fn new(spec: MetalJobSpec) -> Self {
        let init_row = lhd_batch(round_seed(spec.seed, 1), 1, spec.num_dim)
            .into_iter()
            .next()
            .unwrap_or_default();
        let best_unit = vec![0.5f64; spec.num_dim];
        let engine = Some(spec.prototype.clone());
        Self {
            spec,
            engine,
            best_row: init_row,
            best_unit,
            round: 0,
            best_score: f64::NEG_INFINITY,
            best_viol: f64::INFINITY,
            initial_error: 0.0,
            total_accepts: 0,
            worker_s: 0.0,
            pending_rows: None,
            pending_unit: None,
        }
    }

    #[inline]
    pub fn device_name(&self) -> &'static str {
        self.engine
            .as_ref()
            .map_or("unknown", UnifiedEngine::device_name)
    }

    fn prepare_n(&mut self, max_arms: usize) -> Result<Vec<Vec<u16>>, String> {
        if self.pending_rows.is_some() || self.pending_unit.is_some() {
            return Err("proposal batch already pending".into());
        }
        let eng = self.engine.take().ok_or("optimizer state missing")?;
        let want = match eng.init_progress() {
            Some((done, goal)) if goal > done => (goal - done).min(max_arms).max(1),
            _ => max_arms.max(1),
        };
        let mut rng = round_rng(self.spec.seed, self.round + 1);
        let (next_eng, rows) = eng.ask_halves(want, &mut rng).map_err(|e| e.to_string())?;
        self.engine = Some(next_eng);
        self.pending_rows = Some(rows.clone());
        Ok(rows)
    }

    pub fn prepare_batch(&mut self) -> Result<Vec<Vec<u16>>, String> {
        self.prepare_n(self.spec.arms())
    }

    pub fn prepare_proposal(&mut self) -> Result<Vec<u16>, String> {
        let mut rows = self.prepare_n(1)?;
        rows.pop().ok_or_else(|| "empty proposal batch".into())
    }

    pub fn prepare_unit(&mut self) -> Result<Array2<f64>, String> {
        if self.pending_rows.is_some() || self.pending_unit.is_some() {
            return Err("proposal batch already pending".into());
        }
        let eng = self.engine.take().ok_or("optimizer state missing")?;
        let mut rng = round_rng(self.spec.seed, self.round + 1);
        let (next_eng, unit) = eng
            .ask(self.spec.arms(), &mut rng)
            .map_err(|e| e.to_string())?;
        self.engine = Some(next_eng);
        self.pending_unit = Some(unit.clone());
        Ok(unit)
    }

    fn record_step(
        &mut self,
        batch_len: usize,
        scores: &[f64],
        viols: &[f64],
        cost_us: u64,
        loss_fn: &dyn Fn(f64) -> f64,
        mut on_win: impl FnMut(usize, &mut Self),
    ) -> StepRecord {
        let n_obj = self.spec.n_obj;
        self.round += 1;
        let mut accepted = false;
        let mut round_viol = viols.first().copied().unwrap_or(0.0);
        for i in 0..batch_len {
            let sc = crate::experiment::julia::driver::scalar_score(scores, i, n_obj);
            let vl = viols.get(i).copied().unwrap_or(round_viol);
            if sc > self.best_score {
                self.best_score = sc;
                self.best_viol = vl;
                on_win(i, self);
                round_viol = vl;
                accepted = true;
            }
        }
        if accepted {
            self.total_accepts += 1;
        }
        if self.round == 1 {
            self.initial_error = loss_fn(self.best_score);
        }
        self.worker_s += (cost_us as f64) * 1.0e-6;
        StepRecord {
            eval_scores: scores.to_vec(),
            eval_viol: round_viol,
            best_score: self.best_score,
            best_viol: self.best_viol,
            accepted,
            cost_us,
        }
    }

    pub fn commit_batch(
        &mut self,
        scores: &[f64],
        viols: &[f64],
        cost_us: u64,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        let rows = self.pending_rows.take().ok_or("no proposal to commit")?;
        let eng = self.engine.take().ok_or("optimizer state missing")?;
        let n = rows.len();
        let n_obj = self.spec.n_obj;
        let mut rng = round_rng(self.spec.seed, self.round + 1);
        let y = Array2::from_shape_fn((n, n_obj), |(i, m)| scores[i * n_obj + m]);
        self.engine = Some(
            eng.tell_halves(&rows, &y.view(), None, &mut rng)
                .map_err(|e| e.to_string())?,
        );
        Ok(
            self.record_step(n, scores, viols, cost_us, loss_fn, |i, s| {
                s.best_row.copy_from_slice(&rows[i]);
            }),
        )
    }

    pub fn commit_unit(
        &mut self,
        scores: &[f64],
        viols: &[f64],
        cost_us: u64,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        let unit = self
            .pending_unit
            .take()
            .ok_or("no unit proposal to commit")?;
        let eng = self.engine.take().ok_or("optimizer state missing")?;
        let n = unit.nrows();
        let n_obj = self.spec.n_obj;
        let mut rng = round_rng(self.spec.seed, self.round + 1);
        let y = Array2::from_shape_fn((n, n_obj), |(i, m)| scores[i * n_obj + m]);
        self.engine = Some(
            eng.tell(&unit.view(), &y.view(), None, &mut rng)
                .map_err(|e| e.to_string())?,
        );
        let num_dim = self.spec.num_dim;
        Ok(
            self.record_step(n, scores, viols, cost_us, loss_fn, |i, s| {
                if let Some(sl) = unit.as_slice() {
                    s.best_unit
                        .copy_from_slice(&sl[i * num_dim..(i + 1) * num_dim]);
                } else {
                    for (j, &v) in unit.row(i).iter().enumerate() {
                        s.best_unit[j] = v;
                    }
                }
            }),
        )
    }

    pub fn commit_eval(
        &mut self,
        outcome: &EvalOutcome,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        self.commit_batch(&outcome.scores, &[outcome.viol], outcome.cost_us, loss_fn)
    }

    pub fn replay_prefix(
        &mut self,
        steps: &[StepRecord],
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<(), String> {
        for step in steps {
            let _ = self.prepare_proposal()?;
            let outcome = EvalOutcome {
                scores: step.eval_scores.clone(),
                viol: step.eval_viol,
                cost_us: step.cost_us,
            };
            let _ = self.commit_eval(&outcome, loss_fn)?;
        }
        Ok(())
    }
}
