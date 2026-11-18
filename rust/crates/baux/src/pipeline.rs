//! `BitOr` (`|`) compile-time pipeline combinators for neighbor-surrogate
//! optimization.

use core::ops::BitOr;

use ndarray::Array2;
use rand::RngCore;

use crate::candidates::CandidateSampler;
use crate::config::{
    AcquisitionConfig, CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig,
};
use crate::coordinate::DomainFrame;
use crate::error::ENNError;
use crate::hash::AffineKey;
use crate::optimizer::Strategy;
use crate::sense::{Maximize, ObjectiveSense};
use crate::surrogate::ENNSurrogateConfig;
pub use crate::typestate::{
    Active, Fitted, Pending, ProposalTicket, Ready, StepTrace, TypedOptimizer, TypedSurrogate,
    Unfitted, Warmup, pmap_branches, scan_step,
};
pub use crate::unified_opt::{
    AnyBackend, CpuBackend, CudaBackend, MetalBackend, OptBackend, UnifiedEngine,
};

/// Stage 1 combinator: initial space-filling design.
#[derive(Clone, Copy, Debug)]
pub enum InitStage {
    Lhd(usize),
    Random(usize),
}

/// Stage 2 combinator: trust-region candidate perturbation kernel.
#[derive(Clone, Copy, Debug)]
pub enum PerturbStage {
    Raasp(usize),
    Sobol(usize),
    Uniform(usize),
}

/// Stage 3 combinator: surrogate model family.
#[derive(Clone, Copy, Debug)]
pub enum ModelStage {
    Enn(i32),
    Zero,
}

/// Stage 4 combinator: acquisition rule paired with an [`ObjectiveSense`].
#[derive(Clone, Copy, Debug)]
pub struct AcqStage<O: ObjectiveSense = Maximize> {
    pub config: AcquisitionConfig,
    pub sense: O,
}

impl AcqStage<Maximize> {
    pub const fn ucb(beta: f64) -> Self {
        Self {
            config: AcquisitionConfig::UCB { beta },
            sense: Maximize,
        }
    }

    pub const fn thompson() -> Self {
        Self {
            config: AcquisitionConfig::Thompson,
            sense: Maximize,
        }
    }
}

impl<O: ObjectiveSense> AcqStage<O> {
    pub const fn ucb_sense(beta: f64, sense: O) -> Self {
        Self {
            config: AcquisitionConfig::UCB { beta },
            sense,
        }
    }
}

/// Intermediate pipeline: `InitStage | PerturbStage`.
#[derive(Clone, Copy, Debug)]
pub struct DesignPipe {
    pub init: InitStage,
    pub perturb: PerturbStage,
}

/// Intermediate pipeline: `InitStage | PerturbStage | ModelStage`.
#[derive(Clone, Copy, Debug)]
pub struct SurrogatePipe {
    pub design: DesignPipe,
    pub model: ModelStage,
}

/// Complete type-checked optimization pipeline:
/// `InitStage | PerturbStage | ModelStage | AcqStage<O>`.
#[derive(Clone, Copy, Debug)]
pub struct Pipeline<O: ObjectiveSense = Maximize> {
    pub surrogate: SurrogatePipe,
    pub acq: AcqStage<O>,
}

impl BitOr<PerturbStage> for InitStage {
    type Output = DesignPipe;

    fn bitor(self, rhs: PerturbStage) -> Self::Output {
        DesignPipe {
            init: self,
            perturb: rhs,
        }
    }
}

impl BitOr<ModelStage> for DesignPipe {
    type Output = SurrogatePipe;

    fn bitor(self, rhs: ModelStage) -> Self::Output {
        SurrogatePipe {
            design: self,
            model: rhs,
        }
    }
}

impl<O: ObjectiveSense> BitOr<AcqStage<O>> for SurrogatePipe {
    type Output = Pipeline<O>;

    fn bitor(self, rhs: AcqStage<O>) -> Self::Output {
        Pipeline {
            surrogate: self,
            acq: rhs,
        }
    }
}

impl<O: ObjectiveSense> Pipeline<O> {
    fn into_parts(self) -> (OptimizerConfig, Strategy, O) {
        let (init_kind, num_init) = match self.surrogate.design.init {
            InitStage::Lhd(n) => (InitStrategy::LHD, n),
            InitStage::Random(n) => (InitStrategy::Random, n),
        };
        let (candidate_rv, num_candidates) = match self.surrogate.design.perturb {
            PerturbStage::Raasp(n) => (CandidateSampler::RAASP, n),
            PerturbStage::Sobol(n) => (CandidateSampler::Sobol, n),
            PerturbStage::Uniform(n) => (CandidateSampler::Uniform, n),
        };
        let surrogate = match self.surrogate.model {
            ModelStage::Enn(k) => SurrogateConfig::ENN(ENNSurrogateConfig {
                k,
                ..Default::default()
            }),
            ModelStage::Zero => SurrogateConfig::None,
        };
        let config = OptimizerConfig {
            surrogate,
            acquisition: self.acq.config,
            candidates: CandidateConfig {
                candidate_rv,
                min_candidates: num_candidates,
                max_candidates: Some(num_candidates),
                ..Default::default()
            },
            ..Default::default()
        };
        let strategy = Strategy::hybrid(init_kind, num_init);
        (config, strategy, self.acq.sense)
    }

    /// Pure functional compilation of this pipeline onto `backend`, consuming
    /// `key` and returning `(warmup_opt, next_key)`.
    pub fn with_backend<B: OptBackend>(
        self,
        bounds: Array2<f64>,
        backend: B,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Warmup, B>, AffineKey), ENNError> {
        let (config, strategy, sense) = self.into_parts();
        let (draw_key, next_key) = key.step();
        let engine = backend.build_engine(DomainFrame::new(bounds)?, config, strategy, draw_key)?;
        Ok((
            TypedOptimizer::from_engine(engine, sense, backend),
            next_key,
        ))
    }

    /// Pure functional compilation of this pipeline onto [`CpuBackend`],
    /// consuming `key` and returning `(warmup_opt, next_key)`.
    pub fn build_key(
        self,
        bounds: Array2<f64>,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Warmup>, AffineKey), ENNError> {
        self.with_backend(bounds, CpuBackend, key)
    }

    /// Compile this pipeline into a typestate [`TypedOptimizer<O, Warmup>`].
    pub fn build(
        self,
        bounds: Array2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<TypedOptimizer<O, Warmup>, ENNError> {
        let (opt, _) = self.build_key(bounds, AffineKey::new(rng.next_u64()))?;
        Ok(opt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::Minimize;
    use crate::surrogate_api::ObservationView;
    use ndarray::array;

    #[test]
    fn pipeline_ticket() {
        let key = AffineKey::new(42);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(32)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);

        let (warmup_opt, key) = pipe.build_key(bounds, key).unwrap();
        let (x_init, warmup_opt, key) = warmup_opt.design_step(4, key).unwrap();
        let y_init = x_init.map_axis(ndarray::Axis(1), |row| row[0].powi(2) + row[1].powi(2));
        let y_init = y_init.insert_axis(ndarray::Axis(1));

        let (ready_opt, key) = warmup_opt
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap();
        let (pending_opt, ticket, key) = ready_opt.ask_step(2, key).unwrap();
        assert_eq!(ticket.num_arms(), 2);
        let y_step = ticket
            .as_array()
            .map_axis(ndarray::Axis(1), |row| row[0].powi(2) + row[1].powi(2))
            .insert_axis(ndarray::Axis(1));
        let (ready_opt, key) = pending_opt.tell_step(ticket, &y_step.view(), key).unwrap();
        let _ = key.into_key();
        assert!(ready_opt.best_y().unwrap() >= 0.0);
    }

    #[test]
    fn typestate_fit() {
        let key = AffineKey::new(19);
        let unfitted = TypedSurrogate::new(ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        });
        let x = array![[0.1, 0.2], [0.4, 0.5], [0.8, 0.9]];
        let y = array![[1.0], [2.0], [3.0]];
        let (fitted, key) = unfitted.fit_key(&x.view(), &y.view(), None, key).unwrap();
        let _ = key.into_key();
        let cloned = fitted.clone();
        assert_eq!(cloned.obs_len(), 3);
        let pred = cloned.predict(&array![[0.2, 0.3]].view()).unwrap();
        assert_eq!(pred.mu.shape(), &[1, 1]);
    }

    #[test]
    fn pure_branching() {
        let key = AffineKey::new(42);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(32)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);
        let (warmup, key) = pipe.build_key(bounds, key).unwrap();
        let (x_init, warmup, key) = warmup.design_step(4, key).unwrap();
        let y_init = array![[1.0], [2.0], [0.5], [3.0]];
        let (branch_a, key) = warmup
            .activate_step(&x_init.view(), &y_init.view(), key)
            .unwrap();
        let _ = key.into_key();
        let branch_b = branch_a.fork().unwrap();

        let step_key = crate::hash::Key::new(101);
        let (next_a, ticket_a) = branch_a.ask_pure(2, step_key).unwrap();
        let (next_b, ticket_b) = branch_b.ask_pure(2, step_key).unwrap();
        assert_eq!(ticket_a.as_array(), ticket_b.as_array());
        assert_eq!(next_a.digest(), next_b.digest());

        let y_step = array![[0.25], [0.75]];
        let advanced_a = next_a
            .tell_pure(ticket_a, &y_step.view(), step_key)
            .unwrap();
        assert!(
            advanced_a
                .tell_pure(ticket_b, &y_step.view(), step_key)
                .is_err()
        );
    }
}
