//! `BitOr` (`|`) compile-time pipeline combinators for neighbor-surrogate
//! optimization.

use core::ops::BitOr;

use ndarray::Array2;
use rand::RngCore;

use crate::candidates::CandidateSampler;
use crate::config::{
    AcquisitionConfig, CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig,
};
use crate::error::ENNError;
use crate::optimizer::{BayesOptimizer, Strategy};
use crate::sense::{Maximize, ObjectiveSense};
use crate::surrogate::ENNSurrogateConfig;
pub use crate::typestate::{
    Active, Fitted, ProposalTicket, TypedOptimizer, TypedSurrogate, Unfitted, Warmup,
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
    /// Compile this pipeline into a typestate [`TypedOptimizer<O, Warmup>`].
    pub fn build(
        self,
        bounds: Array2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<TypedOptimizer<O, Warmup>, ENNError> {
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
        let opt = BayesOptimizer::new_strategy(bounds, config, strategy, rng)?;
        Ok(TypedOptimizer::new(opt, self.acq.sense))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sense::Minimize;
    use crate::surrogate_api::ObservationView;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn pipeline_ticket() {
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = array![[-5.0, 5.0], [-5.0, 5.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(32)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.5, Minimize);

        let mut warmup_opt = pipe.build(bounds, &mut rng).unwrap();
        let x_init = warmup_opt.warmup_design(4, &mut rng).unwrap();
        let y_init = x_init.map_axis(ndarray::Axis(1), |row| row[0].powi(2) + row[1].powi(2));
        let y_init = y_init.insert_axis(ndarray::Axis(1));

        let mut active_opt = warmup_opt
            .activate(&x_init.view(), &y_init.view(), &mut rng)
            .unwrap();
        let ticket = active_opt.ask_ticket(2, &mut rng).unwrap();
        assert_eq!(ticket.num_arms(), 2);
        let y_step = ticket
            .as_array()
            .map_axis(ndarray::Axis(1), |row| row[0].powi(2) + row[1].powi(2))
            .insert_axis(ndarray::Axis(1));
        active_opt
            .tell_ticket(ticket, &y_step.view(), &mut rng)
            .unwrap();
        assert!(active_opt.best_y().unwrap() >= 0.0);
    }

    #[test]
    fn typestate_fit() {
        let mut rng = StdRng::seed_from_u64(19);
        let unfitted = TypedSurrogate::new(ENNSurrogateConfig {
            k: 2,
            num_candidates: 4,
            num_samples: 2,
            ..Default::default()
        });
        let x = array![[0.1, 0.2], [0.4, 0.5], [0.8, 0.9]];
        let y = array![[1.0], [2.0], [3.0]];
        let fitted = unfitted.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        assert_eq!(fitted.obs_len(), 3);
        let pred = fitted.predict(&array![[0.2, 0.3]].view()).unwrap();
        assert_eq!(pred.mu.shape(), &[1, 1]);
    }
}
