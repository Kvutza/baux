use baux::semantics::random::sobol_seed;
use ndarray::{Array1, Array2};
use rand::RngCore;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::BayesOptimizer;
use super::obs_access::PosteriorStore;
use super::tr_state::TrustRegionState;
use crate::candidates::{CandidateSampler, SobolEngine};
use crate::config::{CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::incumbent_tracker::{IncumbentTracker, enn_k, tracker_surrogate};
use crate::util::argmax_tie;

/// Encapsulated candidate proposal generator and quasi-random sampler state.
pub struct ProposalEngine {
    config: CandidateConfig,
    sobol_engine: Option<SobolEngine>,
    sobol_base: u64,
}

impl ProposalEngine {
    pub fn new(
        num_dim: usize,
        config: CandidateConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let sobol_engine = if config.candidate_rv == CandidateSampler::Sobol {
            let mut eng = SobolEngine::new(num_dim)?;
            eng.scramble(rng);
            Some(eng)
        } else {
            None
        };
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let sobol_base = u64::from_le_bytes(seed_bytes) % (1u64 << 31);
        Ok(Self {
            config,
            sobol_engine,
            sobol_base,
        })
    }

    pub fn config(&self) -> &CandidateConfig {
        &self.config
    }

    pub fn sobol_mut(&mut self) -> Option<&mut SobolEngine> {
        self.sobol_engine.as_mut()
    }

    pub fn sobol_base(&self) -> u64 {
        self.sobol_base
    }

    pub fn reseed_sobol(
        &mut self,
        generation: usize,
        obs_count: usize,
        arms: usize,
    ) -> Result<(), ENNError> {
        let Some(engine) = self.sobol_engine.as_ref() else {
            return Ok(());
        };
        let dimension = engine.dimension();
        let seed = sobol_seed(self.sobol_base, generation, obs_count, arms);
        let mut next = SobolEngine::new(dimension)?;
        let mut rng = StdRng::seed_from_u64(seed);
        next.scramble(&mut rng);
        self.sobol_engine = Some(next);
        Ok(())
    }
}

/// Active incumbent snapshot bundling index, unit coordinates, and scalarized
/// objective.
#[derive(Debug, Clone)]
pub struct IncumbentPoint {
    pub index: usize,
    pub unit_x: Array1<f64>,
    pub y_scalar: Array1<f64>,
}

/// Unified incumbent tracker and active incumbent point ledger.
pub struct IncumbentLedger {
    pub(crate) tracker: IncumbentTracker,
    current: Option<IncumbentPoint>,
}

impl IncumbentLedger {
    pub fn new(config: &OptimizerConfig, num_metrics: usize) -> Self {
        let tracker_m = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => enn_k(enn_config.k),
            SurrogateConfig::None => tracker_surrogate(),
        };
        let tracker = IncumbentTracker::new(tracker_m, config.noise_aware, num_metrics);
        Self {
            tracker,
            current: None,
        }
    }

    pub fn reset(&mut self) {
        self.tracker.reset();
        self.current = None;
    }

    pub fn tell_row(&mut self, index: usize, y_row: &Array1<f64>) {
        self.tracker.tell(index, y_row);
    }

    pub fn unit_x(&self) -> Option<&Array1<f64>> {
        self.current.as_ref().map(|pt| &pt.unit_x)
    }

    pub fn y_scalar(&self) -> Option<&Array1<f64>> {
        self.current.as_ref().map(|pt| &pt.y_scalar)
    }

    pub fn observation_count(&self) -> usize {
        self.tracker.observation_count()
    }

    pub fn update(
        &mut self,
        posterior: &PosteriorStore,
        regions: &TrustRegionState,
        num_dim: usize,
        noise_aware: bool,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let obs_count = posterior.obs_count();
        if obs_count == 0 {
            self.current = None;
            return Ok(());
        }
        if self.tracker.observation_count() != obs_count {
            if let Some(y_obs) = posterior.y_obs() {
                self.tracker.rebuild(&y_obs.view());
            }
        }
        let candidates = self.tracker.ask();
        if candidates.is_empty() {
            self.current = None;
            return Ok(());
        }
        if regions.is_morbo() {
            return self.pick_morbo(&candidates, posterior, regions, num_dim, noise_aware, rng);
        }
        let best_idx = candidates
            .into_iter()
            .max_by(|&a, &b| compare_rows(posterior, a, b))
            .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?;
        self.current = Some(IncumbentPoint {
            index: best_idx,
            unit_x: posterior.x_row(best_idx)?,
            y_scalar: posterior.y_row(best_idx)?,
        });
        Ok(())
    }

    fn pick_morbo(
        &mut self,
        candidates: &[usize],
        posterior: &PosteriorStore,
        regions: &TrustRegionState,
        num_dim: usize,
        noise_aware: bool,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let n_cand = candidates.len();
        let mut y_rows = Array2::zeros((n_cand, regions.num_metrics()));
        for (r, &idx) in candidates.iter().enumerate() {
            let y_row = posterior.y_row(idx)?;
            for m in 0..y_rows.ncols() {
                y_rows[[r, m]] = y_row[m];
            }
        }
        if noise_aware && let Some(surrogate) = posterior.surrogate() {
            let mut x_cand = Array2::zeros((n_cand, num_dim));
            for (r, &idx) in candidates.iter().enumerate() {
                let x_row = posterior.x_row(idx)?;
                for d in 0..num_dim {
                    x_cand[[r, d]] = x_row[d];
                }
            }
            y_rows = surrogate.predict(&x_cand.view())?.mu;
        }
        let scores = regions
            .morbo()
            .ok_or_else(|| ENNError::InvalidParameter("Morbo incumbent without Morbo".to_string()))?
            .scalarize_local(&y_rows.view())
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let best_pos = argmax_tie(scores.as_slice().unwrap_or(&[]), rng);
        let best_idx = candidates[best_pos];
        self.current = Some(IncumbentPoint {
            index: best_idx,
            unit_x: posterior.x_row(best_idx)?,
            y_scalar: y_rows.row(best_pos).to_owned(),
        });
        Ok(())
    }
}

fn compare_rows(posterior: &PosteriorStore, a: usize, b: usize) -> std::cmp::Ordering {
    posterior
        .y_row(a)
        .and_then(|ya| posterior.y_row(b).map(|yb| ya[0].total_cmp(&yb[0])))
        .unwrap_or(std::cmp::Ordering::Equal)
}

impl BayesOptimizer {
    /// Increment restart generation.
    pub fn increment_generation(&mut self) {
        self.warmup.generation += 1;
    }

    /// Get restart generation.
    pub fn restart_generation(&self) -> usize {
        self.warmup.generation
    }

    pub(crate) fn reinit_left(&self) -> Option<usize> {
        self.warmup.reinit_left
    }

    pub(crate) fn init_kind(&self) -> InitStrategy {
        self.warmup.kind
    }

    pub(crate) fn consume_reinit(&mut self, count: usize) {
        self.warmup.consume_reinit(count);
    }

    pub(crate) fn restart_local(&mut self) -> Result<(), ENNError> {
        if self.regions.is_morbo() {
            return Ok(());
        }
        self.posterior.clear()?;
        self.incumbent.reset();
        self.regions.restart_local();
        self.warmup.restart_local();
        Ok(())
    }

    /// Get sobol engine.
    pub fn sobol_mut(&mut self) -> Option<&mut SobolEngine> {
        self.proposals.sobol_mut()
    }

    /// Get sobol seed base.
    pub fn sobol_base(&self) -> u64 {
        self.proposals.sobol_base()
    }

    /// Replace the persistent engine with the source's per-ask scrambled Sobol
    /// engine for MORBO.
    pub(crate) fn reseed_sobol(&mut self, arms: usize) -> Result<(), ENNError> {
        let generation = self.warmup.generation;
        let obs_count = self.posterior.obs_count();
        self.proposals.reseed_sobol(generation, obs_count, arms)
    }

    /// Get init progress from warmup schedule.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        self.warmup.progress()
    }

    /// Current number of stored observations.
    pub fn obs_count(&self) -> usize {
        self.posterior.obs_count()
    }

    #[allow(dead_code)] // retained for tests / morbo callers; tell_turbo must not use it
    pub(crate) fn reset_tracker(&mut self) {
        self.incumbent.reset();
    }

    pub fn update_incumbent(&mut self, rng: &mut dyn RngCore) -> Result<(), ENNError> {
        self.incumbent.update(
            &self.posterior,
            &self.regions,
            self.domain.num_dim(),
            self.config.noise_aware,
            rng,
        )
    }
}
