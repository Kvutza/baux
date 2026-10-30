use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use super::{BayesOptimizer, ObservationDelta};
use crate::config::SurrogateConfig;
use crate::error::ENNError;
use crate::optimizer_dispatch::{VarianceGate, variance_gate};
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};

/// Closed storage source: either empirical observation vectors or an indexed
/// surrogate model.
pub enum PosteriorSource {
    Empirical {
        x_rows: Vec<Array1<f64>>,
        y_rows: Vec<Array1<f64>>,
    },
    Model(BoxedSurrogate),
}

/// Unified posterior and observation store for the ask/tell optimizer.
pub struct PosteriorStore {
    source: PosteriorSource,
    pub(crate) yvar_mode: Option<bool>,
}

impl PosteriorStore {
    pub fn new(config: &SurrogateConfig, provided: Option<BoxedSurrogate>) -> Self {
        let model = provided.or_else(|| match config {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        });
        let source = match model {
            Some(surrogate) => PosteriorSource::Model(surrogate),
            None => PosteriorSource::Empirical {
                x_rows: Vec::new(),
                y_rows: Vec::new(),
            },
        };
        Self {
            source,
            yvar_mode: None,
        }
    }

    pub fn has_surrogate(&self) -> bool {
        matches!(self.source, PosteriorSource::Model(_))
    }

    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        match &self.source {
            PosteriorSource::Model(s) => Some(s.as_ref()),
            PosteriorSource::Empirical { .. } => None,
        }
    }

    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match &mut self.source {
            PosteriorSource::Model(s) => Some(s.as_mut()),
            PosteriorSource::Empirical { .. } => None,
        }
    }

    pub fn check_variance(&mut self, has_yvar: bool) -> Result<(), ENNError> {
        match variance_gate(self.yvar_mode, has_yvar) {
            VarianceGate::Set(value) => self.yvar_mode = Some(value),
            VarianceGate::Reject => {
                return Err(ENNError::InvalidParameter(format!(
                    "y_var must be {} on every tell()",
                    if self.yvar_mode == Some(true) {
                        "provided"
                    } else {
                        "omitted"
                    }
                )));
            }
            VarianceGate::Keep => {}
        }
        Ok(())
    }

    pub fn obs_count(&self) -> usize {
        match &self.source {
            PosteriorSource::Model(s) => s.observation_count().unwrap_or(0),
            PosteriorSource::Empirical { x_rows, .. } => x_rows.len(),
        }
    }

    pub fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.source {
            PosteriorSource::Model(s) => s.x_row(idx),
            PosteriorSource::Empirical { x_rows, .. } => {
                x_rows.get(idx).cloned().ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })
            }
        }
    }

    pub fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        match &self.source {
            PosteriorSource::Model(s) => s.y_row(idx),
            PosteriorSource::Empirical { y_rows, .. } => {
                y_rows.get(idx).cloned().ok_or_else(|| {
                    ENNError::InvalidParameter(format!("observation index {idx} out of range"))
                })
            }
        }
    }

    pub fn unit_obs(&self) -> Option<Array2<f64>> {
        match &self.source {
            PosteriorSource::Model(s) => s.observations_x().ok().flatten(),
            PosteriorSource::Empirical { x_rows, .. } => {
                (!x_rows.is_empty()).then(|| stack_rows(x_rows))
            }
        }
    }

    pub fn y_obs(&self) -> Option<Array2<f64>> {
        match &self.source {
            PosteriorSource::Model(s) => s.observations_y().ok().flatten(),
            PosteriorSource::Empirical { y_rows, .. } => {
                (!y_rows.is_empty()).then(|| stack_rows(y_rows))
            }
        }
    }

    pub fn push_empirical(&mut self, x_row: Array1<f64>, y_row: Array1<f64>) {
        if let PosteriorSource::Empirical { x_rows, y_rows } = &mut self.source {
            x_rows.push(x_row);
            y_rows.push(y_row);
        }
    }

    pub fn fit_delta(
        &mut self,
        delta: &ObservationDelta,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Option<f64>, ENNError> {
        let ncols = delta.y_new.ncols();
        if let Some(nm) = self.surrogate().and_then(|s| s.fitted_metrics()) {
            if nm != ncols {
                return Err(ENNError::InvalidParameter(format!(
                    "y has {ncols} metric columns but the fitted model has {nm}; changing output \
                     width is unsupported"
                )));
            }
        }
        if let PosteriorSource::Model(surrogate) = &mut self.source {
            let start = std::time::Instant::now();
            surrogate.fit_append(&delta.x_view(), &delta.y_view(), yvar, rng)?;
            return Ok(Some(start.elapsed().as_secs_f64()));
        }
        Ok(None)
    }

    pub fn clear(&mut self) -> Result<(), ENNError> {
        match &mut self.source {
            PosteriorSource::Model(s) => s.clear_observations(),
            PosteriorSource::Empirical { x_rows, y_rows } => {
                x_rows.clear();
                y_rows.clear();
                Ok(())
            }
        }
    }

    pub fn schedule_flush(&self) -> Result<(), ENNError> {
        if let PosteriorSource::Model(s) = &self.source {
            s.schedule_flush()?;
        }
        Ok(())
    }

    pub fn wait_flush(&self) -> Result<(), ENNError> {
        if let PosteriorSource::Model(s) = &self.source {
            s.wait_flush()?;
        }
        Ok(())
    }

    pub fn release_pages(&self) -> Result<(), ENNError> {
        if let PosteriorSource::Model(s) = &self.source {
            s.release_pages()?;
        }
        Ok(())
    }
}

/// Public accessor for optimizer observation row reads.
pub struct ObsAccess<'a> {
    opt: &'a BayesOptimizer,
}

impl<'a> ObsAccess<'a> {
    pub(crate) fn new(opt: &'a BayesOptimizer) -> Self {
        Self { opt }
    }

    pub fn observations_empty(&self) -> bool {
        self.opt.posterior.obs_count() == 0
    }

    pub fn x_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        self.opt.posterior.x_row(idx)
    }

    pub fn y_row(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        self.opt.posterior.y_row(idx)
    }
}

pub fn stack_rows(vecs: &[Array1<f64>]) -> Array2<f64> {
    let n = vecs.len();
    let d = vecs[0].len();
    let mut result = Array2::zeros((n, d));
    for (i, v) in vecs.iter().enumerate() {
        for j in 0..d {
            result[[i, j]] = v[j];
        }
    }
    result
}
