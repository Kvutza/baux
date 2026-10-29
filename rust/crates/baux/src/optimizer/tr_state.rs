use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;
use crate::morbo::{MorboTrustRegion, Rescalarize};
use crate::optimizer::multi_tr::MultiTrustRegionState;
use crate::trust_region::{TrustRegionConfig, TrustRegionError, TurboTrustRegion};

#[derive(Debug)]
pub enum TrustRegionState {
    Turbo(TurboTrustRegion),
    Morbo(Box<MorboTrustRegion>),
    Multi {
        state: MultiTrustRegionState,
        pending: Option<Vec<usize>>,
    },
}

impl TrustRegionState {
    pub fn from_config(
        num_dim: usize,
        config: &TrustRegionConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        match config {
            TrustRegionConfig::Turbo(cfg) => Ok(TrustRegionState::Turbo(TurboTrustRegion::new(
                num_dim, *cfg,
            ))),
            TrustRegionConfig::Morbo(settings) => Ok(TrustRegionState::Morbo(Box::new(
                MorboTrustRegion::new(num_dim, settings.clone(), rng)?,
            ))),
        }
    }

    pub fn multi(state: MultiTrustRegionState, pending: Option<Vec<usize>>) -> Self {
        Self::Multi { state, pending }
    }

    pub fn is_morbo(&self) -> bool {
        matches!(self, TrustRegionState::Morbo(_))
    }

    pub fn is_multi(&self) -> bool {
        matches!(self, TrustRegionState::Multi { .. })
    }

    pub fn morbo(&self) -> Option<&MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_ref()),
            _ => None,
        }
    }

    pub fn morbo_mut(&mut self) -> Option<&mut MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_mut()),
            _ => None,
        }
    }

    pub fn length(&self) -> f64 {
        match self {
            TrustRegionState::Turbo(t) => t.length(),
            TrustRegionState::Morbo(m) => m.as_ref().length(),
            TrustRegionState::Multi { state, .. } => state.lengths.get(0).copied().unwrap_or(0.0),
        }
    }

    /// Observations accounted for by the TuRBO length-adaptation state (0 for
    /// Morbo/Multi).
    pub fn turbo_obs(&self) -> usize {
        match self {
            TrustRegionState::Turbo(t) => t.prev_obs(),
            TrustRegionState::Morbo(_) | TrustRegionState::Multi { .. } => 0,
        }
    }

    pub fn set_obs(&mut self, prev_obs: usize) {
        if let TrustRegionState::Turbo(t) = self {
            t.set_watermark(prev_obs);
        }
    }

    pub fn seed_history(&mut self, prefix: &ArrayView1<f64>) {
        if let TrustRegionState::Turbo(t) = self {
            t.seed_history(prefix);
        }
    }

    pub fn set_arms(&mut self, num_arms: usize) {
        match self {
            TrustRegionState::Turbo(t) => t.set_arms(num_arms),
            TrustRegionState::Morbo(m) => m.as_mut().set_arms(num_arms),
            TrustRegionState::Multi { .. } => {}
        }
    }

    pub fn compute_bounds(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        match self {
            TrustRegionState::Turbo(t) => t.compute_bounds(x_center, lengthscales),
            TrustRegionState::Morbo(m) => m.as_ref().compute_bounds(x_center, lengthscales),
            TrustRegionState::Multi { state, .. } => state.compute_bounds(0, lengthscales),
        }
    }

    pub fn needs_restart(&self) -> bool {
        match self {
            TrustRegionState::Turbo(t) => t.needs_restart(),
            TrustRegionState::Morbo(m) => m.as_ref().needs_restart(),
            TrustRegionState::Multi { .. } => false,
        }
    }

    pub fn restart(&mut self, rng: Option<&mut dyn RngCore>) {
        match self {
            TrustRegionState::Turbo(t) => t.restart(),
            TrustRegionState::Morbo(m) => m.as_mut().restart(rng),
            TrustRegionState::Multi { .. } => {}
        }
    }

    pub fn restart_local(&mut self) {
        match self {
            TrustRegionState::Turbo(t) => t.restart_local(),
            TrustRegionState::Morbo(m) => m.as_mut().restart(None),
            TrustRegionState::Multi { .. } => {}
        }
    }

    pub fn resample_propose(&mut self, rng: &mut dyn RngCore) {
        if let TrustRegionState::Morbo(m) = self {
            if m.as_ref().rescalarize() == Rescalarize::OnPropose {
                m.as_mut().resample_weights(rng);
            }
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            TrustRegionState::Turbo(_) | TrustRegionState::Multi { .. } => 1,
            TrustRegionState::Morbo(m) => m.as_ref().num_metrics(),
        }
    }

    pub fn morbo_scalarize(
        &self,
        y: &ArrayView2<f64>,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        match self {
            TrustRegionState::Morbo(m) => m.as_ref().scalarize(y, clip),
            _ => Err(TrustRegionError::InvalidState(
                "scalarize requires Morbo trust region".to_string(),
            )),
        }
    }

    pub fn morbo_only(&mut self, y_new: &ArrayView2<f64>) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => {
                m.as_mut().update_incremental(y_new);
                Ok(())
            }
            _ => Err(ENNError::InvalidParameter(
                "morbo_only requires Morbo".to_string(),
            )),
        }
    }

    pub fn update_morbo(
        &mut self,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update_only(y_incumbent, num_obs)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            _ => Err(ENNError::InvalidParameter(
                "update_morbo requires Morbo".to_string(),
            )),
        }
    }

    pub fn rescale_morbo(&mut self, num_obs: usize) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .rescalarize_weights(num_obs)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            _ => Ok(()),
        }
    }

    pub fn tell_update(
        &mut self,
        y_all: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Turbo(t) => {
                if y_all.ncols() != 1 || y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(
                        "Turbo TR expects 1 objective column and 1 incumbent scalar".to_string(),
                    ));
                }
                let y_1d = y_all.column(0);
                t.update_history(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
            }
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update(&y_all.view(), y_incumbent)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            TrustRegionState::Multi { .. } => Ok(()),
        }
    }

    /// TuRBO trust-region update from the newly told batch only (O(batch), not
    /// O(history)).
    pub fn tell_batch(
        &mut self,
        y_new: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Turbo(t) => {
                if y_new.ncols() != 1 || y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(
                        "Turbo TR expects 1 objective column and 1 incumbent scalar".to_string(),
                    ));
                }
                let y_1d = y_new.column(0);
                t.update_batch(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
            }
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update(&y_new.view(), y_incumbent)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
            TrustRegionState::Multi { .. } => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::morbo::{MorboTRSettings, Rescalarize};
    use crate::trust_region::{TRLengthConfig, TrustRegionConfig};

    fn morbo_cfg(rescalarize: Rescalarize) -> TrustRegionConfig {
        TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize,
            noise_aware: false,
        })
    }

    #[test]
    fn morbo_config() {
        let mut rng = StdRng::seed_from_u64(99);
        let mut tr =
            TrustRegionState::from_config(3, &morbo_cfg(Rescalarize::OnPropose), &mut rng).unwrap();
        assert!(tr.is_morbo());
        assert_eq!(tr.num_metrics(), 2);
        tr.resample_propose(&mut rng);
    }

    #[test]
    fn morbo_propose2() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut tr =
            TrustRegionState::from_config(2, &morbo_cfg(Rescalarize::OnPropose), &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_ne!(w0, w1);
        assert_ne!(w1, w2);
    }

    #[test]
    fn turbo_max() {
        let mut tr = TrustRegionState::Turbo(TurboTrustRegion::new(2, TRLengthConfig::default()));
        tr.set_arms(1);
        let y0 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        tr.tell_update(&y0.view(), &array![1.0].view(), 1).unwrap();
        let len_before = tr.length();

        for (i, inc_val) in [2.0_f64, 3.0, 4.0].into_iter().enumerate() {
            let n = i + 2;
            let mut vals = vec![1.0];
            vals.extend(std::iter::repeat_n(0.5, n - 1));
            let y_mat = Array2::from_shape_vec((n, 1), vals).unwrap();
            tr.tell_update(&y_mat.view(), &array![inc_val].view(), n)
                .unwrap();
        }
        assert!(tr.length() > len_before);
    }

    #[test]
    fn morbo_restart() {
        let mut rng = StdRng::seed_from_u64(12);
        let mut tr =
            TrustRegionState::from_config(2, &morbo_cfg(Rescalarize::OnRestart), &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_eq!(w0, w1);
        tr.restart(Some(&mut rng));
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert_ne!(w1, w2);
    }
}
