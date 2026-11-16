use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;
use crate::morbo::{MorboTrustRegion, Rescalarize};
use crate::optimizer::multi_tr::{MultiTrustRegionState, PendingProposal};
use crate::trust_region::{TrustRegionConfig, TrustRegionError, TurboTrustRegion};

#[derive(Debug, Clone)]
pub enum TrustRegionState {
    Turbo(TurboTrustRegion),
    Morbo(Box<MorboTrustRegion>),
    Multi {
        state: MultiTrustRegionState,
        pending: Option<PendingProposal>,
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

    pub fn multi(state: MultiTrustRegionState, pending: Option<PendingProposal>) -> Self {
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

    pub fn set_obs(mut self, prev_obs: usize) -> Self {
        if let TrustRegionState::Turbo(t) = &mut self {
            t.set_watermark(prev_obs);
        }
        self
    }

    pub fn with_obs(self, prev_obs: usize) -> Self {
        self.set_obs(prev_obs)
    }

    pub fn seed_baseline(mut self, best_value: f64, prev_obs: usize) -> Self {
        if let TrustRegionState::Turbo(t) = &mut self {
            t.seed_baseline(best_value, prev_obs);
        }
        self
    }

    pub fn set_arms(self, num_arms: usize) -> Self {
        match self {
            TrustRegionState::Turbo(mut t) => {
                t.set_arms(num_arms);
                TrustRegionState::Turbo(t)
            }
            TrustRegionState::Morbo(m) => TrustRegionState::Morbo(Box::new(m.set_arms(num_arms))),
            TrustRegionState::Multi { .. } => self,
        }
    }

    pub fn with_arms(self, num_arms: usize) -> Self {
        self.set_arms(num_arms)
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

    pub fn restart(self, rng: Option<&mut dyn RngCore>) -> Self {
        match self {
            TrustRegionState::Turbo(mut t) => {
                t.restart();
                TrustRegionState::Turbo(t)
            }
            TrustRegionState::Morbo(m) => TrustRegionState::Morbo(Box::new(m.restart(rng))),
            TrustRegionState::Multi { .. } => self,
        }
    }

    pub fn with_restart(self, rng: Option<&mut dyn RngCore>) -> Self {
        self.restart(rng)
    }

    pub fn restart_local(self) -> Self {
        match self {
            TrustRegionState::Turbo(mut t) => {
                t.restart_local();
                TrustRegionState::Turbo(t)
            }
            TrustRegionState::Morbo(m) => TrustRegionState::Morbo(Box::new(m.restart(None))),
            TrustRegionState::Multi { .. } => self,
        }
    }

    pub fn resample_propose(self, rng: &mut dyn RngCore) -> (Self, bool) {
        match self {
            TrustRegionState::Morbo(m) => {
                if m.rescalarize() == Rescalarize::OnPropose {
                    let next_m = m.resample_weights(rng);
                    (TrustRegionState::Morbo(Box::new(next_m)), true)
                } else {
                    (TrustRegionState::Morbo(m), false)
                }
            }
            other => (other, false),
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

    pub fn morbo_only(self, y_new: &ArrayView2<f64>) -> Result<Self, ENNError> {
        match self {
            TrustRegionState::Morbo(m) => {
                let next_m = m.update_incremental(y_new);
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            _ => Err(ENNError::InvalidParameter(
                "morbo_only requires Morbo".to_string(),
            )),
        }
    }

    pub fn update_morbo(
        self,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        match self {
            TrustRegionState::Morbo(m) => {
                let next_m = m
                    .update_only(y_incumbent, num_obs)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            _ => Err(ENNError::InvalidParameter(
                "update_morbo requires Morbo".to_string(),
            )),
        }
    }

    pub fn rescale_morbo(
        self,
        num_obs: usize,
        incumbent: Option<ndarray::Array1<f64>>,
    ) -> Result<Self, ENNError> {
        match self {
            TrustRegionState::Morbo(m) => {
                let next_m = m
                    .rescalarize_weights(num_obs, incumbent)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            other => Ok(other),
        }
    }

    pub fn tell_update(
        self,
        y_all: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        match self {
            TrustRegionState::Turbo(mut t) => {
                if y_all.ncols() != 1 || y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(
                        "Turbo TR expects 1 objective column and 1 incumbent scalar".to_string(),
                    ));
                }
                let y_1d = y_all.column(0);
                t.update_history(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Turbo(t))
            }
            TrustRegionState::Morbo(m) => {
                let next_m = m
                    .update(&y_all.view(), y_incumbent)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            TrustRegionState::Multi { .. } => Ok(self),
        }
    }

    pub fn update_history(
        self,
        y_all: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        self.tell_update(y_all, y_incumbent, num_obs)
    }

    /// TuRBO trust-region update from the newly told batch only (O(batch), not
    /// O(history)).
    pub fn update_batch(
        self,
        y_new: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        match self {
            TrustRegionState::Turbo(mut t) => {
                if y_new.ncols() != 1 || y_incumbent.len() != 1 {
                    return Err(ENNError::InvalidParameter(
                        "Turbo TR expects 1 objective column and 1 incumbent scalar".to_string(),
                    ));
                }
                let y_1d = y_new.column(0);
                t.update_batch(&y_1d, num_obs, y_incumbent[0])
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Turbo(t))
            }
            TrustRegionState::Morbo(m) => {
                let next_m = m
                    .update(&y_new.view(), y_incumbent)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            TrustRegionState::Multi { .. } => Ok(self),
        }
    }

    pub fn tell_batch(
        self,
        y_new: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        self.update_batch(y_new, y_incumbent, num_obs)
    }

    pub fn update_posterior(
        self,
        posterior: &super::obs_access::PosteriorStore,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<Self, ENNError> {
        if let Some(y_obs) = posterior.y_obs() {
            self.tell_update(&y_obs.view(), y_incumbent, num_obs)
        } else {
            Ok(self)
        }
    }

    pub(crate) fn freeze_regions(&self, buf: &mut Vec<u8>) {
        match self {
            TrustRegionState::Turbo(t) => {
                buf.push(0);
                t.freeze_turbo(buf);
            }
            TrustRegionState::Morbo(m) => {
                buf.push(1);
                m.freeze_morbo(buf);
            }
            TrustRegionState::Multi { state, pending } => {
                buf.push(2);
                state.freeze_multi(pending.as_ref(), buf);
            }
        }
    }

    pub(crate) fn thaw_regions(
        self,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, ENNError> {
        use crate::snapshot_io::bad_state;
        let tag = reader.read_byte()?;
        match (tag, self) {
            (0, TrustRegionState::Turbo(mut t)) => {
                t.thaw_turbo(reader)?;
                Ok(TrustRegionState::Turbo(t))
            }
            (1, TrustRegionState::Morbo(m)) => {
                let next_m = (*m).thaw_morbo(reader)?;
                Ok(TrustRegionState::Morbo(Box::new(next_m)))
            }
            (2, TrustRegionState::Multi { mut state, .. }) => {
                let pending = state.thaw_multi(reader)?;
                Ok(TrustRegionState::Multi { state, pending })
            }
            _ => Err(bad_state(
                "frozen trust-region variant does not match template optimizer",
            )),
        }
    }

    pub(crate) fn thaw(
        self,
        reader: &mut crate::snapshot_io::WireReader<'_>,
    ) -> Result<Self, ENNError> {
        self.thaw_regions(reader)
    }

    pub(crate) fn hash_state(&self, buf: &mut Vec<u8>) {
        match self {
            TrustRegionState::Turbo(t) => {
                buf.push(0);
                t.hash_turbo(buf);
            }
            TrustRegionState::Morbo(m) => {
                buf.push(1);
                m.hash_morbo(buf);
            }
            TrustRegionState::Multi { state, pending } => {
                buf.push(2);
                state.hash_multi(pending.as_ref(), buf);
            }
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
        let tr =
            TrustRegionState::from_config(3, &morbo_cfg(Rescalarize::OnPropose), &mut rng).unwrap();
        assert!(tr.is_morbo());
        assert_eq!(tr.num_metrics(), 2);
        let (next_tr, resampled) = tr.resample_propose(&mut rng);
        assert!(resampled);
        let _ = next_tr;
    }

    #[test]
    fn morbo_propose2() {
        let mut rng = StdRng::seed_from_u64(11);
        let tr =
            TrustRegionState::from_config(2, &morbo_cfg(Rescalarize::OnPropose), &mut rng).unwrap();
        let w0 = tr.morbo().expect("morbo").weights().to_owned();
        let (tr, resampled1) = tr.resample_propose(&mut rng);
        assert!(resampled1);
        let w1 = tr.morbo().expect("morbo").weights().to_owned();
        let (tr, resampled2) = tr.resample_propose(&mut rng);
        assert!(resampled2);
        let w2 = tr.morbo().expect("morbo").weights().to_owned();
        assert_ne!(w0, w1);
        assert_ne!(w1, w2);
    }

    #[test]
    fn turbo_max() {
        let mut tr = TrustRegionState::Turbo(TurboTrustRegion::new(2, TRLengthConfig::default()));
        tr = tr.set_arms(1);
        let y0 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        tr = tr.tell_update(&y0.view(), &array![1.0].view(), 1).unwrap();
        let len_before = tr.length();

        for (i, inc_val) in [2.0_f64, 3.0, 4.0].into_iter().enumerate() {
            let n = i + 2;
            let mut vals = vec![1.0];
            vals.extend(std::iter::repeat_n(0.5, n - 1));
            let y_mat = Array2::from_shape_vec((n, 1), vals).unwrap();
            tr = tr
                .tell_update(&y_mat.view(), &array![inc_val].view(), n)
                .unwrap();
        }
        assert!(tr.length() > len_before);
    }

    #[test]
    fn morbo_restart() {
        let mut rng = StdRng::seed_from_u64(12);
        let tr =
            TrustRegionState::from_config(2, &morbo_cfg(Rescalarize::OnRestart), &mut rng).unwrap();
        let w0 = tr.morbo().expect("morbo").weights().to_owned();
        let (tr, resampled) = tr.resample_propose(&mut rng);
        assert!(!resampled);
        let w1 = tr.morbo().expect("morbo").weights().to_owned();
        assert_eq!(w0, w1);
        let tr = tr.restart(Some(&mut rng));
        let w2 = tr.morbo().expect("morbo").weights().to_owned();
        assert_ne!(w1, w2);
    }
}
