use super::*;

#[repr(C)]
pub struct ObjectiveParams {
    outcomes: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    variances: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    draws: [[f32; MAX_HISTORY]; MAX_OBJECTIVES],
    scales: [f32; MAX_OBJECTIVES],
    weights: [f32; MAX_OBJECTIVES],
    minima: [f32; MAX_OBJECTIVES],
    maxima: [f32; MAX_OBJECTIVES],
    y_bounds_mode: [u32; MAX_OBJECTIVES],
    y_lower: [f32; MAX_OBJECTIVES],
    y_upper: [f32; MAX_OBJECTIVES],
    affine_a: [f32; MAX_OBJECTIVES],
    affine_b: [f32; MAX_OBJECTIVES],
    affine_c: [f32; MAX_OBJECTIVES],
    stiefel_proj: [[f32; MAX_OBJECTIVES]; MAX_OBJECTIVES],
    stiefel_enabled: u32,
    width: u32,
    mode: u32,
    alpha: f32,
    clip: u32,
    pad: u32,
}

const _: () = {
    assert!(core::mem::size_of::<ObjectiveParams>() == 12888);
};

pub fn validate_history(state: &SearchState, width: usize) -> Result<(), String> {
    if state.history == 0 {
        return Err("Vector acquisition requires measured absolute objective observations".into());
    }
    let records = state.objective_history.rows();
    if records.len() != state.history {
        return Err("Vector acquisition history is missing an objective row".into());
    }
    for (identity, record) in state.identities[..state.history].iter().zip(records) {
        if record.identity != *identity {
            return Err("Vector acquisition history order differs from scalar geometry".into());
        }
        if record.observation.estimates().len() != width {
            return Err("Vector acquisition scales do not match the fixed objective schema".into());
        }
        record.observation.validate()?;
    }
    Ok(())
}

impl ObjectiveParams {
    pub fn pack(
        state: &SearchState,
        config: &ObjectivePolicy,
        morbo: Option<&ResidentMorbo>,
        _root: u64,
        ask: RoundConfig,
    ) -> Self {
        let mut params = Self {
            outcomes: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            variances: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            draws: [[0.0; MAX_HISTORY]; MAX_OBJECTIVES],
            scales: [1.0; MAX_OBJECTIVES],
            weights: [0.0; MAX_OBJECTIVES],
            minima: [f32::INFINITY; MAX_OBJECTIVES],
            maxima: [f32::NEG_INFINITY; MAX_OBJECTIVES],
            y_bounds_mode: [0; MAX_OBJECTIVES],
            y_lower: [0.0; MAX_OBJECTIVES],
            y_upper: [0.0; MAX_OBJECTIVES],
            affine_a: [0.0; MAX_OBJECTIVES],
            affine_b: [1.0; MAX_OBJECTIVES],
            affine_c: [1.0; MAX_OBJECTIVES],
            stiefel_proj: [[0.0; MAX_OBJECTIVES]; MAX_OBJECTIVES],
            stiefel_enabled: 0,
            width: config.scales.len() as u32,
            mode: 0,
            alpha: 0.0,
            clip: 0,
            pad: 0,
        };
        params.scales[..config.scales.len()].copy_from_slice(&config.scales);
        if let Some(ms) = &state.manifold {
            if let Some(bounds) = &ms.config.y_bounds {
                for (m, &b) in bounds.iter().take(config.scales.len()).enumerate() {
                    let (mode, lo, hi) = super::manifold::bounds_mode(b);
                    params.y_bounds_mode[m] = mode;
                    params.y_lower[m] = lo;
                    params.y_upper[m] = hi;
                }
            }
            if let Some(aff) = &ms.affine {
                for m in 0..config.scales.len().min(aff.a.len()) {
                    params.affine_a[m] = aff.a[m] as f32;
                    params.affine_b[m] = aff.b[m] as f32;
                    params.affine_c[m] = aff.c[m] as f32;
                }
            }
            if let Some(st) = &ms.stiefel
                && st.rows() == config.scales.len()
            {
                let proj = st.projector();
                let w = st.rows();
                for i in 0..w {
                    for j in 0..w {
                        params.stiefel_proj[i][j] = proj[i * w + j] as f32;
                    }
                }
                params.stiefel_enabled = 1;
            }
        }
        params.fill_observations(state, ask.acquisition.seed);
        params.configure_mode(config, morbo);
        params
    }

    fn fill_observations(&mut self, state: &SearchState, seed: u64) {
        let y_bounds = state
            .manifold
            .as_ref()
            .and_then(|m| m.config.y_bounds.as_ref());
        for (row, record) in state.objective_history.rows().enumerate() {
            for (metric, estimate) in record.observation.estimates().iter().enumerate() {
                let raw_v = if state.noise_aware {
                    estimate.variance
                } else {
                    0.0
                };
                let bounds_m = y_bounds.and_then(|b| b.get(metric).copied());
                let (wy, wv) = super::manifold::warp_observation(estimate.mean, raw_v, bounds_m);
                self.outcomes[metric][row] = wy;
                self.variances[metric][row] = wv;
                self.draws[metric][row] =
                    crate::hash::normal_metric(seed, record.identity, metric) as f32;
                self.minima[metric] = self.minima[metric].min(estimate.mean);
                self.maxima[metric] = self.maxima[metric].max(estimate.mean);
            }
        }
    }

    fn configure_mode(&mut self, config: &ObjectivePolicy, morbo: Option<&ResidentMorbo>) {
        let ObjectiveAcquisition::Morbo { alpha, .. } = &config.acquisition else {
            return;
        };
        let morbo = morbo.expect("validated MORBO selector requires its state");
        let trust = morbo.trust();
        self.mode = 1;
        self.alpha = *alpha;
        self.clip = u32::from(morbo.clip());
        for (metric, &weight) in trust.weights().iter().enumerate() {
            self.weights[metric] = weight as f32;
        }
        if let (Some(minima), Some(maxima)) = (trust.y_min(), trust.y_max()) {
            for metric in 0..trust.num_metrics() {
                self.minima[metric] = minima[metric] as f32;
                self.maxima[metric] = maxima[metric] as f32;
            }
        }
    }
}
