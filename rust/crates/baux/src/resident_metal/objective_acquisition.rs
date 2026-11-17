//! Independent objective acquisition; scalar control remains caller-supplied.
use super::*;
use crate::Rescalarize;
use crate::hash::domain_prng;
use crate::morbo::{MorboTRSettings, MorboTrustRegion};
use crate::objective_observation::MAX_OBJECTIVES;
use rand::Rng;
use std::sync::Arc;

mod parameters;
use parameters::ObjectiveParams;
mod hypervolume;
use hypervolume::{exclusive_volume, improvement, max_exclusive, measured, values};

pub use crate::config::{ObjectiveAcquisition, ObjectivePolicy};

#[derive(Clone)]
struct ArchiveEntry {
    identity: i64,
    row: Arc<Buffer>,
    observation: crate::objective_observation::ObjectiveObservation,
}

#[derive(Clone)]
struct MorboRegion {
    trust: MorboTrustRegion,
    center: Option<Arc<Buffer>>,
    record: crate::objective_observation::ObjectiveRecord,
    trials: u64,
}

#[derive(Clone)]
pub struct ResidentMorbo {
    regions: Arc<Vec<MorboRegion>>,
    archive: Arc<Vec<ArchiveEntry>>,
    active: usize,
    preferences: Vec<f32>,
    scales: Vec<f32>,
    reference: Vec<f64>,
    seed: u64,
    alpha: f64,
    pub(super) rescalarize: Rescalarize,
    clip: bool,
    observations: usize,
    propose_epoch: usize,
}

fn validate_policy(policy: &ObjectivePolicy) -> Result<(), String> {
    if !(2..=MAX_OBJECTIVES).contains(&policy.scales.len())
        || policy
            .scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err("Vector acquisition requires 2..=8 positive finite objective scales".into());
    }
    if let ObjectiveAcquisition::Morbo {
        weights,
        regions,
        alpha,
        clip,
        ..
    } = &policy.acquisition
    {
        if weights.len() != policy.scales.len()
            || *regions == 0
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || !alpha.is_finite()
            || *alpha < 0.0
            || !*clip
        {
            return Err(
                "MORBO requires regions >= 1, one positive finite preference per objective, \
                 finite nonnegative alpha and clip=true"
                    .into(),
            );
        }
    }
    Ok(())
}

fn chebyshev_utility(
    obs: crate::objective_observation::ObjectiveObservation,
    weights: &ndarray::Array1<f64>,
    y_min: Option<&ndarray::Array1<f64>>,
    y_max: Option<&ndarray::Array1<f64>>,
    alpha: f64,
    clip: bool,
) -> f64 {
    let ests = obs.estimates();
    let mut minimum = f64::INFINITY;
    let mut sum = 0.0f64;
    for (m, est) in ests.iter().enumerate() {
        let (lo, hi) = match (y_min, y_max) {
            (Some(ymin), Some(ymax)) => (ymin[m], ymax[m]),
            _ => (f64::from(est.mean), f64::from(est.mean)),
        };
        let range = hi - lo;
        let mut norm = if range <= 0.0 {
            0.5
        } else {
            (f64::from(est.mean) - lo) / range
        };
        if clip {
            norm = norm.clamp(0.0, 1.0);
        }
        let w = norm * weights[m];
        minimum = minimum.min(w);
        sum += w;
    }
    minimum + alpha * sum
}

impl ResidentMorbo {
    fn new(state: &SearchState, acquisition: &ObjectiveAcquisition) -> Result<Self, String> {
        let ObjectiveAcquisition::Morbo {
            weights,
            regions,
            alpha,
            seed,
            rescalarize,
            clip,
        } = acquisition
        else {
            return Err("Resident MORBO requires MORBO acquisition".into());
        };
        if *regions == 0 {
            return Err("Resident MORBO requires at least 1 trust region".into());
        }
        let incumbent = *state
            .objective_history
            .incumbent()
            .ok_or("Configure MORBO after the initial vector observation")?;
        let mut region_vec = Vec::with_capacity(*regions);
        for region in 0..*regions {
            let mut rng = domain_prng(*seed, region as u64);
            let trust = MorboTrustRegion::new(
                state.dimensions,
                MorboTRSettings {
                    num_metrics: weights.len(),
                    alpha: f64::from(*alpha),
                    length: state.length_config,
                    rescalarize: Rescalarize::OnRestart,
                    noise_aware: state.noise_aware,
                },
                &mut rng,
            )
            .map_err(|error| error.to_string())?;
            let trust = trust.set_arms(state.pool_layout.arms() as usize);
            let trust = trust
                .set_tolerance(state.trust.failure_tolerance() as usize)
                .map_err(|error| error.to_string())?;
            let center = if region == 0 {
                None
            } else {
                let center = state.runtime.buffer::<u16>(state.dimensions);
                state.copy(&state.base, &center)?;
                Some(Arc::new(center))
            };
            region_vec.push(MorboRegion {
                trust,
                center,
                record: incumbent,
                trials: 0,
            });
        }
        let initial_archive = {
            let snap = state.runtime.buffer::<u16>(state.dimensions);
            state.copy(&state.base, &snap)?;
            vec![ArchiveEntry {
                identity: incumbent.identity,
                row: Arc::new(snap),
                observation: incumbent.observation,
            }]
        };
        let mut resident = Self {
            regions: Arc::new(region_vec),
            archive: Arc::new(initial_archive),
            active: 0,
            preferences: weights.clone(),
            scales: Vec::new(),
            reference: incumbent
                .observation
                .estimates()
                .iter()
                .map(|estimate| f64::from(estimate.mean) - 1.0)
                .collect(),
            seed: *seed,
            alpha: f64::from(*alpha),
            rescalarize: *rescalarize,
            clip: *clip,
            observations: 0,
            propose_epoch: 0,
        };
        for region in 0..*regions {
            resident.sample_weights(region, 0)?;
            resident.seed(region, incumbent.observation)?;
        }
        Ok(resident)
    }

    fn install_scales(&mut self, scales: &[f32]) {
        self.scales = scales.to_vec();
        let observation = self.regions[0].record.observation;
        self.reference = observation
            .estimates()
            .iter()
            .zip(scales)
            .map(|(estimate, scale)| f64::from(estimate.mean - scale))
            .collect();
    }

    pub fn regions_mut(&mut self, state: &SearchState) -> Result<(), String> {
        if Arc::strong_count(&self.regions) > 1 {
            let mut cloned = (*self.regions).clone();
            for region in &mut cloned {
                if let Some(center) = &mut region.center {
                    let new_center = state.runtime.buffer::<u16>(state.dimensions);
                    state.copy(center, &new_center)?;
                    *center = Arc::new(new_center);
                }
            }
            self.regions = Arc::new(cloned);
        }
        Ok(())
    }

    pub(super) fn record_archive(
        &mut self,
        state: &SearchState,
        identity: i64,
        row_source: &Buffer,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let snap = state.runtime.buffer::<u16>(state.dimensions);
        state.copy(row_source, &snap)?;
        let archive = Arc::make_mut(&mut self.archive);
        if archive.len() >= MAX_HISTORY {
            archive.remove(0);
        }
        archive.push(ArchiveEntry {
            identity,
            row: Arc::new(snap),
            observation,
        });
        Ok(())
    }

    fn sample_weights(&mut self, region: usize, epoch: usize) -> Result<(), String> {
        let stream = ((region as u64) << 32) ^ epoch as u64;
        let mut rng = domain_prng(self.seed, stream);
        let samples = self
            .preferences
            .iter()
            .map(|preference| {
                -rng.r#gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0).ln() * f64::from(*preference)
            })
            .collect::<Vec<_>>();
        let regions = Arc::make_mut(&mut self.regions);
        regions[region].trust = regions[region]
            .trust
            .clone()
            .set_weights(&samples)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn seed(
        &mut self,
        region: usize,
        observation: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        let values = values(observation);
        let row = values.view().insert_axis(ndarray::Axis(0));
        let regions = Arc::make_mut(&mut self.regions);
        regions[region].trust = regions[region].trust.clone().update_incremental(&row);
        self.observations = self.observations.max(1);
        regions[region].trust = regions[region]
            .trust
            .clone()
            .update_only(&values.view(), self.observations)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn candidate_improves(
        &self,
        history: &crate::objective_observation::ObjectiveWindow,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(bool, f64), String> {
        let before = measured(history.rows(), None, &self.reference, &self.scales)?;
        let after = measured(
            history.rows(),
            Some(candidate),
            &self.reference,
            &self.scales,
        )?;
        Ok(improvement(before, after))
    }

    pub fn observe(
        &mut self,
        candidate: crate::objective_observation::ObjectiveObservation,
        accepted: bool,
        identity: i64,
    ) -> Result<(), String> {
        let region = self.active;
        let observation = candidate;
        let candidate = values(observation);
        let regions = Arc::make_mut(&mut self.regions);
        regions[region].trust = regions[region]
            .trust
            .clone()
            .update_incremental(&candidate.view().insert_axis(ndarray::Axis(0)));
        self.observations += 1;
        regions[region].trials += 1;
        if accepted {
            regions[region].record = crate::objective_observation::ObjectiveRecord {
                identity,
                observation,
            };
        }
        let incumbent = values(regions[region].record.observation);
        regions[region].trust = regions[region]
            .trust
            .clone()
            .update_only(&incumbent.view(), self.observations)
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn restart(
        &mut self,
        region: usize,
        epoch: usize,
        incumbent: crate::objective_observation::ObjectiveObservation,
    ) -> Result<(), String> {
        {
            let regions = Arc::make_mut(&mut self.regions);
            regions[region].trust = regions[region].trust.clone().restart(None);
        }
        self.sample_weights(region, epoch)?;
        self.seed(region, incumbent)
    }

    pub fn trust(&self) -> &MorboTrustRegion {
        &self.regions[self.active].trust
    }

    pub fn clip(&self) -> bool {
        self.clip
    }

    pub fn set_tolerance(&mut self, failures: usize) -> Result<(), String> {
        let regions = Arc::make_mut(&mut self.regions);
        for region in regions.iter_mut() {
            region.trust = region
                .trust
                .clone()
                .set_tolerance(failures)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    pub fn with_tolerance(mut self, failures: usize) -> Result<Self, String> {
        self.set_tolerance(failures)?;
        Ok(self)
    }

    fn select_region(&self) -> usize {
        if let Some(region) = self.regions.iter().position(|region| region.trials == 0) {
            return region;
        }
        let total_trials = self.regions.iter().map(|region| region.trials).sum::<u64>() as f64;
        let records = self
            .regions
            .iter()
            .map(|region| region.record)
            .collect::<Vec<_>>();
        let total_volume =
            measured(records.iter(), None, &self.reference, &self.scales).unwrap_or(0.0);
        self.regions
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                let left_score = exclusive_volume(
                    &records,
                    *left_index,
                    total_volume,
                    &self.reference,
                    &self.scales,
                ) + (2.0 * total_trials.ln() / left.trials as f64).sqrt();
                let right_score = exclusive_volume(
                    &records,
                    *right_index,
                    total_volume,
                    &self.reference,
                    &self.scales,
                ) + (2.0 * total_trials.ln() / right.trials as f64).sqrt();
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap()
    }

    fn region_count(&self) -> usize {
        self.regions.len()
    }

    fn max_region(&self, excluded: Option<usize>) -> Result<usize, String> {
        let excluded = if self.regions.len() <= 1 {
            None
        } else {
            excluded
        };
        let records = self
            .regions
            .iter()
            .map(|region| region.record)
            .collect::<Vec<_>>();
        max_exclusive(&records, excluded, &self.reference, &self.scales)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ObjectiveSelectionReport {
    pub values: [[f32; MAX_OBJECTIVES]; 4],
    pub weights: [f32; MAX_OBJECTIVES],
    pub width: u32,
    pub nondominated_mask: u32,
    pub selected: u32,
    pad: u32,
}

#[derive(Clone)]
pub struct ObjectiveSelector {
    config: ObjectivePolicy,
    reference: Vec<f64>,
    pub morbo: Option<ResidentMorbo>,
    pub score_pipeline: ComputePipelineState,
    pub pipeline: ComputePipelineState,
    pub parameters: Buffer,
    pub report: Buffer,
    pub vectors: Buffer,
}

impl ObjectiveSelector {
    pub(super) fn scales(&self) -> &[f32] {
        &self.config.scales
    }

    fn new(state: &SearchState, config: ObjectivePolicy) -> Result<Self, String> {
        validate_policy(&config)?;
        parameters::validate_history(state, config.scales.len())?;
        let size = size_of::<ObjectiveParams>() as u64;
        let report_size = size_of::<ObjectiveSelectionReport>() as u64;
        let vec_count = (state.pool_layout.count().max(4) as usize) * MAX_OBJECTIVES;
        let vec_size = bytes::<f32>(vec_count)?;
        let region_count = match &config.acquisition {
            ObjectiveAcquisition::Morbo { regions, .. } => *regions,
            ObjectiveAcquisition::Pareto => 0,
        };
        let mut sizes = vec![size, report_size, vec_size];
        sizes.extend(std::iter::repeat_n(
            state.row_bytes(),
            region_count.saturating_sub(1),
        ));
        let resident = sizes.iter().try_fold(0u64, |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or("MORBO memory accounting overflow")
        })?;
        preflight(&state.runtime, &sizes, resident)?;
        let mut morbo = matches!(config.acquisition, ObjectiveAcquisition::Morbo { .. })
            .then(|| ResidentMorbo::new(state, &config.acquisition))
            .transpose()?;
        if let Some(morbo) = &mut morbo {
            morbo.install_scales(&config.scales);
        }
        let source = format!(
            "{}\n{}",
            state.procedural_source(state.pool_layout),
            include_str!("../resident_objectives.metal")
        );
        let reference = state
            .objective_history
            .rows()
            .next()
            .ok_or("Vector acquisition requires an initial observation")?
            .observation
            .estimates()
            .iter()
            .zip(&config.scales)
            .map(|(estimate, scale)| f64::from(estimate.mean - scale))
            .collect();
        Ok(Self {
            config,
            reference,
            morbo,
            score_pipeline: state.runtime.precise(
                &source,
                "Vector search",
                "dense_score_objectives",
            )?,
            pipeline: state
                .runtime
                .precise(&source, "Vector search", "dense_select_objectives")?,
            parameters: state.runtime.buffer::<ObjectiveParams>(1),
            report: state
                .runtime
                .buffer_with(&[ObjectiveSelectionReport::default()]),
            vectors: state.runtime.buffer::<f32>(vec_count),
        })
    }

    pub fn upload(&self, state: &SearchState, root: u64, ask: RoundConfig) {
        let parameters = ObjectiveParams::pack(state, &self.config, self.morbo.as_ref(), root, ask);
        unsafe {
            self.parameters
                .contents()
                .cast::<ObjectiveParams>()
                .write(parameters);
        }
    }
}

impl SearchState {
    pub fn configure_objectives(mut self, config: Option<ObjectivePolicy>) -> Result<Self, String> {
        self.check_idle()?;
        if config.is_some() && self.manifold.is_some() {
            self.refresh_manifold()?;
        }
        self.objective_selector = config
            .map(|config| ObjectiveSelector::new(&self, config))
            .transpose()?;
        if let Some(morbo) = self
            .objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
        {
            self.length = morbo.trust().length();
        }
        Ok(self)
    }

    pub fn validate_selector(&self) -> Result<(), String> {
        if let Some(selector) = &self.objective_selector {
            parameters::validate_history(self, selector.config.scales.len())?;
        }
        Ok(())
    }

    pub fn objective_decision(
        &self,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<Option<(bool, f64)>, String> {
        let Some(selector) = self.objective_selector.as_ref() else {
            return Ok(None);
        };
        if let Some(morbo) = selector.morbo.as_ref() {
            return morbo
                .candidate_improves(&self.objective_history, candidate)
                .map(Some);
        }
        let before = measured(
            self.objective_history.rows(),
            None,
            &selector.reference,
            &selector.config.scales,
        )?;
        let after = measured(
            self.objective_history.rows(),
            Some(candidate),
            &selector.reference,
            &selector.config.scales,
        )?;
        Ok(Some(improvement(before, after)))
    }

    pub fn update_morbo(
        mut self,
        candidate: crate::objective_observation::ObjectiveObservation,
    ) -> Result<Self, String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok(self);
        };
        let Some(mut morbo) = selector.morbo.take() else {
            self.objective_selector = Some(selector);
            return Ok(self);
        };
        let region = morbo.active;
        let accepted = self.base_id == self.observation;
        if morbo.rescalarize == Rescalarize::OnPropose {
            let source_buf: &Buffer = if accepted {
                &self.base
            } else if !self.history_rows.is_empty() && self.resident_history > 0 {
                let last_slot = self.resident_history - 1;
                &self.history_rows[last_slot]
            } else {
                &self.base
            };
            if let Err(e) = morbo.record_archive(&self, self.observation, source_buf, candidate) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
        }
        if let Err(e) = morbo.observe(candidate, accepted, self.observation) {
            selector.morbo = Some(morbo);
            self.objective_selector = Some(selector);
            return Err(e);
        }
        if morbo.trust().needs_restart() {
            self.restart_count += 1;
            let source = match morbo.max_region(Some(region)) {
                Ok(s) => s,
                Err(e) => {
                    selector.morbo = Some(morbo);
                    self.objective_selector = Some(selector);
                    return Err(e);
                }
            };
            if let Err(e) = morbo.regions_mut(&self) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
            let source_center = if source == region {
                Arc::clone(&self.base)
            } else {
                match morbo.regions[source].center.clone() {
                    Some(c) => c,
                    None => {
                        selector.morbo = Some(morbo);
                        self.objective_selector = Some(selector);
                        return Err("MORBO inactive region lost its resident center".into());
                    }
                }
            };
            let source_record = morbo.regions[source].record;
            let regions = Arc::make_mut(&mut morbo.regions);
            regions[region].record = source_record;
            if let Err(e) = morbo.restart(region, self.restart_count, source_record.observation) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
            let fresh_base = self.runtime.buffer::<u16>(self.dimensions);
            if let Err(e) = self.copy(&source_center, &fresh_base) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
            self.base = Arc::new(fresh_base);
            self.base_id = source_record.identity;
            self.best = source_record.observation.control().mean;
            self.best_variance = source_record.observation.control().variance;
        }
        self.length = morbo.trust().length();
        selector.morbo = Some(morbo);
        self.objective_selector = Some(selector);
        Ok(self)
    }

    pub fn has_morbo(&self) -> bool {
        self.objective_selector
            .as_ref()
            .is_some_and(|selector| selector.morbo.is_some())
    }

    pub fn activate_morbo(mut self) -> Result<Self, String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok(self);
        };
        let Some(mut morbo) = selector.morbo.take() else {
            self.objective_selector = Some(selector);
            return Ok(self);
        };
        let region = morbo.select_region();
        if let Err(e) = self.switch_region(&mut morbo, region) {
            selector.morbo = Some(morbo);
            self.objective_selector = Some(selector);
            return Err(e);
        }
        if morbo.rescalarize == Rescalarize::OnPropose {
            morbo.propose_epoch = morbo.propose_epoch.wrapping_add(1);
            let stream_epoch = morbo.propose_epoch.wrapping_add(1_048_576);
            if let Err(e) = morbo.sample_weights(region, stream_epoch) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
            // Re-center on the persistent archived observation maximizing the
            // new scalarization
            let trust = &morbo.regions[region].trust;
            let best_entry = morbo
                .archive
                .iter()
                .filter(|entry| self.identities[..self.history].contains(&entry.identity))
                .max_by(|a, b| {
                    let ua = chebyshev_utility(
                        a.observation,
                        trust.weights(),
                        trust.y_min(),
                        trust.y_max(),
                        morbo.alpha,
                        morbo.clip,
                    );
                    let ub = chebyshev_utility(
                        b.observation,
                        trust.weights(),
                        trust.y_min(),
                        trust.y_max(),
                        morbo.alpha,
                        morbo.clip,
                    );
                    ua.total_cmp(&ub).then_with(|| a.identity.cmp(&b.identity))
                })
                .cloned();
            if let Some(entry) = best_entry {
                let fresh_base = self.runtime.buffer::<u16>(self.dimensions);
                if let Err(e) = self.copy(&entry.row, &fresh_base) {
                    selector.morbo = Some(morbo);
                    self.objective_selector = Some(selector);
                    return Err(e);
                }
                self.base = Arc::new(fresh_base);
                let regions = Arc::make_mut(&mut morbo.regions);
                regions[region].record = crate::objective_observation::ObjectiveRecord {
                    identity: entry.identity,
                    observation: entry.observation,
                };
            }
        }
        let state = &morbo.regions[region];
        self.base_id = state.record.identity;
        self.best = state.record.observation.control().mean;
        self.best_variance = state.record.observation.control().variance;
        self.length = state.trust.length();
        selector.morbo = Some(morbo);
        self.objective_selector = Some(selector);
        Ok(self)
    }

    pub fn morbo_region(&self) -> Option<usize> {
        self.objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
            .map(|morbo| morbo.active)
    }

    pub fn morbo_regions(&self) -> usize {
        self.objective_selector
            .as_ref()
            .and_then(|selector| selector.morbo.as_ref())
            .map_or(0, ResidentMorbo::region_count)
    }

    fn switch_region(&mut self, morbo: &mut ResidentMorbo, region: usize) -> Result<(), String> {
        if region == morbo.active {
            return Ok(());
        }
        morbo.regions_mut(self)?;
        let regions = Arc::make_mut(&mut morbo.regions);
        if regions[morbo.active].center.is_some() {
            return Err("MORBO active region unexpectedly owned a stored center".into());
        }
        let center = regions[region]
            .center
            .take()
            .ok_or("MORBO inactive region lost its resident center")?;
        let previous = std::mem::replace(&mut self.base, center);
        regions[morbo.active].center = Some(previous);
        morbo.active = region;
        if self
            .history_rows
            .iter()
            .any(|row| std::ptr::eq::<metal::BufferRef>(&***row, &**self.base))
        {
            let isolated = self.runtime.buffer::<u16>(self.dimensions);
            self.copy(&self.base, &isolated)?;
            self.base = Arc::new(isolated);
        }
        self.cow_base()?;
        Ok(())
    }

    pub fn finalize_morbo(mut self) -> Result<(Self, Option<usize>), String> {
        let Some(mut selector) = self.objective_selector.take() else {
            return Ok((self, None));
        };
        let Some(mut morbo) = selector.morbo.take() else {
            self.objective_selector = Some(selector);
            return Ok((self, None));
        };
        let region = match morbo.max_region(None) {
            Ok(r) => r,
            Err(e) => {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
        };
        if let Err(e) = self.switch_region(&mut morbo, region) {
            selector.morbo = Some(morbo);
            self.objective_selector = Some(selector);
            return Err(e);
        }
        let record = morbo.regions[region].record;
        let length = morbo.regions[region].trust.length();
        self.base_id = record.identity;
        self.best = record.observation.control().mean;
        self.best_variance = record.observation.control().variance;
        self.length = length;
        selector.morbo = Some(morbo);
        self.objective_selector = Some(selector);
        Ok((self, Some(region)))
    }

    pub fn objective_report(&self) -> Result<Option<ObjectiveSelectionReport>, String> {
        self.check_healthy()?;
        if self.async_command.is_some() {
            return Err(
                "Finish the pending GPU selection before reading its objective report".into(),
            );
        }
        Ok(self.objective_selector.as_ref().and_then(|selector| {
            let report = read::<ObjectiveSelectionReport>(&selector.report, 1)[0];
            (report.width != 0).then_some(report)
        }))
    }

    /// Read all `pool_count` candidate objective acquisition vectors from the
    /// last completed vector selection.
    pub fn objective_vectors(&self) -> Result<Option<Vec<[f32; MAX_OBJECTIVES]>>, String> {
        self.check_healthy()?;
        if self.async_command.is_some() {
            return Err(
                "Finish the pending GPU selection before reading its objective vectors".into(),
            );
        }
        let Some(selector) = &self.objective_selector else {
            return Ok(None);
        };
        let report = read::<ObjectiveSelectionReport>(&selector.report, 1)[0];
        if report.width == 0 {
            return Ok(None);
        }
        let count = self.pool_layout.count() as usize;
        let raw = read::<f32>(&selector.vectors, count * MAX_OBJECTIVES);
        let vecs = raw
            .chunks_exact(MAX_OBJECTIVES)
            .map(|chunk| std::array::from_fn(|m| chunk[m]))
            .collect();
        Ok(Some(vecs))
    }
}

#[cfg(test)]
mod tests;
