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
struct MorboRegion {
    trust: MorboTrustRegion,
    center: Option<Arc<Buffer>>,
    record: crate::objective_observation::ObjectiveRecord,
    trials: u64,
}

#[derive(Clone)]
pub struct ResidentMorbo {
    regions: Arc<Vec<MorboRegion>>,
    active: usize,
    preferences: Vec<f32>,
    scales: Vec<f32>,
    reference: Vec<f64>,
    seed: u64,
    clip: bool,
    observations: usize,
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
        if *rescalarize != Rescalarize::OnRestart {
            return Err(
                "Exact resident MORBO requires rescalarize='on_restart'; changing scalarization \
                 every proposal would require rematerializing a historical billion-weight center"
                    .into(),
            );
        }
        if !(2..=8).contains(regions) {
            return Err("Resident MORBO requires 2..=8 trust regions".into());
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
                    rescalarize: *rescalarize,
                    noise_aware: true,
                },
                &mut rng,
            )
            .map_err(|error| error.to_string())?;
            let trust = trust.set_arms(1);
            let trust = trust
                .set_tolerance(state.trust.failure_tolerance() as usize)
                .map_err(|error| error.to_string())?;
            let center = if region == 0 {
                None
            } else {
                let center = state.runtime.buffer::<u16>(state.dimensions);
                state.copy(&*state.base, &center)?;
                Some(Arc::new(center))
            };
            region_vec.push(MorboRegion {
                trust,
                center,
                record: incumbent,
                trials: 0,
            });
        }
        let mut resident = Self {
            regions: Arc::new(region_vec),
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
            clip: *clip,
            observations: 0,
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
                    state.copy(&**center, &new_center)?;
                    *center = Arc::new(new_center);
                }
            }
            self.regions = Arc::new(cloned);
        }
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
        // The trust region deliberately preserves its global observation
        // watermark across a local restart. Seed the first region at one, but
        // never move that watermark backwards when recentering later regions.
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
    pub pipeline: ComputePipelineState,
    pub parameters: Buffer,
    pub report: Buffer,
}

impl ObjectiveSelector {
    fn new(state: &SearchState, config: ObjectivePolicy) -> Result<Self, String> {
        config.validate()?;
        parameters::validate_history(state, config.scales.len())?;
        let size = size_of::<ObjectiveParams>() as u64;
        let report_size = size_of::<ObjectiveSelectionReport>() as u64;
        let region_count = match &config.acquisition {
            ObjectiveAcquisition::Morbo { regions, .. } => *regions,
            ObjectiveAcquisition::Pareto => 0,
        };
        let mut sizes = vec![size, report_size];
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
            pipeline: state
                .runtime
                .precise(&source, "Vector search", "dense_select_objectives")?,
            parameters: state.runtime.buffer::<ObjectiveParams>(1),
            report: state
                .runtime
                .buffer_with(&[ObjectiveSelectionReport::default()]),
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
    /// Opt in without changing the scalar UCB/Thompson kernel or its ABI.
    /// Passing None restores scalar acquisition without changing observations.
    pub fn configure_objectives(mut self, config: Option<ObjectivePolicy>) -> Result<Self, String> {
        self.check_idle()?;
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
            let source_center = match morbo.regions[source].center.clone() {
                Some(c) => c,
                None => {
                    selector.morbo = Some(morbo);
                    self.objective_selector = Some(selector);
                    return Err("MORBO inactive region lost its resident center".into());
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
            if std::ptr::eq::<metal::BufferRef>(&**self.base, &**source_center)
                || self
                    .history_rows
                    .iter()
                    .any(|row| std::ptr::eq::<metal::BufferRef>(&***row, &**self.base))
            {
                self.base = Arc::new(self.runtime.buffer::<u16>(self.dimensions));
            }
            if let Err(e) = self.cow_base() {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
            if let Err(e) = self.copy(&*source_center, &*self.base) {
                selector.morbo = Some(morbo);
                self.objective_selector = Some(selector);
                return Err(e);
            }
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
            self.copy(&*self.base, &isolated)?;
            self.base = Arc::new(isolated);
        }
        self.cow_base()?;
        Ok(())
    }

    /// Restore the resident Pareto member with the largest exclusive exact
    /// hypervolume contribution for validation and checkpoint export.
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

    /// Read back only bounded candidate acquisition vectors, never model
    /// weights.
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
}

#[cfg(test)]
mod tests;
