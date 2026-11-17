//! Local surrogate ownership for modeled multi-region search.
//! The global posterior remains the public observation ledger; local models
//! receive only relevant observations and score their own candidate segments.
use std::sync::Arc;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use crate::backend::EnnStorage;
use crate::config::SurrogateConfig;
use crate::error::ENNError;
use crate::optimizer::MultiTrustRegionState;
use crate::optimizer::multi_tr::RestartMemoryPolicy;
use crate::scratch::ImLog;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, ENNSurrogateConfig, Surrogate};

/// Construct an independent model for one region generation.
pub type RegionalModelFactory =
    Box<dyn Fn(usize, u64) -> Result<BoxedSurrogate, ENNError> + Send + Sync>;

type RegionalSlotFactory = Box<dyn Fn(usize, u64) -> Result<RegionalSlot, ENNError> + Send + Sync>;

#[derive(Clone)]
enum RegionalSlot {
    Enn(Arc<ENNSurrogate>),
    External(Arc<BoxedSurrogate>),
}

impl RegionalSlot {
    fn observation_count(&self) -> Option<usize> {
        match self {
            Self::Enn(s) => s.observation_count(),
            Self::External(s) => s.observation_count(),
        }
    }

    fn as_surrogate(&self) -> &(dyn Surrogate + Send + Sync) {
        match self {
            Self::Enn(s) => s.as_ref(),
            Self::External(s) => s.as_ref().as_ref(),
        }
    }

    fn as_enn(&self) -> Option<&ENNSurrogate> {
        match self {
            Self::Enn(s) => Some(s.as_ref()),
            Self::External(_) => None,
        }
    }

    fn with_fit(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Enn(s) => {
                let inner = Arc::try_unwrap(s).unwrap_or_else(|arc| (*arc).clone());
                Ok(Self::Enn(Arc::new(inner.fit(x, y, yvar, rng)?)))
            }
            Self::External(s) => {
                let boxed = match Arc::try_unwrap(s) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                Ok(Self::External(Arc::new(boxed.with_fit(x, y, yvar, rng)?)))
            }
        }
    }

    fn with_append(
        self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        match self {
            Self::Enn(s) => {
                let inner = Arc::try_unwrap(s).unwrap_or_else(|arc| (*arc).clone());
                Ok(Self::Enn(Arc::new(inner.append(x, y, yvar, rng)?)))
            }
            Self::External(s) => {
                let boxed = match Arc::try_unwrap(s) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                Ok(Self::External(Arc::new(
                    boxed.with_append(x, y, yvar, rng)?,
                )))
            }
        }
    }

    fn cleared(self) -> Result<Self, ENNError> {
        match self {
            Self::Enn(s) => {
                let inner = Arc::try_unwrap(s).unwrap_or_else(|arc| (*arc).clone());
                Ok(Self::Enn(Arc::new(inner.cleared()?)))
            }
            Self::External(s) => {
                let boxed = match Arc::try_unwrap(s) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                Ok(Self::External(Arc::new(boxed.cleared()?)))
            }
        }
    }

    fn with_thawed(self, bytes: &[u8], pos: &mut usize) -> Result<Self, ENNError> {
        match self {
            Self::Enn(s) => {
                let inner = Arc::try_unwrap(s).unwrap_or_else(|arc| (*arc).clone());
                Ok(Self::Enn(Arc::new(inner.with_thawed(bytes, pos)?)))
            }
            Self::External(s) => {
                let boxed = match Arc::try_unwrap(s) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                Ok(Self::External(Arc::new(boxed.with_thawed(bytes, pos)?)))
            }
        }
    }

    fn freeze_surrogate(&self, buf: &mut Vec<u8>) -> Result<(), ENNError> {
        match self {
            Self::Enn(s) => <ENNSurrogate as Surrogate>::freeze_surrogate(s.as_ref(), buf),
            Self::External(s) => s.freeze_surrogate(buf),
        }
    }

    fn schedule_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::Enn(s) => s.schedule_flush(),
            Self::External(s) => s.schedule_flush(),
        }
    }

    fn wait_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::Enn(s) => s.wait_flush(),
            Self::External(s) => s.wait_flush(),
        }
    }

    fn release_pages(&self) -> Result<(), ENNError> {
        match self {
            Self::Enn(s) => s.release_pages(),
            Self::External(s) => s.release_pages(),
        }
    }
}

#[derive(Clone, Default)]
struct SlotHistory {
    x: ImLog,
    y: ImLog,
    yvar: Option<ImLog>,
}

impl SlotHistory {
    fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn append(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, ENNError> {
        if x.nrows() == 0 {
            return Ok(self);
        }
        let prev_rows = self.x.nrows();
        let num_metrics = y.ncols();
        self.x = self.x.push_rows(x)?;
        self.y = self.y.push_rows(y)?;
        self.yvar = match (self.yvar, yvar) {
            (Some(log), Some(yv)) => Some(log.push_rows(yv)?),
            (None, Some(yv)) => {
                if prev_rows == 0 {
                    Some(ImLog::with_cols(num_metrics).push_rows(yv)?)
                } else {
                    None
                }
            }
            (Some(_), None) | (None, None) => None,
        };
        Ok(self)
    }

    fn replay(&self, model: RegionalSlot, rng: &mut dyn RngCore) -> Result<RegionalSlot, ENNError> {
        if self.x.is_empty() {
            return Ok(model);
        }
        let x = self.x.to_array();
        let y = self.y.to_array();
        let variances = self.yvar.as_ref().map(ImLog::to_array);
        let variance_view = variances.as_ref().map(|v| v.view());
        model.with_fit(&x.view(), &y.view(), variance_view.as_ref(), rng)
    }
}

pub struct RegionalModels {
    models: Arc<Vec<RegionalSlot>>,
    factory: Arc<RegionalSlotFactory>,
    generations: Arc<Vec<u64>>,
    owned_disk: bool,
    history: Arc<Vec<SlotHistory>>,
    restart_memory: RestartMemoryPolicy,
}

impl RegionalModels {
    pub fn built_in(
        config: &SurrogateConfig,
        state: &MultiTrustRegionState,
    ) -> Result<Option<Self>, ENNError> {
        let SurrogateConfig::ENN(enn) = config else {
            return Ok(None);
        };
        let config: ENNSurrogateConfig = enn.clone();
        let owned_disk = config.storage == EnnStorage::Disk;
        #[cfg(not(all(feature = "disk-bpann", not(target_arch = "wasm32"))))]
        if owned_disk {
            return Err(crate::backend::disk_unavailable());
        }
        let root = if owned_disk {
            Some(config.work_dir.clone().ok_or_else(|| {
                ENNError::InvalidParameter("regional disk ENN requires explicit work_dir".into())
            })?)
        } else {
            None
        };
        let slot_factory: RegionalSlotFactory = Box::new(move |slot, generation| {
            let mut local = config.clone();
            if let Some(root) = &root {
                local.work_dir = Some(
                    root.join("regions")
                        .join(format!("slot-{slot}-generation-{generation}")),
                );
            }
            Ok(RegionalSlot::Enn(Arc::new(ENNSurrogate::new(local))))
        });
        let mut models = Self::with_slots(state, slot_factory)?;
        models.owned_disk = owned_disk;
        Ok(Some(models))
    }

    pub fn from_factory(
        state: &MultiTrustRegionState,
        factory: RegionalModelFactory,
    ) -> Result<Self, ENNError> {
        let slot_factory: RegionalSlotFactory = Box::new(move |slot, generation| {
            Ok(RegionalSlot::External(Arc::new(factory(slot, generation)?)))
        });
        Self::with_slots(state, slot_factory)
    }

    fn with_slots(
        state: &MultiTrustRegionState,
        factory: RegionalSlotFactory,
    ) -> Result<Self, ENNError> {
        let generations = (0..state.num_regions())
            .map(|slot| state.generation(slot).unwrap())
            .collect::<Vec<_>>();
        let models = generations
            .iter()
            .enumerate()
            .map(|(slot, &generation)| factory(slot, generation))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            models: Arc::new(models),
            factory: Arc::new(factory),
            generations: Arc::new(generations),
            owned_disk: false,
            history: Arc::new(
                (0..state.num_regions())
                    .map(|_| SlotHistory::default())
                    .collect(),
            ),
            restart_memory: state.restart_memory(),
        })
    }

    pub fn model(&self, slot: usize) -> Option<&(dyn Surrogate + Send + Sync)> {
        self.models.get(slot).and_then(|model| {
            (model.observation_count().unwrap_or(0) > 0).then_some(model.as_surrogate())
        })
    }

    pub fn enn_model(&self, slot: usize) -> Option<&ENNSurrogate> {
        self.models.get(slot).and_then(|model| {
            (model.observation_count().unwrap_or(0) > 0)
                .then(|| model.as_enn())
                .flatten()
        })
    }

    pub fn slot_count(&self) -> usize {
        self.models.len()
    }

    pub fn try_clone(&self) -> Result<Self, ENNError> {
        Ok(Self {
            models: Arc::clone(&self.models),
            factory: Arc::clone(&self.factory),
            generations: Arc::clone(&self.generations),
            owned_disk: false,
            history: Arc::clone(&self.history),
            restart_memory: self.restart_memory,
        })
    }

    pub fn fork(&self) -> Result<Self, ENNError> {
        self.try_clone()
    }

    /// A restarted region starts a new local trajectory. The global model can
    /// score candidates until new observations fit its local model.
    pub fn sync(
        mut self,
        state: &MultiTrustRegionState,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let any_changed = (0..self.models.len())
            .any(|slot| state.generation(slot).unwrap() != self.generations[slot]);
        if !any_changed {
            return Ok(self);
        }
        let models_mut = Arc::make_mut(&mut self.models);
        let gens_mut = Arc::make_mut(&mut self.generations);
        let hist_mut = Arc::make_mut(&mut self.history);
        for slot in 0..models_mut.len() {
            let generation = state.generation(slot).unwrap();
            if generation != gens_mut[slot] {
                let mut replacement = (self.factory)(slot, generation)?;
                if self.restart_memory == RestartMemoryPolicy::RetainRelevant {
                    replacement = hist_mut[slot].replay(replacement, rng)?;
                } else {
                    hist_mut[slot].clear();
                }
                let old_model = std::mem::replace(&mut models_mut[slot], replacement);
                if self.owned_disk {
                    let _ = old_model.cleared()?;
                }
                gens_mut[slot] = generation;
            }
        }
        Ok(self)
    }

    pub fn counts(&self) -> Vec<usize> {
        self.models
            .iter()
            .map(|model| model.observation_count().unwrap_or(0))
            .collect()
    }

    pub(crate) fn freeze_models(&self, buf: &mut Vec<u8>) -> Result<(), ENNError> {
        use crate::snapshot_io::{write_bool, write_len, write_mat, write_word};
        write_len(buf, self.models.len());
        for slot in 0..self.models.len() {
            write_word(buf, self.generations[slot]);
            self.models[slot].freeze_surrogate(buf)?;
            let hist = &self.history[slot];
            if hist.is_empty() {
                write_len(buf, 0);
            } else {
                write_len(buf, 1);
                write_mat(buf, &hist.x.to_array());
                write_mat(buf, &hist.y.to_array());
                match &hist.yvar {
                    Some(yv) => {
                        write_bool(buf, true);
                        write_mat(buf, &yv.to_array());
                    }
                    None => write_bool(buf, false),
                }
            }
        }
        Ok(())
    }

    pub(crate) fn thaw_models(
        mut self,
        num_dim: usize,
        bytes: &[u8],
        pos: &mut usize,
    ) -> Result<Self, ENNError> {
        use crate::snapshot_io::{MAX_METRICS, MAX_OBS, MAX_REGIONS, WireReader, bad_state};
        let mut reader = WireReader::from_slice(bytes, *pos);
        let slot_count = reader.read_len(MAX_REGIONS)?;
        if slot_count != self.models.len() {
            return Err(bad_state("frozen RegionalModels slot count mismatch"));
        }
        let mut new_models = Vec::new();
        new_models
            .try_reserve_exact(slot_count)
            .map_err(|_| bad_state("allocation failed"))?;
        let mut new_gens = Vec::new();
        new_gens
            .try_reserve_exact(slot_count)
            .map_err(|_| bad_state("allocation failed"))?;
        let mut new_hist = Vec::new();
        new_hist
            .try_reserve_exact(slot_count)
            .map_err(|_| bad_state("allocation failed"))?;

        for slot in 0..slot_count {
            let generation = reader.read_word()?;
            let slot_model = (self.factory)(slot, generation)?;
            let mut cur_pos = reader.position();
            let thawed_model = slot_model.with_thawed(bytes, &mut cur_pos)?;
            reader = WireReader::from_slice(bytes, cur_pos);

            let chunk_count = reader.read_len(MAX_OBS)?;
            let mut hist = SlotHistory::default();
            for _ in 0..chunk_count {
                let x = reader.read_mat(MAX_OBS, num_dim, true)?;
                let y = reader.read_mat(x.nrows(), MAX_METRICS, true)?;
                if x.ncols() != num_dim || y.nrows() != x.nrows() || y.ncols() == 0 {
                    return Err(bad_state("invalid ModelHistory chunk shape"));
                }
                let yvar = if reader.read_bool()? {
                    let yv = reader.read_mat(x.nrows(), y.ncols(), true)?;
                    if yv.shape() != y.shape() || yv.iter().any(|&v| v < 0.0) {
                        return Err(bad_state("invalid ModelHistory yvar chunk"));
                    }
                    Some(yv)
                } else {
                    None
                };
                let yvar_view = yvar.as_ref().map(|v| v.view());
                hist = hist.append(&x.view(), &y.view(), yvar_view.as_ref())?;
            }
            new_models.push(thawed_model);
            new_gens.push(generation);
            new_hist.push(hist);
        }

        if self.owned_disk {
            let old_models = Arc::try_unwrap(std::mem::take(&mut self.models))
                .unwrap_or_else(|arc| (*arc).clone());
            for model in old_models {
                let _ = model.cleared();
            }
        }
        self.models = Arc::new(new_models);
        self.generations = Arc::new(new_gens);
        self.history = Arc::new(new_hist);
        *pos = reader.position();
        Ok(self)
    }

    pub(crate) fn hash_models(&self, buf: &mut Vec<u8>) {
        let _ = self.freeze_models(buf);
    }

    /// Warm-up data has no generating region and seeds every local model in
    /// parallel via [`std::thread::scope`] and counter-folded per-slot PRNGs.
    pub fn warmup(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let mut seeds = Vec::with_capacity(self.models.len());
        for slot in 0..self.models.len() {
            let seed = if self.owned_disk {
                rng.next_u64()
            } else {
                crate::hash::stream_seed(self.generations[slot], slot as u64)
            };
            seeds.push(seed);
        }
        let models =
            Arc::try_unwrap(std::mem::take(&mut self.models)).unwrap_or_else(|arc| (*arc).clone());
        #[cfg(target_arch = "wasm32")]
        {
            let mut updated_models = Vec::with_capacity(models.len());
            for (model, seed) in models.into_iter().zip(seeds) {
                updated_models.push(fit_slot(model, x, y, yvar, seed)?);
            }
            self.models = Arc::new(updated_models);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let updated_models =
                std::thread::scope(|scope| -> Result<Vec<RegionalSlot>, ENNError> {
                    let mut handles = Vec::with_capacity(models.len());
                    for (model, seed) in models.into_iter().zip(seeds) {
                        handles.push(scope.spawn(move || fit_slot(model, x, y, yvar, seed)));
                    }
                    let mut results = Vec::with_capacity(handles.len());
                    for handle in handles {
                        results.push(handle.join().map_err(|_| {
                            ENNError::InvalidParameter("regional warmup thread panicked".into())
                        })??);
                    }
                    Ok(results)
                })?;
            self.models = Arc::new(updated_models);
        }
        if self.restart_memory == RestartMemoryPolicy::RetainRelevant {
            let hist_mut = Arc::make_mut(&mut self.history);
            for slot_hist in hist_mut.iter_mut() {
                let taken = std::mem::take(slot_hist);
                *slot_hist = taken.append(x, y, yvar)?;
            }
        }
        Ok(self)
    }

    /// Fit each regional model on the subset selected by the sharing policy in
    /// parallel across disjoint region slots via [`std::thread::scope`].
    pub fn route(
        mut self,
        state: &MultiTrustRegionState,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        origins: Option<&[usize]>,
        lengthscales: Option<&ArrayView1<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        self = self.sync(state, rng)?;
        let retain = self.restart_memory == RestartMemoryPolicy::RetainRelevant;
        let mut slot_plans: Vec<Option<(Vec<usize>, u64)>> = Vec::with_capacity(self.models.len());
        for slot in 0..self.models.len() {
            if !state.active_mask[slot] {
                slot_plans.push(None);
                continue;
            }
            let (lower, upper) = state.compute_bounds(slot, lengthscales);
            let mut rows = Vec::new();
            for row in 0..x.nrows() {
                if state.accepts(slot, row, &x.row(row), &lower, &upper, origins) {
                    rows.push(row);
                }
            }
            if rows.is_empty() {
                slot_plans.push(None);
                continue;
            }
            let seed = if self.owned_disk {
                rng.next_u64()
            } else {
                crate::hash::stream_seed(self.generations[slot], slot as u64)
            };
            slot_plans.push(Some((rows, seed)));
        }

        let models =
            Arc::try_unwrap(std::mem::take(&mut self.models)).unwrap_or_else(|arc| (*arc).clone());

        #[cfg(target_arch = "wasm32")]
        {
            let mut updated_models = Vec::with_capacity(models.len());
            for (slot, (model, plan)) in models.into_iter().zip(slot_plans).enumerate() {
                if let Some((rows, seed)) = plan {
                    let (updated, local_batch) = route_slot(model, x, y, yvar, &rows, seed)?;
                    updated_models.push(updated);
                    if retain {
                        if let Some((lx, ly, lyv)) = local_batch {
                            let hist_mut = Arc::make_mut(&mut self.history);
                            let taken = std::mem::take(&mut hist_mut[slot]);
                            let yv_view = lyv.as_ref().map(|v| v.view());
                            hist_mut[slot] =
                                taken.append(&lx.view(), &ly.view(), yv_view.as_ref())?;
                        }
                    }
                } else {
                    updated_models.push(model);
                }
            }
            self.models = Arc::new(updated_models);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            type RoutedBatch = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);
            let results = std::thread::scope(
                |scope| -> Result<Vec<(RegionalSlot, Option<RoutedBatch>)>, ENNError> {
                    let mut handles = Vec::with_capacity(models.len());
                    for (model, plan) in models.into_iter().zip(slot_plans) {
                        if let Some((rows, seed)) = plan {
                            handles.push(
                                scope.spawn(move || route_slot(model, x, y, yvar, &rows, seed)),
                            );
                        } else {
                            handles.push(scope.spawn(move || Ok((model, None))));
                        }
                    }
                    let mut res = Vec::with_capacity(handles.len());
                    for handle in handles {
                        res.push(handle.join().map_err(|_| {
                            ENNError::InvalidParameter("regional route thread panicked".into())
                        })??);
                    }
                    Ok(res)
                },
            )?;
            let mut updated_models = Vec::with_capacity(results.len());
            for (slot, (model, local_batch)) in results.into_iter().enumerate() {
                updated_models.push(model);
                if retain {
                    if let Some((lx, ly, lyv)) = local_batch {
                        let hist_mut = Arc::make_mut(&mut self.history);
                        let taken = std::mem::take(&mut hist_mut[slot]);
                        let yv_view = lyv.as_ref().map(|v| v.view());
                        hist_mut[slot] = taken.append(&lx.view(), &ly.view(), yv_view.as_ref())?;
                    }
                }
            }
            self.models = Arc::new(updated_models);
        }

        Ok(self)
    }

    pub fn with_flushed(self) -> Result<Self, ENNError> {
        for model in self.models.iter() {
            model.schedule_flush()?;
            model.wait_flush()?;
        }
        Ok(self)
    }

    pub fn with_released(self) -> Result<Self, ENNError> {
        for model in self.models.iter() {
            model.release_pages()?;
        }
        Ok(self)
    }
}

impl Drop for RegionalModels {
    fn drop(&mut self) {
        if self.owned_disk {
            let models = Arc::try_unwrap(std::mem::take(&mut self.models))
                .unwrap_or_else(|arc| (*arc).clone());
            for model in models {
                if let Err(error) = model.cleared() {
                    eprintln!("baux: regional disk ENN cleanup failed: {error}");
                }
            }
        }
    }
}

impl Clone for RegionalModels {
    fn clone(&self) -> Self {
        self.try_clone().expect("RegionalModels clone failed")
    }
}

const _: () = {
    fn assert_sync<T: Send + Sync>() {}
    fn check() {
        assert_sync::<RegionalModels>();
    }
};

fn fit_slot(
    model: RegionalSlot,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    seed: u64,
) -> Result<RegionalSlot, ENNError> {
    use rand::SeedableRng;
    let mut slot_rng = rand::rngs::StdRng::seed_from_u64(seed);
    model.with_append(x, y, yvar, &mut slot_rng)
}

fn route_slot(
    model: RegionalSlot,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rows: &[usize],
    seed: u64,
) -> Result<
    (
        RegionalSlot,
        Option<(Array2<f64>, Array2<f64>, Option<Array2<f64>>)>,
    ),
    ENNError,
> {
    use rand::SeedableRng;
    let local_x: Array2<f64> = x.select(Axis(0), rows);
    let local_y: Array2<f64> = y.select(Axis(0), rows);
    let local_var = yvar.map(|values| values.select(Axis(0), rows));
    let variance_view = local_var.as_ref().map(|values| values.view());
    let mut slot_rng = rand::rngs::StdRng::seed_from_u64(seed);
    let updated = model.with_append(
        &local_x.view(),
        &local_y.view(),
        variance_view.as_ref(),
        &mut slot_rng,
    )?;
    Ok((updated, Some((local_x, local_y, local_var))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::optimizer::multi_tr::RestartMemoryPolicy;
    use crate::optimizer::{MultiTrustRegionConfig, SharingPolicy};
    use crate::trust_region::TRLengthConfig;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn model_routing() {
        let x = array![[0.1], [0.45], [0.9]];
        let y = array![[0.1], [0.5], [0.9]];
        let centers = array![[0.25], [0.75]];
        let origins = [1, 0, 1];
        for (policy, expected) in [
            (SharingPolicy::BoxContain, [2, 2]),
            (SharingPolicy::EllipsoidBall, [2, 2]),
            (SharingPolicy::OriginOnly, [1, 2]),
            (SharingPolicy::NearestCenter, [2, 1]),
        ] {
            let mut rng = StdRng::seed_from_u64(7);
            let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
            config.sharing_policy = policy;
            let mut state =
                MultiTrustRegionState::new(1, config, Some(&centers.view()), &mut rng).unwrap();
            let mut models = RegionalModels::built_in(&SurrogateConfig::default(), &state)
                .unwrap()
                .unwrap();
            models = models
                .route(
                    &state,
                    &x.view(),
                    &y.view(),
                    None,
                    Some(&origins),
                    None,
                    &mut rng,
                )
                .unwrap();
            assert_eq!(models.counts(), expected, "{policy:?}");
            state.restart_region(0, &array![0.3].view()).unwrap();
            models = models.sync(&state, &mut rng).unwrap();
            assert_eq!(models.counts(), [0, expected[1]], "{policy:?}");
            assert!(models.model(0).is_none());
        }
    }

    #[test]
    fn retainedhistory() {
        let mut rng = StdRng::seed_from_u64(11);
        let centers = array![[0.25], [0.75]];
        let mut config = MultiTrustRegionConfig::new(2, TRLengthConfig::default());
        config.restart_memory_policy = RestartMemoryPolicy::RetainRelevant;
        config.sharing_policy = SharingPolicy::OriginOnly;
        let mut state =
            MultiTrustRegionState::new(1, config, Some(&centers.view()), &mut rng).unwrap();
        let provider_config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            ..Default::default()
        };
        let factory: RegionalModelFactory = Box::new(move |_, _| {
            Ok(Box::new(ENNSurrogate::new(provider_config.clone())) as BoxedSurrogate)
        });
        let mut models = RegionalModels::from_factory(&state, factory).unwrap();
        let x = array![[0.2], [0.8]];
        let y = array![[1.0], [2.0]];
        models = models
            .route(
                &state,
                &x.view(),
                &y.view(),
                None,
                Some(&[0, 1]),
                None,
                &mut rng,
            )
            .unwrap();
        assert_eq!(models.counts(), [1, 1]);

        state.restart_region(0, &array![0.3].view()).unwrap();
        models = models.sync(&state, &mut rng).unwrap();
        assert_eq!(models.counts(), [1, 1]);
        assert!(models.model(0).is_some());
    }

    #[test]
    #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
    fn disk_regions() {
        let directory = tempfile::TempDir::new().unwrap();
        let root = directory.path();
        let config = ENNSurrogateConfig {
            k: 2,
            num_candidates: 2,
            num_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(root.to_path_buf()),
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        let centers = array![[0.25], [0.75]];
        let mut state = MultiTrustRegionState::new(
            1,
            MultiTrustRegionConfig::new(2, TRLengthConfig::default()),
            Some(&centers.view()),
            &mut rng,
        )
        .unwrap();
        let older = root.join("regions/slot-0-generation-0/num_obs.bin");
        let other = root.join("regions/slot-1-generation-0/num_obs.bin");
        {
            let mut models = RegionalModels::built_in(&SurrogateConfig::ENN(config), &state)
                .unwrap()
                .unwrap();
            let x = array![[0.1], [0.45], [0.9]];
            let y = array![[0.1], [0.5], [0.9]];
            models = models.warmup(&x.view(), &y.view(), None, &mut rng).unwrap();
            assert!(older.exists());
            assert!(other.exists());
            state.restart_region(0, &array![0.3].view()).unwrap();
            let _models = models.sync(&state, &mut rng).unwrap();
            assert!(!older.exists());
            assert!(other.exists());
        }
        assert!(!other.exists());
    }
}
