//! Local surrogate ownership for modeled multi-region search.
//! The global posterior remains the public observation ledger; local models
//! receive only relevant observations and score their own candidate segments.
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rand::RngCore;

use crate::backend::EnnStorage;
use crate::config::SurrogateConfig;
use crate::error::ENNError;
use crate::optimizer::MultiTrustRegionState;
use crate::optimizer::multi_tr::RestartMemoryPolicy;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, ENNSurrogateConfig, Surrogate};

/// Construct an independent model for one region generation.
pub type RegionalModelFactory =
    Box<dyn Fn(usize, u64) -> Result<BoxedSurrogate, ENNError> + Send + Sync>;

pub struct RegionalModels {
    models: Vec<BoxedSurrogate>,
    factory: RegionalModelFactory,
    generations: Vec<u64>,
    owned_disk: bool,
    history: Vec<Vec<ModelHistory>>,
    restart_memory: RestartMemoryPolicy,
}

struct ModelHistory {
    x: Array2<f64>,
    y: Array2<f64>,
    yvar: Option<Array2<f64>>,
}

impl ModelHistory {
    fn replay(
        chunks: &[Self],
        model: &mut dyn Surrogate,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        if chunks.is_empty() {
            return Ok(());
        }
        let x = ndarray::concatenate(
            Axis(0),
            &chunks
                .iter()
                .map(|chunk| chunk.x.view())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        let y = ndarray::concatenate(
            Axis(0),
            &chunks
                .iter()
                .map(|chunk| chunk.y.view())
                .collect::<Vec<_>>(),
        )
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        let variances = if chunks.iter().all(|chunk| chunk.yvar.is_some()) {
            Some(
                ndarray::concatenate(
                    Axis(0),
                    &chunks
                        .iter()
                        .filter_map(|chunk| chunk.yvar.as_ref().map(|values| values.view()))
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| ENNError::InvalidParameter(error.to_string()))?,
            )
        } else {
            None
        };
        let variance_view = variances.as_ref().map(|values| values.view());
        model.fit(&x.view(), &y.view(), variance_view.as_ref(), rng)
    }
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
        let root = if owned_disk {
            Some(
                config
                    .work_dir
                    .clone()
                    .or_else(EnnStorage::work_env)
                    .ok_or_else(|| {
                        ENNError::InvalidParameter(
                            "regional disk ENN requires work_dir or BAUX_WORK_DIR".into(),
                        )
                    })?,
            )
        } else {
            None
        };
        let factory: RegionalModelFactory = Box::new(move |slot, generation| {
            let mut local = config.clone();
            if let Some(root) = &root {
                local.work_dir = Some(
                    root.join("regions")
                        .join(format!("slot-{slot}-generation-{generation}")),
                );
            }
            Ok(Box::new(ENNSurrogate::new(local)) as BoxedSurrogate)
        });
        let mut models = Self::from_factory(state, factory)?;
        models.owned_disk = owned_disk;
        Ok(Some(models))
    }

    pub fn from_factory(
        state: &MultiTrustRegionState,
        factory: RegionalModelFactory,
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
            models,
            factory,
            generations,
            owned_disk: false,
            history: (0..state.num_regions()).map(|_| Vec::new()).collect(),
            restart_memory: state.restart_memory(),
        })
    }

    pub fn model(&self, slot: usize) -> Option<&(dyn Surrogate + Send + Sync)> {
        self.models.get(slot).and_then(|model| {
            (model.observation_count().unwrap_or(0) > 0).then_some(model.as_ref())
        })
    }

    pub fn slot_count(&self) -> usize {
        self.models.len()
    }

    /// A restarted region starts a new local trajectory. The global model can
    /// score candidates until new observations fit its local model.
    pub fn sync(
        &mut self,
        state: &MultiTrustRegionState,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        for slot in 0..self.models.len() {
            let generation = state.generation(slot).unwrap();
            if generation != self.generations[slot] {
                let replacement = (self.factory)(slot, generation)?;
                if self.owned_disk {
                    self.models[slot].clear_observations()?;
                }
                self.models[slot] = replacement;
                self.generations[slot] = generation;
                if self.restart_memory == RestartMemoryPolicy::RetainRelevant {
                    ModelHistory::replay(&self.history[slot], self.models[slot].as_mut(), rng)?;
                } else {
                    self.history[slot].clear();
                }
            }
        }
        Ok(())
    }

    pub fn counts(&self) -> Vec<usize> {
        self.models
            .iter()
            .map(|model| model.observation_count().unwrap_or(0))
            .collect()
    }

    /// Warm-up data has no generating region and seeds every local model in
    /// parallel via [`std::thread::scope`] and counter-folded per-slot PRNGs.
    pub fn warmup(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let mut seeds = Vec::with_capacity(self.models.len());
        for slot in 0..self.models.len() {
            let seed = if self.owned_disk {
                rng.next_u64()
            } else {
                crate::hash::stream_seed(self.generations[slot], slot as u64)
            };
            seeds.push(seed);
        }
        std::thread::scope(|scope| -> Result<(), ENNError> {
            let mut handles = Vec::with_capacity(self.models.len());
            for (model, seed) in self.models.iter_mut().zip(seeds) {
                handles.push(scope.spawn(move || fit_slot(model.as_mut(), x, y, yvar, seed)));
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    ENNError::InvalidParameter("regional warmup thread panicked".into())
                })??;
            }
            Ok(())
        })?;
        if self.restart_memory == RestartMemoryPolicy::RetainRelevant {
            for history in &mut self.history {
                history.push(ModelHistory {
                    x: x.to_owned(),
                    y: y.to_owned(),
                    yvar: yvar.map(|values| values.to_owned()),
                });
            }
        }
        Ok(())
    }

    /// Fit each regional model on the subset selected by the sharing policy in
    /// parallel across disjoint region slots via [`std::thread::scope`].
    pub fn route(
        &mut self,
        state: &MultiTrustRegionState,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        origins: Option<&[usize]>,
        lengthscales: Option<&ArrayView1<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        self.sync(state, rng)?;
        let retain = self.restart_memory == RestartMemoryPolicy::RetainRelevant;
        let mut tasks: Vec<(&mut BoxedSurrogate, &mut Vec<ModelHistory>, Vec<usize>, u64)> =
            Vec::new();
        for (slot, (model, history)) in self
            .models
            .iter_mut()
            .zip(self.history.iter_mut())
            .enumerate()
        {
            if !state.active_mask[slot] {
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
                continue;
            }
            let seed = if self.owned_disk {
                rng.next_u64()
            } else {
                crate::hash::stream_seed(self.generations[slot], slot as u64)
            };
            tasks.push((model, history, rows, seed));
        }
        std::thread::scope(|scope| -> Result<(), ENNError> {
            let mut handles = Vec::with_capacity(tasks.len());
            for (model, history, rows, seed) in tasks {
                handles.push(scope.spawn(move || {
                    route_slot(model.as_mut(), history, x, y, yvar, &rows, seed, retain)
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    ENNError::InvalidParameter("regional route thread panicked".into())
                })??;
            }
            Ok(())
        })
    }
}

impl Drop for RegionalModels {
    fn drop(&mut self) {
        if self.owned_disk {
            for model in &mut self.models {
                if let Err(error) = model.clear_observations() {
                    eprintln!("baux: regional disk ENN cleanup failed: {error}");
                }
            }
        }
    }
}

fn fit_slot(
    model: &mut dyn Surrogate,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    seed: u64,
) -> Result<(), ENNError> {
    use rand::SeedableRng;
    let mut slot_rng = rand::rngs::StdRng::seed_from_u64(seed);
    model.fit_append(x, y, yvar, &mut slot_rng)
}

#[allow(clippy::too_many_arguments)]
fn route_slot(
    model: &mut dyn Surrogate,
    history: &mut Vec<ModelHistory>,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rows: &[usize],
    seed: u64,
    retain: bool,
) -> Result<(), ENNError> {
    use rand::SeedableRng;
    let local_x: Array2<f64> = x.select(Axis(0), rows);
    let local_y: Array2<f64> = y.select(Axis(0), rows);
    let local_var = yvar.map(|values| values.select(Axis(0), rows));
    let variance_view = local_var.as_ref().map(|values| values.view());
    let mut slot_rng = rand::rngs::StdRng::seed_from_u64(seed);
    model.fit_append(
        &local_x.view(),
        &local_y.view(),
        variance_view.as_ref(),
        &mut slot_rng,
    )?;
    if retain {
        history.push(ModelHistory {
            x: local_x,
            y: local_y,
            yvar: local_var,
        });
    }
    Ok(())
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
            models
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
            models.sync(&state, &mut rng).unwrap();
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
        // Use the same provider factory path as supplied regional models.
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
        models
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
        models.sync(&state, &mut rng).unwrap();
        assert_eq!(models.counts(), [1, 1]);
        assert!(models.model(0).is_some());
    }

    #[test]
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
            models.warmup(&x.view(), &y.view(), None, &mut rng).unwrap();
            assert!(older.exists());
            assert!(other.exists());
            state.restart_region(0, &array![0.3].view()).unwrap();
            models.sync(&state, &mut rng).unwrap();
            assert!(!older.exists());
            assert!(other.exists());
        }
        assert!(!other.exists());
    }
}
