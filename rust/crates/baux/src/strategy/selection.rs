//! Unified segment-based acquisition selection over candidate pools.

use ndarray::{Array2, ArrayView2, s};
use rand::RngCore;
use rand::seq::SliceRandom;

use crate::acquisition::{ParetoAcquisition, RandomAcquisition, UCBAcquisition};
use crate::config::AcquisitionConfig;
use crate::error::ENNError;
use crate::optimizer::{BayesOptimizer, TrustRegionState};
use crate::util::argmax_tie;

#[derive(Clone, Copy)]
pub(super) struct CandidateSegment {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) arms: usize,
    pub(super) region: usize,
}

fn single_segment(nrows: usize, num_arms: usize) -> [CandidateSegment; 1] {
    [CandidateSegment {
        start: 0,
        end: nrows,
        arms: num_arms,
        region: 0,
    }]
}

#[allow(dead_code)]
pub(super) fn select_random(
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let indices = random_segments(&single_segment(x_cand.nrows(), num_arms), rng)?;
    Ok(select_indices(x_cand, &indices))
}

/// Select arms via Thompson sampling (posterior draw).
#[allow(dead_code)]
pub(super) fn select_thompson(
    optimizer: &BayesOptimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let indices = thompson_segments(
        optimizer,
        surrogate,
        x_cand,
        &single_segment(x_cand.nrows(), num_arms),
        rng,
    )?;
    Ok(select_indices(x_cand, &indices))
}

/// Select arms via UCB (upper confidence bound).
#[allow(dead_code)]
pub(super) fn select_ucb(
    optimizer: &BayesOptimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let indices = ucb_segments(
        optimizer,
        surrogate,
        x_cand,
        &single_segment(x_cand.nrows(), num_arms),
        beta,
        rng,
    )?;
    Ok(select_indices(x_cand, &indices))
}

/// Select arms via Pareto frontier.
#[allow(dead_code)]
pub(super) fn select_pareto(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let indices = pareto_segments(
        surrogate,
        x_cand,
        &single_segment(x_cand.nrows(), num_arms),
        rng,
    )?;
    Ok(select_indices(x_cand, &indices))
}

/// Select fixed arm counts from candidate segments after one surrogate pass.
pub(super) fn select_segment(
    optimizer: &BayesOptimizer,
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    if let TrustRegionState::Multi { state, .. } = optimizer.trust_region() {
        let mut chosen = Vec::new();
        for segment in segments {
            let model = optimizer
                .local_models
                .as_ref()
                .map(|models| {
                    if segment.region >= models.slot_count() {
                        return Err(ENNError::InvalidParameter(
                            "regional model slot is missing".into(),
                        ));
                    }
                    Ok(models.model(segment.region))
                })
                .transpose()?
                .flatten();
            let global = optimizer.surrogate().filter(|surrogate| {
                optimizer.posterior.obs_count() > 0
                    || surrogate.observation_count().is_some_and(|count| count > 0)
            });
            let surrogate = model.or(global);
            let acquisition = state.acquisition_for(segment.region, optimizer.config().acquisition);
            let local = candidates.slice(s![segment.start..segment.end, ..]);
            let indices = select_with(
                optimizer,
                surrogate,
                &local,
                &single_segment(local.nrows(), segment.arms),
                acquisition,
                rng,
            )?;
            chosen.extend(indices.into_iter().map(|index| segment.start + index));
        }
        return Ok(chosen);
    }
    let global = optimizer.surrogate().filter(|surrogate| {
        optimizer.posterior.obs_count() > 0
            || surrogate.observation_count().is_some_and(|count| count > 0)
    });
    select_with(
        optimizer,
        global,
        candidates,
        segments,
        optimizer.config().acquisition,
        rng,
    )
}

/// Select `num_arms` winners jointly across all regional candidate segments
/// (TuRBO-m).
pub(super) fn select_joint(
    optimizer: &BayesOptimizer,
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<(Vec<usize>, Vec<CandidateSegment>), ENNError> {
    let TrustRegionState::Multi { state, .. } = optimizer.trust_region() else {
        unreachable!("select_joint requires TrustRegionState::Multi");
    };
    let mut scored = Vec::with_capacity(candidates.nrows());
    for segment in segments {
        let local = candidates.slice(s![segment.start..segment.end, ..]);
        let scores = score_segment(optimizer, state, segment.region, &local, rng)?;
        for (offset, score) in scores.into_iter().enumerate() {
            scored.push((score, segment.region, segment.start + offset));
        }
    }
    scored.shuffle(rng);
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    let chosen = &scored[..num_arms.min(scored.len())];
    let mut indices = Vec::with_capacity(chosen.len());
    let mut winners = Vec::new();
    for segment in segments {
        let group: Vec<usize> = chosen
            .iter()
            .filter(|&&(_, region, _)| region == segment.region)
            .map(|&(_, _, idx)| idx)
            .collect();
        if !group.is_empty() {
            winners.push(CandidateSegment {
                start: segment.start,
                end: segment.end,
                arms: group.len(),
                region: segment.region,
            });
            indices.extend(group);
        }
    }
    Ok((indices, winners))
}

fn score_segment(
    optimizer: &BayesOptimizer,
    state: &crate::optimizer::MultiTrustRegionState,
    region: usize,
    local: &ArrayView2<f64>,
    rng: &mut dyn RngCore,
) -> Result<Vec<f64>, ENNError> {
    let model = optimizer
        .local_models
        .as_ref()
        .map(|models| {
            if region >= models.slot_count() {
                return Err(ENNError::InvalidParameter(
                    "regional model slot is missing".into(),
                ));
            }
            Ok(models.model(region))
        })
        .transpose()?
        .flatten();
    let global = optimizer.surrogate().filter(|surrogate| {
        optimizer.posterior.obs_count() > 0
            || surrogate.observation_count().is_some_and(|count| count > 0)
    });
    let surrogate = model.or(global);
    let acquisition = state.acquisition_for(region, optimizer.config().acquisition);
    score_with(optimizer, surrogate, local, acquisition, rng)
}

fn score_with(
    optimizer: &BayesOptimizer,
    surrogate: Option<&(dyn crate::surrogate::Surrogate + Send + Sync)>,
    local: &ArrayView2<f64>,
    acquisition: AcquisitionConfig,
    rng: &mut dyn RngCore,
) -> Result<Vec<f64>, ENNError> {
    let n = local.nrows();
    let Some(surrogate) = surrogate else {
        return Ok((0..n)
            .map(|_| rand::Rng::gen_range(rng, 0.0..1.0))
            .collect());
    };
    match acquisition {
        AcquisitionConfig::Random => Ok((0..n)
            .map(|_| rand::Rng::gen_range(rng, 0.0..1.0))
            .collect()),
        AcquisitionConfig::Thompson => {
            let samples = surrogate.sample(local, 1, rng)?;
            if optimizer.trust_region().is_morbo() {
                let scores = morbo_samples(optimizer, &samples, 1, n)?;
                return Ok(scores.to_vec());
            }
            Ok((0..n).map(|i| samples[[0, i, 0]]).collect())
        }
        AcquisitionConfig::UCB { beta } => {
            let pred = surrogate.predict(local)?;
            if optimizer.trust_region().is_morbo() {
                let values = &pred.mu + &(pred.se * beta);
                let scores = optimizer
                    .trust_region()
                    .morbo_scalarize(&values.view(), false)
                    .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
                return Ok(scores.to_vec());
            }
            Ok((0..n)
                .map(|i| pred.mu[[i, 0]] + beta * pred.se[[i, 0]])
                .collect())
        }
        AcquisitionConfig::Pareto => {
            let pred = surrogate.predict(local)?;
            Ok((0..n).map(|i| pred.mu[[i, 0]] + pred.se[[i, 0]]).collect())
        }
    }
}

fn select_with(
    optimizer: &BayesOptimizer,
    surrogate: Option<&(dyn crate::surrogate::Surrogate + Send + Sync)>,
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    acquisition: AcquisitionConfig,
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let Some(surrogate) = surrogate else {
        return random_segments(segments, rng);
    };
    match acquisition {
        AcquisitionConfig::Random => random_segments(segments, rng),
        AcquisitionConfig::Thompson => {
            thompson_segments(optimizer, surrogate, candidates, segments, rng)
        }
        AcquisitionConfig::UCB { beta } => {
            ucb_segments(optimizer, surrogate, candidates, segments, beta, rng)
        }
        AcquisitionConfig::Pareto => pareto_segments(surrogate, candidates, segments, rng),
    }
}

pub(super) fn random_segments(
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let mut selected = Vec::new();
    for segment in segments {
        let local = RandomAcquisition
            .select(segment.end - segment.start, segment.arms, rng)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

fn morbo_samples(
    optimizer: &BayesOptimizer,
    samples: &ndarray::Array3<f64>,
    max_arms: usize,
    n_cand: usize,
) -> Result<ndarray::Array1<f64>, ENNError> {
    let num_metrics = samples.shape()[2];
    let mut flat = Array2::zeros((max_arms * n_cand, num_metrics));
    for arm in 0..max_arms {
        for cand in 0..n_cand {
            for m in 0..num_metrics {
                flat[[arm * n_cand + cand, m]] = samples[[arm, cand, m]];
            }
        }
    }
    optimizer
        .trust_region()
        .morbo_scalarize(&flat.view(), false)
        .map_err(|error| ENNError::InvalidParameter(error.to_string()))
}

pub(super) fn thompson_segments(
    optimizer: &BayesOptimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let max_arms = segments.iter().map(|s| s.arms).max().unwrap_or(1);
    let samples = surrogate.sample(candidates, max_arms, rng)?;
    let n_cand = candidates.nrows();
    if optimizer.trust_region().is_morbo() {
        let flat_scores = morbo_samples(optimizer, &samples, max_arms, n_cand)?;
        let mut selected = Vec::new();
        for segment in segments {
            let mut local = Vec::with_capacity(segment.arms);
            for arm in 0..segment.arms {
                let mut values: Vec<f64> = (segment.start..segment.end)
                    .map(|cand| flat_scores[arm * n_cand + cand])
                    .collect();
                if local.len() < values.len() {
                    for &prev in &local {
                        values[prev] = f64::NEG_INFINITY;
                    }
                }
                local.push(argmax_tie(&values, rng));
            }
            selected.extend(local.into_iter().map(|index| segment.start + index));
        }
        return Ok(selected);
    }
    let mut selected = Vec::new();
    for segment in segments {
        let mut local = Vec::with_capacity(segment.arms);
        for arm in 0..segment.arms {
            let mut values: Vec<f64> = (segment.start..segment.end)
                .map(|cand| samples[[arm, cand, 0]])
                .collect();
            if local.len() < values.len() {
                for &prev in &local {
                    values[prev] = f64::NEG_INFINITY;
                }
            }
            local.push(argmax_tie(&values, rng));
        }
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

pub(super) fn ucb_segments(
    optimizer: &BayesOptimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let prediction = surrogate.predict(candidates)?;
    if optimizer.trust_region().is_morbo() {
        let values = &prediction.mu + &(prediction.se * beta);
        let scores = optimizer
            .trust_region()
            .morbo_scalarize(&values.view(), false)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        let mut selected = Vec::new();
        for segment in segments {
            let mut local = (segment.start..segment.end).collect::<Vec<_>>();
            local.shuffle(rng);
            local.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
            selected.extend(local.into_iter().take(segment.arms));
        }
        return Ok(selected);
    }
    let ucb = UCBAcquisition::new(beta);
    let mut selected = Vec::new();
    for segment in segments {
        let mu = prediction.mu.slice(s![segment.start..segment.end, 0]);
        let sigma = prediction.se.slice(s![segment.start..segment.end, 0]);
        let local = ucb
            .select(&mu, &sigma, segment.arms, rng)
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

pub(super) fn pareto_segments(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    candidates: &ArrayView2<f64>,
    segments: &[CandidateSegment],
    rng: &mut dyn RngCore,
) -> Result<Vec<usize>, ENNError> {
    let prediction = surrogate.predict(candidates)?;
    let pareto = ParetoAcquisition::new();
    let mut selected = Vec::new();
    for segment in segments {
        let local = pareto
            .select(
                &prediction.mu.slice(s![segment.start..segment.end, ..]),
                &prediction.se.slice(s![segment.start..segment.end, ..]),
                segment.arms,
                rng,
            )
            .map_err(|error| ENNError::InvalidParameter(error.to_string()))?;
        selected.extend(local.into_iter().map(|index| segment.start + index));
    }
    Ok(selected)
}

/// Select arms using acquisition function over a single region segment.
pub(super) fn select_arms(
    optimizer: &BayesOptimizer,
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let indices = select_segment(
        optimizer,
        x_cand,
        &single_segment(x_cand.nrows(), num_arms),
        rng,
    )?;
    Ok(select_indices(x_cand, &indices))
}

/// Select rows by indices.
pub(super) fn select_indices(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    use ndarray::Axis;
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i).to_owned()).collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .expect("stack should succeed for same-shaped rows")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ENNError;
    use crate::optimizer::{RegionalModelFactory, RegionalModels, Strategy, StrategyState};
    use crate::optimizer_config::{OptimizerConfig, SurrogateConfig};
    use crate::surrogate::{Surrogate, SurrogatePrediction};
    use ndarray::{Array1, Axis, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::sync::{Arc, Mutex};

    struct CallModel {
        slot: usize,
        observations: usize,
        calls: Arc<Mutex<[[usize; 2]; 2]>>,
    }

    impl Surrogate for CallModel {
        fn fit(
            &mut self,
            _: &ArrayView2<f64>,
            _: &ArrayView2<f64>,
            _: Option<&ArrayView2<f64>>,
            _: &mut dyn RngCore,
        ) -> Result<(), ENNError> {
            Ok(())
        }

        fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
            self.calls.lock().unwrap()[self.slot][0] += 1;
            Ok(SurrogatePrediction {
                mu: x.column(0).to_owned().insert_axis(Axis(1)),
                se: Array2::zeros((x.nrows(), 1)),
            })
        }

        fn sample(
            &self,
            x: &ArrayView2<f64>,
            num_samples: usize,
            _: &mut dyn RngCore,
        ) -> Result<ndarray::Array3<f64>, ENNError> {
            self.calls.lock().unwrap()[self.slot][1] += 1;
            let mut samples = ndarray::Array3::zeros((num_samples, x.nrows(), 1));
            for sample in 0..num_samples {
                for row in 0..x.nrows() {
                    samples[[sample, row, 0]] = x[[row, 0]];
                }
            }
            Ok(samples)
        }

        fn lengthscales(&self) -> Option<Array1<f64>> {
            None
        }

        fn observation_count(&self) -> Option<usize> {
            Some(self.observations)
        }
    }

    fn multi_optimizer(config: crate::optimizer::MultiTrustRegionConfig) -> BayesOptimizer {
        let mut rng = StdRng::seed_from_u64(29);
        let strategy_state = StrategyState::configured(1, config, 0, &mut rng).unwrap();
        BayesOptimizer::new_strategy(
            array![[0.0, 1.0]],
            OptimizerConfig {
                surrogate: SurrogateConfig::None,
                ..Default::default()
            },
            Strategy::Experimental(strategy_state),
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn empty_local() {
        let mut rng = StdRng::seed_from_u64(19);
        let strategy = Strategy::Experimental(StrategyState::new(1, 1, 0, &mut rng).unwrap());
        let optimizer = BayesOptimizer::new_strategy(
            array![[0.0, 1.0]],
            crate::turbo_enn(),
            strategy,
            &mut rng,
        )
        .unwrap();
        let candidates = array![[0.2], [0.8]];
        let segments = [CandidateSegment {
            start: 0,
            end: 2,
            arms: 1,
            region: 0,
        }];
        // The global ENN is also unfitted. Falling back to it would error.
        let selected = select_segment(&optimizer, &candidates.view(), &segments, &mut rng).unwrap();
        assert_eq!(selected.len(), 1);
        assert!(selected[0] < 2);
    }

    #[test]
    fn localacquisition() {
        let mut config = crate::optimizer::MultiTrustRegionConfig::new(2, Default::default());
        config.region_acquisitions[0] = Some(AcquisitionConfig::UCB { beta: 0.0 });
        config.region_acquisitions[1] = Some(AcquisitionConfig::Thompson);
        let mut optimizer = multi_optimizer(config);
        let calls = Arc::new(Mutex::new([[0, 0]; 2]));
        let factory: RegionalModelFactory = {
            let calls = Arc::clone(&calls);
            Box::new(move |slot, _| {
                Ok(Box::new(CallModel {
                    slot,
                    observations: 1,
                    calls: Arc::clone(&calls),
                }) as Box<dyn Surrogate + Send + Sync>)
            })
        };
        let state = match optimizer.trust_region() {
            TrustRegionState::Multi { state, .. } => state,
            _ => unreachable!(),
        };
        optimizer.local_models = Some(RegionalModels::from_factory(state, factory).unwrap());
        let candidates = array![[0.1], [0.9], [0.2], [0.8]];
        let segments = [
            CandidateSegment {
                start: 0,
                end: 2,
                arms: 1,
                region: 0,
            },
            CandidateSegment {
                start: 2,
                end: 4,
                arms: 1,
                region: 1,
            },
        ];
        select_segment(
            &optimizer,
            &candidates.view(),
            &segments,
            &mut StdRng::seed_from_u64(31),
        )
        .unwrap();
        assert_eq!(*calls.lock().unwrap(), [[1, 0], [0, 1]]);
    }

    #[test]
    fn globalfallback() {
        let calls = Arc::new(Mutex::new([[0, 0]; 2]));
        let mut rng = StdRng::seed_from_u64(37);
        let optimizer = BayesOptimizer::new_surrogate(
            array![[0.0, 1.0]],
            OptimizerConfig {
                surrogate: SurrogateConfig::None,
                acquisition: AcquisitionConfig::UCB { beta: 0.0 },
                ..Default::default()
            },
            Strategy::turbo(),
            Box::new(CallModel {
                slot: 0,
                observations: 1,
                calls: Arc::clone(&calls),
            }),
            &mut rng,
        )
        .unwrap();
        let candidates = array![[0.1], [0.9]];
        let segment = [CandidateSegment {
            start: 0,
            end: 2,
            arms: 1,
            region: 0,
        }];
        select_segment(&optimizer, &candidates.view(), &segment, &mut rng).unwrap();
        assert_eq!(*calls.lock().unwrap(), [[1, 0], [0, 0]]);

        let mut config = crate::optimizer::MultiTrustRegionConfig::new(1, Default::default());
        config.region_acquisitions[0] = Some(AcquisitionConfig::Thompson);
        let optimizer = multi_optimizer(config);
        let selected = select_segment(&optimizer, &candidates.view(), &segment, &mut rng).unwrap();
        assert_eq!(selected.len(), 1);
        assert!(selected[0] < candidates.nrows());
    }

    #[test]
    fn emptylocalglobal() {
        let mut multi = crate::optimizer::MultiTrustRegionConfig::new(1, Default::default());
        multi.region_acquisitions[0] = Some(AcquisitionConfig::UCB { beta: 0.0 });
        let mut rng = StdRng::seed_from_u64(41);
        let strategy =
            Strategy::Experimental(StrategyState::configured(1, multi, 0, &mut rng).unwrap());
        let global_calls = Arc::new(Mutex::new([[0, 0]; 2]));
        let local_calls = Arc::new(Mutex::new([[0, 0]; 2]));
        let local_factory: RegionalModelFactory = {
            let calls = Arc::clone(&local_calls);
            Box::new(move |slot, _| {
                Ok(Box::new(CallModel {
                    slot,
                    observations: 0,
                    calls: Arc::clone(&calls),
                }) as Box<dyn Surrogate + Send + Sync>)
            })
        };
        let optimizer = BayesOptimizer::new_regional(
            array![[0.0, 1.0]],
            OptimizerConfig {
                surrogate: SurrogateConfig::None,
                ..Default::default()
            },
            strategy,
            Box::new(CallModel {
                slot: 0,
                observations: 1,
                calls: Arc::clone(&global_calls),
            }),
            local_factory,
            &mut rng,
        )
        .unwrap();
        let candidates = array![[0.1], [0.9]];
        let segment = [CandidateSegment {
            start: 0,
            end: 2,
            arms: 1,
            region: 0,
        }];
        select_segment(&optimizer, &candidates.view(), &segment, &mut rng).unwrap();
        assert_eq!(*global_calls.lock().unwrap(), [[1, 0], [0, 0]]);
        assert_eq!(*local_calls.lock().unwrap(), [[0, 0], [0, 0]]);
    }
}
