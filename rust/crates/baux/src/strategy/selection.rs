//! Unified segment-based acquisition selection over candidate pools.

use ndarray::s;
use rand::seq::SliceRandom;

use super::*;

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
    let config = optimizer.config().acquisition;
    let Some(surrogate) = optimizer.surrogate() else {
        return random_segments(segments, rng);
    };
    match config {
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
                for &prev in &local {
                    values[prev] = f64::NEG_INFINITY;
                }
                local.push(argmax_tie(&values, rng));
            }
            selected.extend(local.into_iter().map(|index| segment.start + index));
        }
        return Ok(selected);
    }
    let mask_dups = segments.len() > 1;
    let mut selected = Vec::new();
    for segment in segments {
        let mut local = Vec::with_capacity(segment.arms);
        for arm in 0..segment.arms {
            let mut values: Vec<f64> = (segment.start..segment.end)
                .map(|cand| samples[[arm, cand, 0]])
                .collect();
            if mask_dups {
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
