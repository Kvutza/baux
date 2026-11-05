use crate::objective_observation::{ObjectiveObservation, ObjectiveRecord};
use ndarray::{Array1, Array2};

fn normalized(observation: ObjectiveObservation, reference: &[f64], scales: &[f32]) -> Vec<f64> {
    observation
        .estimates()
        .iter()
        .zip(reference)
        .zip(scales)
        .map(|((estimate, reference), scale)| {
            let conservative = f64::from(estimate.mean) - 2.0 * f64::from(estimate.variance).sqrt();
            (conservative - reference) / f64::from(*scale)
        })
        .collect()
}

pub fn measured<'a>(
    rows: impl Iterator<Item = &'a ObjectiveRecord>,
    candidate: Option<ObjectiveObservation>,
    reference: &[f64],
    scales: &[f32],
) -> Result<f64, String> {
    let width = scales.len();
    let mut values = rows
        .flat_map(|record| normalized(record.observation, reference, scales))
        .collect::<Vec<_>>();
    let mut count = values.len() / width;
    if let Some(candidate) = candidate {
        values.extend(normalized(candidate, reference, scales));
        count += 1;
    }
    let points =
        Array2::from_shape_vec((count, width), values).map_err(|error| error.to_string())?;
    crate::hypervolume::hypervolume_max(&points.view(), &Array1::zeros(width).view())
        .map_err(|error| error.to_string())
}

pub fn improvement(before: f64, after: f64) -> (bool, f64) {
    let improvement = after - before;
    (
        improvement > f64::EPSILON * after.abs().max(1.0),
        improvement,
    )
}

pub fn values(observation: ObjectiveObservation) -> Array1<f64> {
    Array1::from_iter(
        observation
            .estimates()
            .iter()
            .map(|estimate| f64::from(estimate.mean)),
    )
}

pub fn exclusive_volume(
    records: &[ObjectiveRecord],
    excluded: usize,
    total: f64,
    reference: &[f64],
    scales: &[f32],
) -> f64 {
    let volume = measured(
        records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| (index != excluded).then_some(record)),
        None,
        reference,
        scales,
    )
    .unwrap_or(0.0);
    total - volume
}

pub fn max_exclusive(
    records: &[ObjectiveRecord],
    excluded: Option<usize>,
    reference: &[f64],
    scales: &[f32],
) -> Result<usize, String> {
    let total = measured(records.iter(), None, reference, scales)?;
    Ok((0..records.len())
        .filter(|&region| excluded != Some(region))
        .max_by(|&left, &right| {
            exclusive_volume(records, left, total, reference, scales)
                .total_cmp(&exclusive_volume(records, right, total, reference, scales))
                .then_with(|| right.cmp(&left))
        })
        .unwrap_or_else(|| excluded.unwrap_or(0)))
}
