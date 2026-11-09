//! Matrix workload and the runtime programming/consumer schedule.
use baux::artifact::json::{Value, json};
use deser::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[deser(deny_unknown_fields)]
pub(super) struct Shape {
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub(super) struct Config {
    pub shapes: Vec<Shape>,
    pub banks: Vec<usize>,
    pub queues: Vec<usize>,
    pub samples: usize,
    pub ready_periods: Vec<usize>,
    pub ready_cycles: Vec<usize>,
    pub initial_stalls: Vec<usize>,
    pub load_gaps: Vec<usize>,
}

pub(super) fn unique<T: PartialEq>(values: &[T]) -> bool {
    !values.is_empty()
        && values
            .iter()
            .enumerate()
            .all(|(i, x)| !values[..i].contains(x))
}

fn power_grid(values: &[usize]) -> bool {
    unique(values) && values.iter().all(|&n| n.is_power_of_two() && n <= 32)
}

fn at(values: &[usize], window: usize) -> usize {
    values[window % values.len()]
}

impl Config {
    pub fn validate(&self, lanes: &[usize], windows: usize) -> Result<(), String> {
        if !unique(&self.shapes)
            || self
                .shapes
                .iter()
                .any(|s| !(1..=256).contains(&s.rows) || !(1..=4096).contains(&s.cols))
            || !power_grid(&self.banks)
            || !power_grid(&self.queues)
            || !power_grid(lanes)
            || !(1..=4096).contains(&self.samples)
        {
            return Err("invalid matrix shape, power-of-two hardware grid, or sample count".into());
        }
        for values in [
            &self.ready_periods,
            &self.ready_cycles,
            &self.initial_stalls,
            &self.load_gaps,
        ] {
            if values.len() != 1 && values.len() != windows {
                return Err("matrix schedules need one entry or one entry per window".into());
            }
        }
        for w in 0..windows {
            let period = at(&self.ready_periods, w);
            if !(1..=10000).contains(&period)
                || !(1..=period).contains(&at(&self.ready_cycles, w))
                || at(&self.initial_stalls, w) > 1_000_000
                || at(&self.load_gaps, w) > 10000
            {
                return Err("invalid matrix programming or consumer schedule".into());
            }
        }
        Ok(())
    }

    pub fn query_cycles(
        &self,
        shape: Shape,
        lanes: usize,
        banks: usize,
        pipeline: bool,
        queue: usize,
        window: usize,
    ) -> usize {
        let arithmetic =
            shape.cols.div_ceil(lanes) * (lanes.div_ceil(banks) + 1 + usize::from(pipeline));
        let period = at(&self.ready_periods, window);
        let ready = at(&self.ready_cycles, window);
        let initial = at(&self.initial_stalls, window);
        let mut departures = Vec::with_capacity(shape.rows);
        let mut next_row = 0;
        for row in 0..shape.rows {
            let mut push = next_row + arithmetic;
            if row >= queue {
                push = push.max(departures[row - queue]);
            }
            let previous = departures.last().map_or(0, |time| time + 1);
            let available = (push + 1).max(previous).max(initial);
            let phase = (available - initial) % period;
            let departure = available + if phase < ready { 0 } else { period - phase };
            departures.push(departure);
            next_row = push + 1;
        }
        let compute = departures.last().copied().unwrap_or(0) + 2;
        let upload = shape.cols.div_ceil(banks) * (1 + at(&self.load_gaps, window));
        1 + shape.rows * upload + self.samples * (upload + compute)
    }

    pub fn request(
        &self,
        shape: Shape,
        design: &Value,
        target: u32,
        weights: &[f64],
        context: &Value,
        seed: u64,
    ) -> Value {
        let window = context["window"].as_u64().unwrap_or(0) as usize;
        let split = match context["phase"].as_str() {
            Some("prequential") => "test",
            Some("validation") => "validation",
            _ => "train",
        };
        json!({"version": 1, "rows": shape.rows, "cols": shape.cols,
            "bits": design["bits"], "lanes": design["lanes"], "banks": design["banks"],
            "pipeline": design["pipeline"], "queue": design["queue"],
            "target_seed": target, "weights": weights, "samples": self.samples,
            "split": split, "data_seed": seed.wrapping_add(window as u64 * 1_000_003) % 2_147_483_647,
            "ready_period": at(&self.ready_periods,window), "ready_cycles": at(&self.ready_cycles,window),
            "initial_stall": at(&self.initial_stalls,window), "load_gap": at(&self.load_gaps,window)})
    }
}

pub(super) fn check_cycles(observation: &Value, cycles: usize) -> Result<(), String> {
    let parts = &observation["cycle_breakdown"];
    let mut total = 0;
    for key in ["reset", "weight_load", "input_load", "compute"] {
        total += parts[key]
            .as_u64()
            .ok_or_else(|| format!("missing matrix cycle phase {key}"))? as usize;
    }
    if total != cycles || total == 0 {
        return Err("matrix cycle accounting mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_validation() {
        let mut config = Config {
            shapes: vec![Shape { rows: 3, cols: 7 }],
            banks: vec![1, 4],
            queues: vec![1, 8],
            samples: 8,
            ready_periods: vec![1, 7],
            ready_cycles: vec![1],
            initial_stalls: vec![0],
            load_gaps: vec![0],
        };
        assert!(config.validate(&[1, 8], 2).is_ok());
        assert!(config.validate(&[3], 2).is_err());
        assert!(config.validate(&[1], 3).is_err());
        config.ready_cycles = vec![2];
        assert!(config.validate(&[1], 2).is_err());
    }
}
