//! Validated nested-search configuration and architecture encoding.
use super::flow;
use super::matvec::{self, unique};
use baux::config::OptimizerSpec;
use deser::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub(super) struct Config {
    pub(super) version: u32,
    pub(super) experiment: String,
    #[deser(default)]
    pub(super) dimensions: Vec<usize>,
    #[deser(default)]
    pub(super) matrix: Option<matvec::Config>,
    #[deser(default)]
    pub(super) flow: Option<flow::Config>,
    pub(super) seeds: Vec<u64>,
    pub(super) budgets: Vec<usize>,
    pub(super) methods: Vec<String>,
    pub(super) bits: Vec<usize>,
    pub(super) lanes: Vec<usize>,
    pub(super) pipelines: Vec<bool>,
    pub(super) target_seeds: Vec<u32>,
    pub(super) inner_evaluations: usize,
    pub(super) cycle_budget: usize,
    pub(super) cache_capacity: usize,
    pub(super) workers: usize,
    pub(super) inner: OptimizerSpec,
    pub(super) outer: OptimizerSpec,
}

impl Config {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.version != 1
            || ![
                "hardcaml-nested-scaling",
                "hardcaml-matvec",
                "hardcaml-flowlob",
            ]
            .contains(&self.experiment.as_str())
            || (self.experiment == "hardcaml-matvec") != self.matrix.is_some()
            || (self.experiment == "hardcaml-flowlob") != self.flow.is_some()
        {
            return Err("expected version 1 hardcaml-nested-scaling".into());
        }
        if (!self.has_validation()
            && (!unique(&self.dimensions) || self.dimensions.iter().any(|&n| n == 0 || n > 1024)))
            || !unique(&self.bits)
            || self
                .bits
                .iter()
                .any(|&n| !(if self.flow.is_some() { 20..=28 } else { 4..=20 }).contains(&n))
            || !unique(&self.lanes)
            || self.lanes.iter().any(|&n| {
                n == 0
                    || n > 32
                    || (!self.has_validation() && self.dimensions.iter().any(|&d| n > d))
            })
            || !unique(&self.pipelines)
            || !unique(&self.seeds)
            || !unique(&self.budgets)
            || self.budgets.iter().any(|&n| n == 0 || n > 100_000)
            || !unique(&self.methods)
            || self
                .methods
                .iter()
                .any(|s| !["fixed", "random", "baux"].contains(&s.as_str()))
        {
            return Err(
                "invalid or duplicate design grid, dimensions, seeds, budgets, or methods".into(),
            );
        }
        if self.target_seeds.is_empty()
            || self.target_seeds.len() > 10_000
            || !(2..=100_000).contains(&self.inner_evaluations)
            || !(1..=1024).contains(&self.cache_capacity)
            || !(1..=64).contains(&self.workers)
            || self.cycle_budget > 1_000_000_000_000
        {
            return Err("invalid workload, inner budget, cache capacity, or workers".into());
        }
        if let Some(matrix) = &self.matrix {
            matrix.validate(&self.lanes, self.target_seeds.len())?;
            if !self.dimensions.is_empty() {
                return Err("use matrix.shapes instead of dimensions".into());
            }
            self.validate_matrix(matrix)?;
        }
        if let Some(flow) = &self.flow {
            flow.validate()?;
            if !self.dimensions.is_empty() {
                return Err("use flow.models instead of dimensions".into());
            }
            for values in [&self.lanes, &flow.banks, &flow.queues] {
                if !unique(values) || values.iter().any(|&n| !n.is_power_of_two() || n > 32) {
                    return Err("invalid FlowLOB hardware grid".into());
                }
            }
        }
        for &dimension in &self.dimensions {
            for design in self.designs() {
                if self.cycle_budget / design.cycles(dimension) < 2 {
                    return Err(
                        "cycle-budget must permit at least two evaluations for every design".into(),
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_matrix(&self, matrix: &matvec::Config) -> Result<(), String> {
        for shape in &matrix.shapes {
            for design in self.designs() {
                for window in 0..self.target_seeds.len() {
                    let cycles = matrix.query_cycles(
                        *shape,
                        design.lanes,
                        design.banks,
                        design.pipeline,
                        design.queue,
                        window,
                    );
                    if self.cycle_budget / cycles < 3 {
                        return Err(format!(
                            "cycle-budget {} needs at least {} for {}x{} {:?} window {} (two training queries and validation)",
                            self.cycle_budget,
                            3 * cycles,
                            shape.rows,
                            shape.cols,
                            design,
                            window
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn designs(&self) -> Vec<Design> {
        let mut designs = Vec::new();
        for &bits in &self.bits {
            for &lanes in &self.lanes {
                for &pipeline in &self.pipelines {
                    designs.push(Design {
                        bits,
                        lanes,
                        pipeline,
                        banks: 0,
                        queue: 0,
                    });
                }
            }
        }
        if let Some((banks, queues)) = self.resources() {
            let mut expanded = Vec::new();
            for design in designs {
                for &banks in banks {
                    for &queue in queues {
                        expanded.push(Design {
                            banks,
                            queue,
                            ..design
                        });
                    }
                }
            }
            return expanded;
        }
        designs
    }

    pub(super) fn has_validation(&self) -> bool {
        self.matrix.is_some() || self.flow.is_some()
    }
    fn resources(&self) -> Option<(&Vec<usize>, &Vec<usize>)> {
        if let Some(flow) = &self.flow {
            return Some((&flow.banks, &flow.queues));
        }
        self.matrix.as_ref().map(|m| (&m.banks, &m.queues))
    }
    pub(super) fn outer_dimension(&self) -> usize {
        if self.has_validation() { 5 } else { 3 }
    }

    pub(super) fn decode(&self, point: &[f64]) -> Design {
        let pick = |len: usize, x: f64| ((x * len as f64) as usize).min(len - 1);
        Design {
            bits: self.bits[pick(self.bits.len(), point[0])],
            lanes: self.lanes[pick(self.lanes.len(), point[1])],
            pipeline: self.pipelines[pick(self.pipelines.len(), point[2])],
            banks: self
                .resources()
                .map_or(0, |(banks, _)| banks[pick(banks.len(), point[3])]),
            queue: self
                .resources()
                .map_or(0, |(_, queues)| queues[pick(queues.len(), point[4])]),
        }
    }

    pub(super) fn encode(&self, d: Design) -> Vec<f64> {
        let center =
            |index: Option<usize>, len: usize| (index.unwrap_or(0) as f64 + 0.5) / len as f64;
        let mut point = vec![
            center(self.bits.iter().position(|&x| x == d.bits), self.bits.len()),
            center(
                self.lanes.iter().position(|&x| x == d.lanes),
                self.lanes.len(),
            ),
            center(
                self.pipelines.iter().position(|&x| x == d.pipeline),
                self.pipelines.len(),
            ),
        ];
        if let Some((banks, queues)) = self.resources() {
            point.push(center(
                banks.iter().position(|&x| x == d.banks),
                banks.len(),
            ));
            point.push(center(
                queues.iter().position(|&x| x == d.queue),
                queues.len(),
            ));
        }
        point
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub(super) struct Design {
    pub(super) banks: usize,
    pub(super) queue: usize,
    pub(super) bits: usize,
    pub(super) lanes: usize,
    pub(super) pipeline: bool,
}

impl Design {
    pub(super) fn cycles(self, dimension: usize) -> usize {
        dimension.div_ceil(self.lanes) + usize::from(self.pipeline) + 3
    }
}
