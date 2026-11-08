//! Fixed-budget host comparisons of memory, acquisition roles, and starts.

use baux::artifact::{
    json::{self, Value, json},
    runs,
};
use deser::Deserialize;
use std::{fs, path::Path};

mod automatic;

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Config {
    version: u32,
    experiment: String,
    dimensions: usize,
    evaluations: usize,
    regions: usize,
    #[deser(default = default_reps())]
    repetitions: u32,
    seeds: Option<Vec<u64>>,
    sharing: Vec<String>,
    starts: Vec<String>,
    restart_after_evaluations: usize,
    allocation: String,
    controller: String,
    restart_memory: Vec<String>,
    roles: Option<Vec<String>>,
    restart_centers: Option<Vec<String>>,
    restart_candidates: Option<usize>,
    objectives: Option<Vec<String>>,
    length_init: Option<f64>,
    length_min: Option<f64>,
    length_max: Option<f64>,
    failure_tolerance: Option<i32>,
    success_tolerance: Option<i32>,
}

const fn default_reps() -> u32 {
    3
}

pub(super) fn execute(root: &Path, rounds: Option<u32>, text: &str) -> Result<bool, String> {
    let mut config: Config = baux::artifact::toml::from_str(text).map_err(|e| e.to_string())?;
    if config.version != 1 || config.experiment != "host-regions" {
        return Err("host regional experiment requires version 1".into());
    }
    if let Some(rounds) = rounds {
        config.evaluations = rounds as usize;
        config.restart_after_evaluations = if config.controller == "turbo" {
            0
        } else {
            config.evaluations / 2
        };
    }
    validate(&config)?;
    let resolved = format!(
        "version = 1\nexperiment = \"host-regions\"\ndimensions = {}\nevaluations = {}\nregions = {}\nrepetitions = {}\nsharing = {:?}\nstarts = {:?}\nrestart_after_evaluations = {}\nallocation = {:?}\ncontroller = {:?}\nrestart_memory = {:?}\nroles = {:?}\nrestart_centers = {:?}\nrestart_candidates = {}\nobjectives = {:?}\nlength_init = {}\nlength_min = {}\nlength_max = {}\nfailure_tolerance = {}\nsuccess_tolerance = {}\n",
        config.dimensions,
        config.evaluations,
        config.regions,
        config.resolved_seeds().len(),
        config.sharing,
        config.starts,
        config.restart_after_evaluations,
        config.allocation,
        config.controller,
        config.restart_memory,
        config.roles(),
        config.centers(),
        config.candidates(),
        config.objectives(),
        config.length().length_init,
        config.length().length_min,
        config.length().length_max,
        config.fail(),
        config.succeed()
    );
    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let output = root
        .join(".cache/baux/runs/regional")
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;
    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &executable)?;
    record.model = if config.objectives() == ["double_well"] {
        "host-double-well".into()
    } else {
        "host-synthetic-suite".into()
    };
    record.optimizer = config.experiment.clone();
    record.protocol = runs::digest(resolved.as_bytes());
    let result = measure(&config).and_then(|report| {
        let bytes = baux::artifact::cbor::to_vec(&report).map_err(|e| e.to_string())?;
        fs::write(output.join("regional.cbor"), &bytes).map_err(|e| e.to_string())?;
        println!(
            "Host regional experiment completed. Evidence: {}",
            output.display()
        );
        Ok(())
    });
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}

fn validate(config: &Config) -> Result<(), String> {
    if config.dimensions == 0
        || config.evaluations < 8
        || config.evaluations % 8 != 0
        || config.regions != 4
        || config.allocation != "least_trials"
        || config.restart_memory != ["fresh", "retain_relevant"]
        || config.resolved_seeds().is_empty()
        || config.sharing.is_empty()
        || config.starts.is_empty()
    {
        return Err("host-regions requires positive dimensions, evaluations divisible by 8, repetitions/seeds, sharing policies, and start layouts".into());
    }
    if !matches!(
        (config.controller.as_str(), config.restart_after_evaluations),
        ("fixed_radius", n) if n == config.evaluations / 2
    ) && !matches!(
        (config.controller.as_str(), config.restart_after_evaluations),
        ("turbo", 0)
    ) {
        return Err(
            "host-regions requires a midpoint fixed-length restart or automatic TuRBO retirement"
                .into(),
        );
    }
    let length = config.length();
    if !length.length_init.is_finite()
        || !length.length_min.is_finite()
        || !length.length_max.is_finite()
        || length.length_min <= 0.0
        || length.length_min > length.length_init
        || length.length_init > length.length_max
        || length.length_max > 2.0
        || config.fail() <= 0
        || config.succeed() <= 0
    {
        return Err("host-regions has invalid TuRBO controller parameters".into());
    }
    if config.sharing.iter().any(|name| {
        !matches!(
            name.as_str(),
            "origin_only" | "box_contain" | "ellipsoid_ball" | "nearest_center"
        )
    }) {
        return Err("host-regions has an unknown sharing policy".into());
    }
    if config
        .starts
        .iter()
        .any(|name| !matches!(name.as_str(), "separated" | "clustered"))
    {
        return Err("host-regions has an unknown start layout".into());
    }
    let roles = config.roles();
    if roles.is_empty()
        || roles.len() > 2
        || (roles.len() == 2 && roles[0] == roles[1])
        || roles
            .iter()
            .any(|name| !matches!(name.as_str(), "uniform" | "split_beta"))
    {
        return Err("host-regions has an unknown acquisition-role diagnostic".into());
    }
    let centers = config.centers();
    if centers.is_empty()
        || centers.len() > 3
        || centers
            .iter()
            .any(|name| !matches!(name.as_str(), "midpoint" | "uniform" | "maximin"))
        || centers
            .iter()
            .enumerate()
            .any(|(index, name)| centers[..index].contains(name))
        || (config.controller == "turbo" && centers.contains(&"midpoint".into()))
        || config.candidates() == 0
    {
        return Err("host-regions has an invalid restart-center diagnostic".into());
    }
    let objectives = config.objectives();
    if objectives.is_empty()
        || objectives.len() > 3
        || objectives
            .iter()
            .any(|name| !matches!(name.as_str(), "double_well" | "rastrigin" | "ackley"))
        || objectives
            .iter()
            .enumerate()
            .any(|(index, name)| objectives[..index].contains(name))
    {
        return Err("host-regions has an invalid objective diagnostic".into());
    }
    Ok(())
}

impl Config {
    fn resolved_seeds(&self) -> Vec<u64> {
        super::resolved_seeds(&self.experiment, self.repetitions, &self.seeds)
    }

    fn roles(&self) -> Vec<String> {
        self.roles.clone().unwrap_or_else(|| vec!["uniform".into()])
    }

    fn centers(&self) -> Vec<String> {
        self.restart_centers
            .clone()
            .unwrap_or_else(|| vec!["midpoint".into()])
    }

    fn candidates(&self) -> usize {
        self.restart_candidates.unwrap_or(64)
    }

    fn objectives(&self) -> Vec<String> {
        self.objectives
            .clone()
            .unwrap_or_else(|| vec!["double_well".into()])
    }

    fn length(&self) -> baux::trust_region::TRLengthConfig {
        let default = baux::trust_region::TRLengthConfig::default();
        baux::trust_region::TRLengthConfig::new(
            self.length_init.unwrap_or(default.length_init),
            self.length_min.unwrap_or(default.length_min),
            self.length_max.unwrap_or(default.length_max),
        )
    }

    fn fail(&self) -> i32 {
        self.failure_tolerance.unwrap_or(5)
    }

    fn succeed(&self) -> i32 {
        self.success_tolerance.unwrap_or(3)
    }
}

struct ArmResult {
    report: Value,
    prefix_values: Vec<f64>,
    prefix_counts: Vec<usize>,
}

fn measure(config: &Config) -> Result<Value, String> {
    let mut pairs = Vec::new();
    for objective in config.objectives() {
        pairs.extend(measure_objective(config, &objective)?);
    }
    Ok(json!({
        "experiment":"host-regions", "dimensions":config.dimensions,
        "objectives":config.objectives(),
        "objective_domain":"x=3*u-1.5 for unit-coordinate u",
        "interpretation":if config.controller == "turbo" { "matched fixed-budget host comparison with actual automatic TuRBO retirement and configured candidate-ranked replacement; diagnostic failure tolerance and length bounds make retirement observable; outcomes do not establish policy superiority" } else { "matched fixed-budget host comparison with a forced restart after half the budget; split-beta and maximin are BAUX diagnostics, not MTRBO or APOSMM; outcomes are policy-associated, with no general superiority claim" },
        "results":pairs
    }))
}

fn measure_objective(config: &Config, objective: &str) -> Result<Vec<Value>, String> {
    use ndarray::Array2;
    use rand::Rng;

    let mut pairs = Vec::new();
    for seed in config.resolved_seeds() {
        for layout in &config.starts {
            let mut start_rng = baux::prng(seed);
            let centers =
                Array2::from_shape_fn((config.regions, config.dimensions), |(slot, _)| {
                    let base = start_center(layout, slot);
                    (base + start_rng.gen_range(-0.01..0.01)).clamp(0.0, 1.0)
                });
            for name in &config.sharing {
                for role in config.roles() {
                    for center_policy in config.centers() {
                        pairs.push(pair_report(
                            config,
                            seed,
                            &centers,
                            layout,
                            name,
                            &role,
                            &center_policy,
                            objective,
                        )?);
                    }
                }
            }
        }
    }
    Ok(pairs)
}

fn pair_report(
    config: &Config,
    seed: u64,
    centers: &ndarray::Array2<f64>,
    layout: &str,
    name: &str,
    role: &str,
    center_policy: &str,
    objective: &str,
) -> Result<Value, String> {
    let sharing = sharing_policy(name);
    let fresh = run_arm(
        config,
        seed,
        centers,
        sharing,
        "fresh",
        role,
        center_policy,
        objective,
    )?;
    let retained = run_arm(
        config,
        seed,
        centers,
        sharing,
        "retain_relevant",
        role,
        center_policy,
        objective,
    )?;
    if fresh.prefix_values != retained.prefix_values
        || fresh.prefix_counts != retained.prefix_counts
    {
        return Err("matched arms diverged before restart".into());
    }
    if config.controller == "turbo" {
        let first = fresh.report["automatic_restarts"]
            .as_seq()
            .and_then(|events| events.first());
        let second = retained.report["automatic_restarts"]
            .as_seq()
            .and_then(|events| events.first());
        let same = match (first, second) {
            (None, None) => true,
            (Some(a), Some(b)) => [
                "evaluation_count",
                "slot",
                "generation_after",
                "center_after",
            ]
            .iter()
            .all(|field| a[*field] == b[*field]),
            _ => false,
        };
        if !same {
            return Err("matched arms diverged at first automatic retirement".into());
        }
    }
    Ok(json!({
        "seed":seed, "sharing":name, "starts":layout, "roles":role,
        "objective":objective,
        "restart_centers":center_policy,
        "regions":config.regions, "pre_restart_matched":true,
        "pairing_prefix_semantics":if config.controller == "turbo" { "observations through first retirement tell and local-model counts before that tell" } else { "observations and local-model counts immediately before forced restart" },
        "pre_restart_observation_values":fresh.prefix_values,
        "pre_restart_model_counts":fresh.prefix_counts,
        "fresh":fresh.report, "retain_relevant":retained.report
    }))
}

fn start_center(layout: &str, slot: usize) -> f64 {
    match layout {
        "separated" => [0.08, 0.36, 0.64, 0.92][slot],
        "clustered" => 0.16 + slot as f64 * 0.015,
        _ => unreachable!("validated layout"),
    }
}

fn sharing_policy(name: &str) -> baux::optimizer::SharingPolicy {
    use baux::optimizer::SharingPolicy;
    match name {
        "origin_only" => SharingPolicy::OriginOnly,
        "box_contain" => SharingPolicy::BoxContain,
        "ellipsoid_ball" => SharingPolicy::EllipsoidBall,
        "nearest_center" => SharingPolicy::NearestCenter,
        _ => unreachable!("validated sharing"),
    }
}

fn run_arm(
    config: &Config,
    seed: u64,
    centers: &ndarray::Array2<f64>,
    sharing: baux::optimizer::SharingPolicy,
    memory: &str,
    roles: &str,
    center_policy: &str,
    objective: &str,
) -> Result<ArmResult, String> {
    if config.controller == "turbo" {
        return automatic::run_arm(
            config,
            seed,
            centers,
            sharing,
            memory,
            roles,
            center_policy,
            objective,
        );
    }
    let mut rng = baux::prng(seed);
    let mut optimizer = make_optimizer(
        config,
        centers,
        sharing,
        memory,
        roles,
        center_policy,
        &mut rng,
    )?;
    let mut values = Vec::with_capacity(config.evaluations);
    let mut region_counts = vec![0u64; config.regions];
    let mut pre_values = None;
    let mut pre_counts = None;
    let mut before_restart = None;
    let mut after_restart = None;
    let mut generation = None;
    let mut restart_center = None;
    let mut separation = None;

    for start in (0..config.evaluations).step_by(config.regions) {
        if start == config.restart_after_evaluations {
            pre_values = Some(values.clone());
            pre_counts = Some(optimizer.model_counts().ok_or("missing local models")?);
            let center = choose_center(&optimizer, center_policy, seed, config)?;
            let baux::optimizer::TrustRegionState::Multi { state, .. } = optimizer.trust_region()
            else {
                return Err("restart distance needs multi-region state".into());
            };
            separation = Some(restart_distance(&center, state));
            let (before, after, next_generation) = do_restart(&mut optimizer, &center, &mut rng)?;
            restart_center = Some(center);
            before_restart = Some(before);
            after_restart = Some(after);
            generation = Some(next_generation);
        }
        let count = config.regions.min(config.evaluations - start);
        let points = optimizer.ask(count, &mut rng).map_err(|e| e.to_string())?;
        let previous = region_trials(&optimizer)?;
        let rewards = score(&points, config.dimensions, objective);
        values.extend(rewards.column(0).iter().copied());
        optimizer
            .tell_variance(&points.view(), &rewards.view(), None, &mut rng)
            .map_err(|e| e.to_string())?;
        add_trials(&mut region_counts, &previous, &region_trials(&optimizer)?)?;
    }

    let before = before_restart.ok_or("restart checkpoint missing")?;
    let after = after_restart.ok_or("restart checkpoint missing")?;
    let report = arm_report(
        config,
        memory,
        roles,
        center_policy,
        values,
        region_counts,
        restart_center.ok_or("restart center missing")?,
        separation.ok_or("restart distance missing")?,
        generation,
        before,
        after,
        optimizer.model_counts(),
    );
    Ok(ArmResult {
        report,
        prefix_values: pre_values.ok_or("restart checkpoint missing")?,
        prefix_counts: pre_counts.ok_or("restart checkpoint missing")?,
    })
}

fn make_optimizer(
    config: &Config,
    centers: &ndarray::Array2<f64>,
    sharing: baux::optimizer::SharingPolicy,
    memory: &str,
    roles: &str,
    center_policy: &str,
    rng: &mut baux::Prng,
) -> Result<baux::BayesOptimizer, String> {
    use baux::optimizer::{
        AllocationPolicy, InitStrategyState, MultiTrustRegionConfig, MultiTrustRegionState,
        RegionController, RestartCenterPolicy, RestartCenterSelection, RestartMemoryPolicy,
        Strategy, StrategyState,
    };
    use ndarray::Array2;

    let mut config_tr = MultiTrustRegionConfig::new(config.regions, config.length());
    config_tr.fail_tolerance = config.fail();
    config_tr.succ_tolerance = config.succeed();
    config_tr.sharing_policy = sharing;
    config_tr.allocation_policy = AllocationPolicy::LeastTrials;
    config_tr.controller = if config.controller == "turbo" {
        RegionController::Turbo
    } else {
        RegionController::FixedRadius
    };
    config_tr.restart_memory_policy = match memory {
        "fresh" => RestartMemoryPolicy::Fresh,
        _ => RestartMemoryPolicy::RetainRelevant,
    };
    config_tr.restart_center_policy = RestartCenterPolicy {
        candidates: config.candidates(),
        selection: if center_policy == "maximin" {
            RestartCenterSelection::Maximin
        } else {
            RestartCenterSelection::First
        },
    };
    if roles == "split_beta" {
        // BAUX diagnostic motivated by separate exploitation and uncertainty
        // regions in Das et al. (2026), https://arxiv.org/abs/2605.06618.
        // This beta assignment is not the paper's MTRBO algorithm.
        config_tr.region_acquisitions[0] = Some(baux::config::AcquisitionConfig::UCB { beta: 0.0 });
        config_tr.region_acquisitions[1] = Some(baux::config::AcquisitionConfig::UCB { beta: 4.0 });
    }
    let state =
        MultiTrustRegionState::new(config.dimensions, config_tr, Some(&centers.view()), rng)
            .map_err(|e| e.to_string())?;
    let strategy = Strategy::Experimental(StrategyState {
        init: InitStrategyState::new(baux::InitStrategy::LHD, 0),
        multi_tr: state,
        in_init: false,
        pending_proposal: None,
    });
    let bounds = Array2::from_shape_fn((config.dimensions, 2), |(_, side)| side as f64);
    baux::BayesOptimizer::new_strategy(bounds, baux::turbo_enn(), strategy, rng)
        .map_err(|e| e.to_string())
}

fn do_restart(
    optimizer: &mut baux::BayesOptimizer,
    center: &[f64],
    rng: &mut baux::Prng,
) -> Result<(Vec<usize>, Vec<usize>, u64), String> {
    use baux::optimizer::TrustRegionState;

    let before = optimizer.model_counts().ok_or("missing local models")?;
    let generation = match optimizer.trust_region() {
        TrustRegionState::Multi { state, .. } => state.generation(0).ok_or("missing slot")?,
        _ => return Err("not a host multi-region optimizer".into()),
    };
    optimizer
        .restart_region(0, &ndarray::Array1::from_vec(center.to_vec()).view(), rng)
        .map_err(|e| e.to_string())?;
    let after = optimizer.model_counts().ok_or("missing local models")?;
    Ok((before, after, generation + 1))
}

fn choose_center(
    optimizer: &baux::BayesOptimizer,
    policy: &str,
    seed: u64,
    config: &Config,
) -> Result<Vec<f64>, String> {
    use baux::optimizer::TrustRegionState;

    if policy == "midpoint" {
        return Ok(vec![0.5; config.dimensions]);
    }
    let mut rng = baux::prng(seed ^ 0x6c8e_9cf5_d347_a1b2);
    let TrustRegionState::Multi { state, .. } = optimizer.trust_region() else {
        return Err("restart center needs multi-region state".into());
    };
    state
        .choose_center(0, &mut rng)
        .map(|center| center.to_vec())
        .map_err(|error| error.to_string())
}

fn restart_distance(point: &[f64], state: &baux::optimizer::MultiTrustRegionState) -> f64 {
    let mut nearest = f64::INFINITY;
    for slot in 1..state.num_regions() {
        let mut distance = 0.0;
        for (&coordinate, &center) in point.iter().zip(state.centers.row(slot)) {
            distance += (coordinate - center).powi(2);
        }
        nearest = nearest.min(distance);
    }
    nearest
}

fn region_trials(optimizer: &baux::BayesOptimizer) -> Result<Vec<u64>, String> {
    match optimizer.trust_region() {
        baux::optimizer::TrustRegionState::Multi { state, .. } => Ok(state.trial_counts()),
        _ => Err("not a host multi-region optimizer".into()),
    }
}

fn add_trials(total: &mut [u64], before: &[u64], after: &[u64]) -> Result<(), String> {
    for (slot, ((sum, before), after)) in total.iter_mut().zip(before).zip(after).enumerate() {
        *sum += after
            .checked_sub(*before)
            .ok_or_else(|| format!("trial counter reset during tell in slot {slot}"))?;
    }
    Ok(())
}

fn score(
    points: &ndarray::Array2<f64>,
    dimensions: usize,
    objective: &str,
) -> ndarray::Array2<f64> {
    ndarray::Array2::from_shape_fn((points.nrows(), 1), |(row, _)| {
        let values = points.row(row).mapv(|unit| 3.0 * unit - 1.5);
        -objective_loss(objective, &values, dimensions)
    })
}

fn objective_loss(objective: &str, values: &ndarray::Array1<f64>, dimensions: usize) -> f64 {
    let mean = |term: &dyn Fn(f64) -> f64| {
        values.iter().map(|&value| term(value)).sum::<f64>() / dimensions as f64
    };
    match objective {
        "double_well" => mean(&|value| {
            let residual = value * value - 1.0;
            residual * residual + 0.05 * value
        }),
        "rastrigin" => {
            mean(&|value| value * value - 10.0 * (std::f64::consts::TAU * value).cos() + 10.0)
        }
        "ackley" => {
            let radius = mean(&|value| value * value).sqrt();
            let periodic = mean(&|value| (std::f64::consts::TAU * value).cos());
            -20.0 * (-0.2 * radius).exp() - periodic.exp() + 20.0 + std::f64::consts::E
        }
        _ => unreachable!("validated objective"),
    }
}

fn arm_report(
    config: &Config,
    memory: &str,
    roles: &str,
    center_policy: &str,
    values: Vec<f64>,
    region_counts: Vec<u64>,
    restart_center: Vec<f64>,
    separation: f64,
    generation: Option<u64>,
    before: Vec<usize>,
    after: Vec<usize>,
    final_counts: Option<Vec<usize>>,
) -> Value {
    let delta = after
        .iter()
        .zip(&before)
        .map(|(new, old)| *new as isize - *old as isize)
        .collect::<Vec<_>>();
    json!({
        "restart_memory":memory, "allocation":config.allocation,
        "roles":roles,
        "restart_center_policy":center_policy,
        "restart_candidates":config.candidates(),
        "region_acquisition_beta":if roles == "split_beta" { vec![0.0,4.0,2.0,2.0] } else { vec![2.0;4] },
        "controller":config.controller,
        "acquisition":"turbo_enn (RAASP, ENN, per-region UCB beta as reported)",
        "evaluation_count_including_startup":config.evaluations,
        "initial_region_proposal_budget_indices":[1, config.regions],
        "post_startup_budget_indices":[config.regions + 1, config.evaluations],
        "observation_value_budget_indices":[1, config.evaluations],
        "observation_values":values,
        "forced_restart_after_evaluations":config.restart_after_evaluations,
        "restarted_slot":0, "restart_generation_after":generation,
        "restart_center_unit":restart_center,
        "restart_nearest_active_distance_squared":separation,
        "local_model_counts_before_restart":before,
        "local_model_counts_after_restart":after,
        "local_model_count_delta_at_restart":delta,
        "local_model_counts_final":final_counts,
        "region_evaluations":region_counts
    })
}

#[cfg(test)]
mod tests {
    use super::{Config, measure, validate};

    #[test]
    fn host_config() {
        let config: Config = baux::artifact::toml::from_str(
            "version=1\nexperiment='host-regions'\ndimensions=8\nevaluations=64\nregions=4\nseeds=[17]\nsharing=['box_contain']\nstarts=['separated']\nrestart_after_evaluations=32\nallocation='least_trials'\ncontroller='fixed_radius'\nrestart_memory=['fresh','retain_relevant']\n",
        ).unwrap();
        assert!(validate(&config).is_ok());
        assert_eq!(config.roles(), ["uniform"]);
        assert_eq!(config.objectives(), ["double_well"]);
        let mut invalid = config;
        invalid.evaluations = 63;
        assert!(validate(&invalid).is_err());
        invalid.evaluations = 64;
        invalid.roles = Some(vec!["unknown".into()]);
        assert!(validate(&invalid).is_err());
        invalid.roles = Some(vec!["uniform".into(), "uniform".into()]);
        assert!(validate(&invalid).is_err());
        invalid.roles = None;
        invalid.restart_centers = Some(vec!["uniform".into(), "maximin".into()]);
        assert!(validate(&invalid).is_ok());
        invalid.restart_candidates = Some(0);
        assert!(validate(&invalid).is_err());
        invalid.restart_candidates = Some(64);
        invalid.objectives = Some(vec!["ackley".into(), "rastrigin".into()]);
        assert!(validate(&invalid).is_ok());
    }

    #[test]
    fn automaticretirement() {
        let config: Config = baux::artifact::toml::from_str(
            "version=1\nexperiment='host-regions'\ndimensions=8\nevaluations=64\nregions=4\nseeds=[17]\nsharing=['origin_only']\nstarts=['separated']\nrestart_after_evaluations=0\nallocation='least_trials'\ncontroller='turbo'\nrestart_memory=['fresh','retain_relevant']\nroles=['uniform']\nrestart_centers=['uniform']\nrestart_candidates=1\nobjectives=['double_well']\nlength_init=0.8\nlength_min=0.2\nlength_max=1.6\nfailure_tolerance=2\nsuccess_tolerance=3\n",
        )
        .unwrap();
        validate(&config).unwrap();
        let report = measure(&config).unwrap();
        let pair = &report["results"][0];
        assert_eq!(pair["pre_restart_matched"], true);
        for memory in ["fresh", "retain_relevant"] {
            let arm = &pair[memory];
            assert_eq!(arm["observation_values"].as_seq().unwrap().len(), 64);
            assert!(!arm["automatic_restarts"].as_seq().unwrap().is_empty());
        }
    }
}
