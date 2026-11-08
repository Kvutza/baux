//! Recorded resident multi-start experiment, routed through `baux tune`.

mod host;
use baux::artifact::{
    json::{self, Value, json},
    runs,
};
use deser::Deserialize;
use std::{fs, path::Path};

#[derive(Deserialize)]
#[deser(deny_unknown_fields)]
struct Config {
    version: u32,
    experiment: String,
    dimensions: usize,
    evaluations: usize,
    regions: Vec<usize>,
    #[deser(default = default_reps())]
    repetitions: u32,
    seeds: Option<Vec<u64>>,
    sharing: Vec<String>,
    starts: Vec<String>,
}

const fn default_reps() -> u32 {
    3
}

pub(super) fn resolved_seeds(
    experiment: &str,
    repetitions: u32,
    seeds: &Option<Vec<u64>>,
) -> Vec<u64> {
    seeds.clone().unwrap_or_else(|| {
        (0..repetitions)
            .map(|rep| baux::tune_seed(experiment, "proposal", rep, 0))
            .collect()
    })
}

impl Config {
    fn resolved_seeds(&self) -> Vec<u64> {
        resolved_seeds(&self.experiment, self.repetitions, &self.seeds)
    }
}

pub(super) fn execute(root: &Path, path: &Path, rounds: Option<u32>) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    let experiment = value["experiment"].as_str();
    if experiment == Some("host-regions") {
        return host::execute(root, rounds, &text);
    }
    if experiment != Some("resident-regions") {
        return Ok(false);
    }
    let mut config: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if config.version != 1 {
        return Err("regional experiment requires version 1".into());
    }
    if let Some(rounds) = rounds {
        config.evaluations = rounds as usize;
    }
    let seeds = config.resolved_seeds();
    if config.regions.is_empty()
        || seeds.is_empty()
        || config.sharing.is_empty()
        || config.starts.is_empty()
    {
        return Err(
            "regional experiment requires regions, repetitions/seeds, sharing policies, and start layouts"
                .into(),
        );
    }
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
    let resolved = format!(
        "version = 1\nexperiment = \"resident-regions\"\ndimensions = {}\nevaluations = {}\nregions = {:?}\nrepetitions = {}\nsharing = {:?}\nstarts = {:?}\n",
        config.dimensions,
        config.evaluations,
        config.regions,
        seeds.len(),
        config.sharing,
        config.starts
    );
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;
    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &executable)?;
    record.model = "bf16-double-well".into();
    record.optimizer = config.experiment.clone();
    record.protocol = runs::digest(resolved.as_bytes());
    let result = measure(&config).and_then(|report| {
        let bytes = baux::artifact::cbor::to_vec(&report).map_err(|e| e.to_string())?;
        fs::write(output.join("regional.cbor"), &bytes).map_err(|e| e.to_string())?;
        println!(
            "Resident regional experiment completed. Evidence: {}",
            output.display()
        );
        Ok(())
    });
    runs::finish(root, &mut record, &result)?;
    result?;
    Ok(true)
}

#[cfg(target_os = "macos")]
fn measure(config: &Config) -> Result<Value, String> {
    use baux::{
        optimizer::multi_tr::SharingPolicy,
        resident_metal::{RegionalTrial, regional_trial},
        semantics::optimizer::portfolio::Allocation,
    };
    let mut results = Vec::new();
    for seed in config.resolved_seeds() {
        for &regions in &config.regions {
            for layout in &config.starts {
                let clustered = match layout.as_str() {
                    "separated" => false,
                    "clustered" => true,
                    _ => return Err("unknown regional start layout".into()),
                };
                for policy_name in &config.sharing {
                    let sharing = match policy_name.as_str() {
                        "origin_only" => SharingPolicy::OriginOnly,
                        "box_contain" => SharingPolicy::BoxContain,
                        "ellipsoid_ball" => SharingPolicy::EllipsoidBall,
                        "nearest_center" => SharingPolicy::NearestCenter,
                        _ => return Err("unknown regional sharing policy".into()),
                    };
                    for (allocation_name, allocation) in [
                        ("least_trials", Allocation::LeastTrials),
                        (
                            "incumbent_ucb1",
                            Allocation::IncumbentUcb1 { exploration: 0.1 },
                        ),
                    ] {
                        let evidence = regional_trial(RegionalTrial {
                            dimensions: config.dimensions,
                            regions,
                            evaluations: config.evaluations,
                            seed,
                            allocation,
                            sharing,
                            clustered,
                        })?;
                        results.push(json!({"seed": seed, "regions": regions,
                        "sharing": policy_name, "starts": layout, "allocation": allocation_name,
                        "trace": evidence.trace, "seconds_excluding_initialization": evidence.seconds,
                        "phase_timing_semantics": "host wall-clock elapsed; proposal includes queued sharing flush; objective includes Metal dispatch, completion, and scalar readback; tell includes result admission and synchronous sharing work; final_sync is explicit final sharing flush",
                        "proposal_seconds": evidence.proposal_seconds,
                        "objective_seconds": evidence.objective_seconds,
                        "tell_seconds": evidence.tell_seconds,
                        "final_sync_seconds": evidence.final_sync_seconds,
                        "resident_allocation_estimate_bytes": evidence.resident_bytes,
                        "region_evaluations": evidence.region_evaluations, "restarts": evidence.restarts,
                        "delivered_imports": evidence.delivered_imports,
                        "device": evidence.device, "parameter_readbacks": 0,
                        "objective_evaluations": config.evaluations}));
                    }
                }
            }
        }
    }
    Ok(
        json!({"experiment": "resident-regions", "objective": "negative mean ((x*x-1)^2 + 0.05*x)",
        "dimensions": config.dimensions, "results": results}),
    )
}

#[cfg(not(target_os = "macos"))]
fn measure(_config: &Config) -> Result<Value, String> {
    Err("resident-regions currently requires the Metal backend".into())
}
