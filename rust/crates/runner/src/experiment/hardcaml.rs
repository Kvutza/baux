//! Budgeted nested search on executable, quantized hardware.
mod architecture;
mod flow;
mod matvec;
mod worker;
use architecture::{Config, Design};
use baux::artifact::{
    json::{self, Value, json},
    runs,
};
use baux::{BayesOptimizer, prng};
use baux_tooling::build::BuildSystem;
use deser::Serialize;
use ndarray::Array2;
use rand::Rng;
use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::Path,
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
use worker::Worker;

fn number(value: &Value, key: &str) -> Result<f64, String> {
    value[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("missing or nonfinite worker field {key}"))
}

fn optimizer(
    dimension: usize,
    spec: &baux::config::OptimizerSpec,
    rng: &mut baux::Prng,
) -> Result<BayesOptimizer, String> {
    let bounds = Array2::from_shape_fn((dimension, 2), |(_, j)| j as f64);
    BayesOptimizer::from_spec(bounds, spec, 0, rng).map_err(|e| e.to_string())
}

fn row(values: Vec<f64>) -> Result<Array2<f64>, String> {
    Array2::from_shape_vec((1, values.len()), values).map_err(|e| e.to_string())
}

struct Adapted {
    weights: Vec<f64>,
    loss: f64,
    reference: f64,
    score: f64,
    evaluations: usize,
    cycles: usize,
}

#[derive(Default, Serialize)]
struct OptimizerProfile {
    create_seconds: f64,
    ask_seconds: f64,
    tell_seconds: f64,
}

fn timed<T>(seconds: &mut f64, action: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let start = Instant::now();
    let result = action();
    *seconds += start.elapsed().as_secs_f64();
    result
}

struct Trace {
    file: fs::File,
    evaluations: usize,
    cycles: usize,
    construction_seconds: f64,
    simulation_seconds: f64,
    cache_misses: usize,
    designs: HashSet<Design>,
    inner: OptimizerProfile,
    outer: OptimizerProfile,
}

impl Trace {
    fn new(path: &Path) -> Result<Self, String> {
        Ok(Self {
            file: fs::File::create(path).map_err(|e| e.to_string())?,
            evaluations: 0,
            cycles: 0,
            construction_seconds: 0.0,
            simulation_seconds: 0.0,
            cache_misses: 0,
            designs: HashSet::new(),
            inner: OptimizerProfile::default(),
            outer: OptimizerProfile::default(),
        })
    }

    fn evaluate(
        &mut self,
        worker: &mut Worker,
        d: Design,
        target: u32,
        weights: &[f64],
        context: &Value,
    ) -> Result<Value, String> {
        let start = Instant::now();
        let observation = worker.evaluate(d, target, weights, context)?;
        let cycles = observation["cycles"].as_u64().ok_or("missing cycles")? as usize
            + observation["reset_and_check_cycles"]
                .as_u64()
                .ok_or("missing overhead cycles")? as usize;
        if worker.is_flow() {
            flow::check_cycles(&observation, cycles)?;
        }
        if worker.is_matrix() {
            matvec::check_cycles(&observation, cycles)?;
        }
        if worker
            .expected_cycles(d, weights.len(), context)
            .is_some_and(|expected| cycles != expected)
        {
            return Err("cycle budget disagrees with simulated circuit".into());
        }
        self.evaluations += 1;
        self.designs.insert(d);
        self.cycles += cycles;
        self.construction_seconds += number(&observation["construction"], "seconds")?;
        self.simulation_seconds +=
            number(&observation["simulation_and_reference_check"], "seconds")?;
        self.cache_misses += usize::from(observation["cache_hit"].as_bool() == Some(false));
        json::write_line(
            &mut self.file,
            &json!({"context": context, "evaluation": self.evaluations,
            "roundtrip_seconds": start.elapsed().as_secs_f64(), "observation": observation}),
        )
        .map_err(|e| e.to_string())?;
        Ok(observation)
    }
}

fn adapt(
    config: &Config,
    worker: &mut Worker,
    trace: &mut Trace,
    design: Design,
    target: u32,
    initial: &[f64],
    seed: u64,
    context: &Value,
) -> Result<Adapted, String> {
    let mut rng = prng(seed);
    let mut opt = timed(&mut trace.inner.create_seconds, || {
        optimizer(initial.len(), &config.inner, &mut rng)
    })?;
    let mut count = config.inner_evaluations;
    let mut best = Adapted {
        weights: initial.to_vec(),
        loss: f64::INFINITY,
        reference: f64::INFINITY,
        score: f64::INFINITY,
        evaluations: 0,
        cycles: 0,
    };
    let mut cost = 0;
    while best.evaluations < count {
        let unit = if best.evaluations == 0 {
            row(initial.iter().map(|x| (x + 1.0) * 0.5).collect())?
        } else {
            timed(&mut trace.inner.ask_seconds, || {
                opt.ask(1, &mut rng).map_err(|e| e.to_string())
            })?
        };
        let weights: Vec<_> = unit.iter().map(|x| 2.0 * x - 1.0).collect();
        let before = trace.cycles;
        let observation = trace.evaluate(worker, design, target, &weights, context)?;
        let measured = trace.cycles - before;
        if best.evaluations == 0 {
            cost = measured;
            let reserve = usize::from(config.has_validation());
            let allowed = (config.cycle_budget / cost).saturating_sub(reserve);
            if allowed < 2 {
                return Err(format!(
                    "cycle-budget {} cannot fit {} queries at {} cycles for {:?}",
                    config.cycle_budget,
                    2 + reserve,
                    cost,
                    design
                ));
            }
            count = count.min(allowed);
        } else if measured != cost {
            return Err("data-dependent matrix query cost".into());
        }
        best.evaluations += 1;
        best.cycles += measured;
        let loss = number(&observation, "loss")?;
        timed(&mut trace.inner.tell_seconds, || {
            opt.tell(&unit.view(), &row(vec![-loss])?.view(), &mut rng)
                .map_err(|e| e.to_string())
        })?;
        if loss < best.loss {
            best.weights = weights;
            best.loss = loss;
            best.reference = number(&observation, "reference_loss")?;
        }
    }
    best.score = best.reference;
    if config.has_validation() {
        let mut validation = context.clone();
        validation["phase"] = json!("validation");
        let before = trace.cycles;
        let observed = trace.evaluate(worker, design, target, &best.weights, &validation)?;
        if trace.cycles - before != cost {
            return Err("validation cost differs from training cost".into());
        }
        best.score = number(&observed, "loss")?;
        best.evaluations += 1;
        best.cycles += cost;
    }
    Ok(best)
}

#[derive(Clone, Serialize)]
struct Job {
    dimension: usize,
    shape: Option<matvec::Shape>,
    model: Option<flow::Model>,
    seed: u64,
    budget: usize,
    method: String,
}

impl Job {
    fn candidates(&self, window: usize) -> usize {
        if self.method == "fixed" && window > 0 {
            1
        } else {
            self.budget
        }
    }

    fn uses_baux(&self, window: usize) -> bool {
        self.method == "baux" || (self.method == "fixed" && window == 0)
    }
}

struct SearchResult {
    design: Design,
    adapted: Adapted,
    candidates: Vec<Value>,
    unique_designs: usize,
}

fn search(
    config: &Config,
    job: &Job,
    window: usize,
    worker: &mut Worker,
    trace: &mut Trace,
    weights: &[f64],
    design: Design,
) -> Result<SearchResult, String> {
    let target = config.target_seeds[window];
    let seed = job
        .seed
        .wrapping_add((window as u64).wrapping_mul(1_000_003));
    let mut rng = prng(seed);
    let mut outer = if job.uses_baux(window) {
        Some(timed(&mut trace.outer.create_seconds, || {
            optimizer(config.outer_dimension(), &config.outer, &mut rng)
        })?)
    } else {
        None
    };
    let count = job.candidates(window);
    let mut selected: Option<(Design, Adapted)> = None;
    let mut candidates = Vec::new();
    let mut unique_designs = HashSet::new();
    for candidate in 0..count {
        let unit = if candidate == 0 {
            row(config.encode(design))?
        } else if job.method == "random" {
            row((0..config.outer_dimension())
                .map(|_| rng.gen_range(0.0..1.0))
                .collect())?
        } else {
            timed(&mut trace.outer.ask_seconds, || {
                outer
                    .as_mut()
                    .ok_or("missing outer optimizer")?
                    .ask(1, &mut rng)
                    .map_err(|e| e.to_string())
            })?
        };
        let values: Vec<_> = unit.iter().copied().collect();
        let proposed = config.decode(&values);
        unique_designs.insert(proposed);
        // Every candidate starts from the same deployed weights and RNG
        // seed.
        let adapted = adapt(
            config,
            worker,
            trace,
            proposed,
            target,
            weights,
            seed,
            &json!({"window": window, "phase": "inner", "candidate": candidate}),
        )?;
        candidates.push(json!({"design": proposed, "loss": adapted.loss,
                "reference_loss": adapted.reference, "selection_loss": adapted.score, "evaluations": adapted.evaluations, "cycles": adapted.cycles}));
        if let Some(outer) = &mut outer {
            timed(&mut trace.outer.tell_seconds, || {
                outer
                    .tell(&unit.view(), &row(vec![-adapted.score])?.view(), &mut rng)
                    .map_err(|e| e.to_string())
            })?;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, previous)| adapted.score < previous.score)
        {
            selected = Some((proposed, adapted));
        }
    }
    let (design, adapted) = selected.ok_or("empty candidate set")?;
    Ok(SearchResult {
        design,
        adapted,
        candidates,
        unique_designs: unique_designs.len(),
    })
}

fn trial(config: &Config, job: &Job, binary: &Path, directory: &Path) -> Result<Value, String> {
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let mut worker = Worker::spawn(binary, config, job.shape, job.model, job.seed)?;
    let startup_seconds = start.elapsed().as_secs_f64();
    let mut trace = Trace::new(&directory.join("evaluations.jsonl"))?;
    let mut windows =
        fs::File::create(directory.join("windows.jsonl")).map_err(|e| e.to_string())?;
    let mut weights = vec![0.0; job.dimension];
    let mut design = config.decode(&vec![0.5; config.outer_dimension()]);
    let mut cumulative_prequential_loss = 0.0;
    let mut cumulative_hardware_loss = 0.0;
    let mut switches = 0;
    for (window, &target) in config.target_seeds.iter().enumerate() {
        let window_start = Instant::now();
        let before = trace.evaluate(
            &mut worker,
            design,
            target,
            &weights,
            &json!({"window": window, "phase": "prequential"}),
        )?;
        cumulative_prequential_loss += number(&before, "reference_loss")?;
        cumulative_hardware_loss += number(&before, "loss")?;
        let selected = search(
            config,
            job,
            window,
            &mut worker,
            &mut trace,
            &weights,
            design,
        )?;
        let next_design = selected.design;
        let adapted = selected.adapted;
        switches += usize::from(next_design != design);
        design = next_design;
        weights = adapted.weights;
        json::write_line(
            &mut windows,
            &json!({"window": window, "target_seed": target,
            "prequential_reference_loss": number(&before, "reference_loss")?,
            "prequential_hardware_loss": number(&before, "loss")?,
            "selection_loss": adapted.score, "adapted_reference_loss": adapted.reference, "design": design,
            "candidates": selected.candidates, "unique_designs": selected.unique_designs,
            "host_seconds": window_start.elapsed().as_secs_f64()}),
        )
        .map_err(|e| e.to_string())?;
        windows.flush().map_err(|e| e.to_string())?;
        trace.file.flush().map_err(|e| e.to_string())?;
    }
    let generation = worker.generate(
        design,
        &weights,
        *config.target_seeds.last().ok_or("missing regime")?,
        config.target_seeds.len() - 1,
        directory,
    )?;
    let export_seconds = worker.export_rtl(design, directory)?;
    let summary = json!({"job": job, "generation": generation, "rtl_export_seconds": export_seconds, "host_seconds": start.elapsed().as_secs_f64(),
        "worker_startup_seconds": startup_seconds, "evaluations": trace.evaluations,
        "simulated_cycles_including_reset_and_checks": trace.cycles,
        "cumulative_prequential_reference_loss": cumulative_prequential_loss,
        "cumulative_prequential_hardware_loss": cumulative_hardware_loss,
        "design_switches": switches, "cache_misses": trace.cache_misses,
        "unique_designs": trace.designs.len(),
        "inner_optimizer": trace.inner, "outer_optimizer": trace.outer,
        "construction_seconds": trace.construction_seconds,
        "simulation_and_reference_check_seconds": trace.simulation_seconds,
        "final_design": design,
        "final_weights": if config.flow.is_some() { json!(null) } else { json!(weights) },
        "final_coordinates": if config.flow.is_some() { json!(weights) } else { json!(null) }});
    fs::write(
        directory.join("summary.json"),
        json::pretty_string(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(summary)
}

fn jobs(config: &Config) -> Vec<Job> {
    let mut jobs = Vec::new();
    let shapes: Vec<_> = match &config.matrix {
        Some(m) => m
            .shapes
            .iter()
            .map(|&s| (s.rows * s.cols, Some(s), None))
            .collect(),
        None => config.dimensions.iter().map(|&d| (d, None, None)).collect(),
    };
    let shapes = if let Some(flow) = &config.flow {
        flow.models
            .iter()
            .map(|&m| (m.parameters(), None, Some(m)))
            .collect()
    } else {
        shapes
    };
    for (dimension, shape, model) in shapes {
        for &seed in &config.seeds {
            for &budget in &config.budgets {
                jobs.extend(config.methods.iter().map(|method| Job {
                    dimension,
                    shape,
                    model,
                    seed,
                    budget,
                    method: method.clone(),
                }));
            }
        }
    }
    jobs
}

fn run(config: &Config, binary: &Path, output: &Path) -> Result<(), String> {
    let jobs = jobs(config);
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..config.workers.min(jobs.len()) {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else {
                        break;
                    };
                    let result = trial(
                        config,
                        job,
                        binary,
                        &output.join(format!("trial-{index:05}")),
                    );
                    eprintln!(
                        "Hardcaml {}/{}: {} d={} budget={} {}",
                        index + 1,
                        jobs.len(),
                        job.method,
                        job.dimension,
                        job.budget,
                        if result.is_ok() { "complete" } else { "failed" }
                    );
                    match results.lock() {
                        Ok(mut results) => results.push((index, result)),
                        Err(_) => break,
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().map_err(|e| e.to_string())?;
    results.sort_by_key(|(index, _)| *index);
    let mut summaries = Vec::new();
    for (index, result) in results {
        summaries.push(result.map_err(|e| format!("trial {index}: {e}"))?);
    }
    if summaries.len() != jobs.len() {
        return Err("incomplete trial set".into());
    }
    let summary = json!({"schema": if config.flow.is_some() { "baux.hardcaml-flowlob.v1" } else if config.has_validation() { "baux.hardcaml-matvec.v1" } else { "baux.hardcaml-nested-scaling.v1" }, "workers": config.workers,
        "wall_seconds": start.elapsed().as_secs_f64(), "trials": summaries,
        "scope": if config.flow.is_some() { "paper-based FlowLOB transformer and ODE sampler; host-orchestrated RTL kernels or integer simulation; synthetic data unless dataset provided; no market calibration or physical FPGA claims" } else if config.has_validation() { "programmable banked matrix-vector accelerator; changing synthetic linear teacher and IO schedule; not FlowLOB or a calibrated market" } else { "synthetic moving-target distance diagnostic; not FlowLOB or a calibrated market" },
        "objective": if config.flow.is_some() { "inner minimizes conditional flow-matching training loss over all parameter coordinates; outer minimizes held-out quantized validation loss; generated trajectories and book metrics reported separately" } else if config.has_validation() { "inner minimizes hardware training MSE; outer minimizes held-out hardware validation MSE under a cycle budget; test MSE measured before adaptation" } else { "inner maximizes negative simulated quantized loss; outer maximizes negative final float64 reference loss" },
        "budget": "cycle-budget per candidate includes all training and validation queries; inner-evaluations caps training queries; prequential test queries are separately reported; reset, weight programming, input programming and output drain included for matrix circuits",
        "protocol": "evaluate deployed weights before adaptation; fresh optimizer histories each window; carry selected weights and design; candidate branches share initial weights and seed",
        "switching": "host construction cost measured; no modeled FPGA deployment cost or physical speedup",
        "baseline": "fixed receives the same initial BAUX hardware calibration as baux, then freezes its design and adapts weights only; random and baux continue hardware search each window; actual costs reported",
        "allocation": "OCaml GC allocation counters and heap capacity per phase; excludes native allocations and is not process peak RSS",
        "cache": "bounded per-trial LRU of arithmetic backends; integer FlowLOB mode elaborates RTL only for export; no cached objective values"});
    fs::write(
        output.join("summary.json"),
        json::pretty_string(&summary).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn provenance(output: &Path) -> Result<(), String> {
    let switch = std::env::var("BAUX_OCAML_SWITCH").unwrap_or_else(|_| "5.2.0+ox".into());
    for (name, arguments) in [
        (
            "compiler.txt",
            vec![
                "exec",
                "--switch",
                switch.as_str(),
                "--",
                "ocamlopt",
                "-config",
            ],
        ),
        (
            "packages.txt",
            vec![
                "list",
                "--switch",
                switch.as_str(),
                "--installed",
                "--columns=name,version",
            ],
        ),
    ] {
        let result = Command::new("opam")
            .args(arguments)
            .output()
            .map_err(|e| e.to_string())?;
        if !result.status.success() {
            return Err(format!("cannot record {name}"));
        }
        fs::write(output.join(name), result.stdout).map_err(|e| e.to_string())?;
    }
    let system = Command::new("uname")
        .arg("-a")
        .output()
        .map_err(|e| e.to_string())?;
    if !system.status.success() {
        return Err("cannot record host platform".into());
    }
    fs::write(output.join("host.txt"), system.stdout).map_err(|e| e.to_string())
}

fn validate(config: &Config) -> Result<(), String> {
    config.validate()?;
    optimizer(config.outer_dimension(), &config.outer, &mut prng(0))?;
    optimizer(jobs(config)[0].dimension, &config.inner, &mut prng(0))?;
    Ok(())
}

pub(super) fn execute(
    root: &Path,
    path: &Path,
    rounds: Option<u32>,
    engine: BuildSystem,
) -> Result<bool, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    if !matches!(
        value["experiment"].as_str(),
        Some("hardcaml-nested-scaling" | "hardcaml-matvec" | "hardcaml-flowlob")
    ) {
        return Ok(false);
    }
    if rounds.is_some() {
        return Err("set budgets, inner-evaluations and target-seeds in the scaling TOML".into());
    }
    let mut config: Config = baux::artifact::toml::from_str(&text).map_err(|e| e.to_string())?;
    validate(&config)?;
    engine.run(root, "//ocaml:tool", &["hardware-build".into()])?;
    let binary = root.join(".cache/baux/ocaml-build/default/bin/hardware_bench.exe");
    let commit = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let output = root
        .join(".cache/baux/runs/hardcaml")
        .join(nonce.to_string());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let binary = Worker::snapshot(&binary, &output)?;
    if let Some(flow) = &mut config.flow {
        flow.snapshot_inputs(&output)?;
    }
    fs::write(output.join("requested.toml"), &text).map_err(|e| e.to_string())?;
    let resolved = baux::artifact::toml::pretty_string(&config).map_err(|e| e.to_string())?;
    fs::write(output.join("config.toml"), &resolved).map_err(|e| e.to_string())?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut record = runs::start_worker(root, &output, &resolved, commit.trim(), &executable)?;
    record.model = if config.flow.is_some() {
        "hardcaml-flowlob-reproduction"
    } else if config.has_validation() {
        "hardcaml-programmable-matvec"
    } else {
        "hardcaml-moving-distance"
    }
    .into();
    record.optimizer = "nested-baux-with-baselines".into();
    record.recipe = "simulation-scaling".into();
    record.protocol = runs::digest(resolved.as_bytes());
    let result = provenance(&output).and_then(|()| run(&config, &binary, &output));
    runs::finish(root, &mut record, &result)?;
    println!("Hardcaml scaling evidence: {}", output.display());
    result?;
    Ok(true)
}

#[cfg(test)]
mod tests;
