//! Incremental stabilization engine with Paired-LCB rungs and Singleflight
//! broker.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use baux::artifact::inc;
use baux::artifact::study;
use baux::artifact::toml;

use super::broker::EvalBroker;
use super::digest;
use super::import;
use super::observer;
use crate::experiment::julia::config::Config as JuliaConfig;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::experiment::julia::metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::experiment::julia::metal_step;

pub enum RunnerFiber {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    Metal(metal_step::MetalFiber),
    Synthetic {
        round: usize,
        best_score: f64,
        best_viol: f64,
        num_dim: usize,
    },
}

impl RunnerFiber {
    fn prepare_proposal(&mut self) -> Result<Vec<u16>, String> {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(m) => m.prepare_proposal(),
            Self::Synthetic { round, num_dim, .. } => {
                let r = (*round + 1) as u16;
                Ok(vec![r.wrapping_mul(1009); *num_dim])
            }
        }
    }

    fn commit_eval(
        &mut self,
        outcome: &inc::EvalOutcome,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<inc::StepRecord, String> {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(m) => m.commit_eval(outcome, loss_fn),
            Self::Synthetic {
                round,
                best_score,
                best_viol,
                ..
            } => {
                *round += 1;
                let sc = outcome.scores[0];
                let accepted = sc > *best_score;
                if accepted {
                    *best_score = sc;
                    *best_viol = outcome.viol;
                }
                Ok(inc::StepRecord {
                    eval_scores: outcome.scores.clone(),
                    eval_viol: outcome.viol,
                    best_score: *best_score,
                    best_viol: *best_viol,
                    accepted,
                    cost_us: outcome.cost_us,
                })
            }
        }
    }

    fn replay_prefix(
        &mut self,
        steps: &[inc::StepRecord],
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<(), String> {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Metal(m) => m.replay_prefix(steps, loss_fn),
            Self::Synthetic {
                round,
                best_score,
                best_viol,
                ..
            } => {
                *round = steps.len();
                if let Some(last) = steps.last() {
                    *best_score = last.best_score;
                    *best_viol = last.best_viol;
                }
                Ok(())
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn metal_spec(
    root: &Path,
    cell: &study::StudyCell,
    rep: u32,
    target_rounds: usize,
) -> Result<metal_step::MetalJobSpec, String> {
    let cfg: JuliaConfig = toml::from_str(&cell.canonical_toml).map_err(|e| e.to_string())?;
    let (num_dim, n_obj) = if cell.experiment.starts_with("tokamak") {
        if cell.experiment == "tokamak-tokagym" {
            (baux::tokamak::TOKAGYM_PARAMS, 1)
        } else {
            (baux::tokamak::TOTAL_PARAMS, 1)
        }
    } else {
        let (domain, _, _, _) = crate::experiment::julia::space::resolve_space(root, &cfg)?;
        (
            domain.num_dim(),
            crate::experiment::julia::config::obj_count(&cfg),
        )
    };
    let seed = baux::tune_seed(&cell.experiment, "proposal", rep, 0);
    let cfg_opt = crate::experiment::julia::driver::opt_cfg(&cfg, n_obj)?;
    let enn_res = metal::res_enn(&cfg, &cfg_opt, seed)?;
    let tr_cfg = match &cfg_opt.trust_region {
        baux::trust_region::TrustRegionConfig::Turbo(l) => *l,
        baux::trust_region::TrustRegionConfig::Morbo(m) => m.length,
    };
    let has_surrogate = matches!(
        cfg_opt.surrogate,
        baux::optimizer_config::SurrogateConfig::ENN(_)
    );
    Ok(metal_step::MetalJobSpec {
        rep,
        seed,
        num_dim,
        n_obj,
        rounds: target_rounds,
        has_surrogate,
        tr_cfg,
        kernel: metal::resolve_kernel(&cfg.optimizer),
        reg_spec: metal::regional_spec(&cfg.optimizer),
        enn_res,
        policy: metal::objective_policy(&cfg, n_obj, seed),
    })
}

fn create_fiber(
    root: &Path,
    cell: &study::StudyCell,
    rep: u32,
    target_rounds: usize,
) -> Result<(RunnerFiber, u64, usize), String> {
    let seed = baux::tune_seed(&cell.experiment, "proposal", rep, 0);
    if std::env::var("BAUX_INCREMENTAL_SYNTHETIC").as_deref() == Ok("1") {
        return Ok((
            RunnerFiber::Synthetic {
                round: 0,
                best_score: f64::NEG_INFINITY,
                best_viol: 0.0,
                num_dim: 16,
            },
            seed,
            16,
        ));
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let spec = metal_spec(root, cell, rep, target_rounds)?;
        let dim = spec.num_dim;
        Ok((
            RunnerFiber::Metal(metal_step::MetalFiber::new(spec)),
            seed,
            dim,
        ))
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (root, cell, rep, target_rounds);
        Ok((
            RunnerFiber::Synthetic {
                round: 0,
                best_score: f64::NEG_INFINITY,
                best_viol: 0.0,
                num_dim: 16,
            },
            seed,
            16,
        ))
    }
}

pub fn plan_execution(
    root: &Path,
    store: &inc::IncStore,
    study: &study::CompiledStudy,
) -> Result<(), String> {
    let mut stats = observer::PlanStats::default();
    let mut rows = Vec::with_capacity(study.cells.len());
    for cell in &study.cells {
        let env_d = digest::env_digest(root, cell)?;
        let opt_d = digest::opt_digest(root, cell)?;
        let (mut cutoff, mut resume, mut cold) = (0, 0, 0);
        for rep in 0..cell.repetitions {
            let key = inc::traj_key(&env_d, &opt_d, rep);
            if let Ok(Some(head)) = store.get_head(&key) {
                if head.completed_round >= cell.rounds {
                    cutoff += 1;
                } else if head.completed_round > 0 && head.has_eval_trace {
                    resume += 1;
                } else {
                    cold += 1;
                }
            } else {
                cold += 1;
            }
        }
        stats.total_cells += 1;
        stats.total_trajectories += cell.repetitions as usize;
        stats.cutoff_hits += cutoff;
        stats.suffix_resumes += resume;
        stats.cold_runs += cold;
        rows.push(observer::PlanRow {
            cell_id: cell.cell_id.clone(),
            reps: cell.repetitions,
            cutoff,
            resume,
            cold,
        });
    }
    observer::print_plan(&rows, &stats);
    Ok(())
}

fn determine_rungs(configured: &[u32], target_rounds: u32) -> Vec<u32> {
    let mut rungs: Vec<u32> = if configured.is_empty() {
        vec![target_rounds]
    } else {
        configured.to_vec()
    };
    rungs.retain(|&r| r > 0 && r < target_rounds);
    rungs.push(target_rounds);
    rungs.sort_unstable();
    rungs.dedup();
    rungs
}

struct FiberSlot {
    cell_idx: usize,
    rep: u32,
    seed: u64,
    num_dim: usize,
    fiber: RunnerFiber,
    cur_round: usize,
    initial_error: f64,
    total_accepts: u32,
    elapsed_s: f64,
    worker_s: f64,
    traj_key: String,
    env_d: String,
    opt_d: String,
    buffered: Vec<inc::StepRecord>,
    flush_start: u32,
}

fn flush_fiber(store: &inc::IncStore, s: &mut FiberSlot) -> Result<(), String> {
    if s.buffered.is_empty() {
        return Ok(());
    }
    let last = s.buffered.last().cloned().unwrap();
    let head = inc::TrajHead {
        env_digest: s.env_d.clone(),
        opt_digest: s.opt_d.clone(),
        rep: s.rep,
        seed: s.seed,
        num_dim: s.num_dim,
        completed_round: s.cur_round as u32,
        has_eval_trace: true,
        initial_error: s.initial_error,
        best_score: last.best_score,
        total_accepts: s.total_accepts,
        elapsed_s: s.elapsed_s,
        worker_s: s.worker_s,
    };
    store.append_steps(&s.traj_key, &head, s.flush_start, &s.buffered)?;
    s.buffered.clear();
    s.flush_start = (s.cur_round + 1) as u32;
    Ok(())
}

fn step_fibers(
    root: &Path,
    store: &inc::IncStore,
    broker: &EvalBroker,
    cells: &[study::StudyCell],
    slots: &mut [FiberSlot],
    rung: u32,
) -> Result<(), String> {
    let target = rung as usize;
    let exp = cells[0].experiment.clone();
    loop {
        let step_t0 = Instant::now();
        let mut proposals = Vec::new();
        for (i, slot) in slots.iter_mut().enumerate() {
            if slot.cur_round < target {
                proposals.push((i, slot.fiber.prepare_proposal()?));
            }
        }
        if proposals.is_empty() {
            break;
        }
        let env_d = slots[proposals[0].0].env_d.clone();
        let c_ref = &cells[slots[proposals[0].0].cell_idx];
        let outcomes = broker.evaluate_batch(root, store, &env_d, c_ref, &proposals)?;
        let step_dt = step_t0.elapsed().as_secs_f64() / (outcomes.len().max(1) as f64);
        let loss_fn = |sc| import::score_loss(&exp, sc);
        for (idx, outcome) in outcomes {
            let slot = &mut slots[idx];
            let rec = slot.fiber.commit_eval(&outcome, &loss_fn)?;
            slot.cur_round += 1;
            if slot.cur_round == 1 {
                slot.initial_error = loss_fn(rec.best_score);
            }
            if rec.accepted {
                slot.total_accepts += 1;
            }
            slot.elapsed_s += step_dt;
            slot.worker_s += (outcome.cost_us as f64) * 1.0e-6;
            slot.buffered.push(rec);
            if slot.buffered.len() >= 100 || slot.cur_round == target {
                flush_fiber(store, slot)?;
            }
        }
    }
    Ok(())
}

fn collect_slots(
    root: &Path,
    store: &inc::IncStore,
    cells: &[study::StudyCell],
    active_cells: &BTreeSet<String>,
    rung: u32,
) -> Result<Vec<FiberSlot>, String> {
    let mut slots = Vec::new();
    for (c_idx, cell) in cells.iter().enumerate() {
        if !active_cells.contains(&cell.cell_id) {
            continue;
        }
        let env_d = digest::env_digest(root, cell)?;
        let opt_d = digest::opt_digest(root, cell)?;
        let loss_fn = |sc| import::score_loss(&cell.experiment, sc);
        for rep in 0..cell.repetitions {
            let key = inc::traj_key(&env_d, &opt_d, rep);
            if let Ok(Some(slice)) = store.slice_trajectory(&key, rung)
                && slice.horizon >= rung
            {
                continue;
            }
            let (mut fiber, seed, num_dim) = create_fiber(root, cell, rep, cell.rounds as usize)?;
            let (mut start_round, mut init_err, mut accepts, mut el_s, mut wk_s) =
                (1u32, 0.0, 0u32, 0.0, 0.0);
            if let Ok(Some((head, steps))) = store.load_prefix(&key)
                && head.has_eval_trace
                && head.completed_round > 0
            {
                fiber.replay_prefix(&steps, &loss_fn)?;
                start_round = head.completed_round + 1;
                init_err = head.initial_error;
                accepts = head.total_accepts;
                el_s = head.elapsed_s;
                wk_s = head.worker_s;
            }
            slots.push(FiberSlot {
                cell_idx: c_idx,
                rep,
                seed,
                num_dim,
                fiber,
                cur_round: (start_round - 1) as usize,
                initial_error: init_err,
                total_accepts: accepts,
                elapsed_s: el_s,
                worker_s: wk_s,
                traj_key: key,
                env_d: env_d.clone(),
                opt_d: opt_d.clone(),
                buffered: Vec::new(),
                flush_start: start_round,
            });
        }
    }
    Ok(slots)
}

fn collect_arms(
    root: &Path,
    store: &inc::IncStore,
    cells: &[study::StudyCell],
    active_cells: &BTreeSet<String>,
    rung: u32,
) -> Result<Vec<inc::ArmInput>, String> {
    let mut inputs = Vec::new();
    for cell in cells {
        if !active_cells.contains(&cell.cell_id) {
            continue;
        }
        let env_d = digest::env_digest(root, cell)?;
        let opt_d = digest::opt_digest(root, cell)?;
        let mut traces = Vec::new();
        for rep in 0..cell.repetitions {
            let key = inc::traj_key(&env_d, &opt_d, rep);
            if let Ok(Some(slice)) = store.slice_trajectory(&key, rung) {
                let loss_curve: Vec<f64> = slice
                    .best_scores
                    .iter()
                    .map(|&sc| import::score_loss(&cell.experiment, sc))
                    .collect();
                let final_loss = *loss_curve
                    .last()
                    .unwrap_or(&import::score_loss(&cell.experiment, slice.final_score));
                traces.push(inc::SeedLossTrace {
                    rep,
                    seed: slice.head.seed,
                    init_loss: slice.initial_error,
                    final_loss,
                    accepts: slice.accepts_at_horizon,
                    elapsed_s: slice.head.elapsed_s,
                    loss_curve,
                });
            }
        }
        inputs.push(inc::ArmInput {
            cell_id: cell.cell_id.clone(),
            arm_id: cell.arm_id.clone(),
            baseline: cell.baseline,
            factors: cell.factors.clone(),
            traces,
        });
    }
    Ok(inputs)
}

fn run_rungs(
    root: &Path,
    store: &inc::IncStore,
    cells: &[study::StudyCell],
    study: &study::CompiledStudy,
    jobs: Option<usize>,
) -> Result<Option<(inc::PairedReport, Vec<inc::ArmInput>)>, String> {
    let env_id = &cells[0].env_id;
    let broker = EvalBroker::new(&cells[0].experiment, jobs);
    let target_rounds = cells.iter().map(|c| c.rounds).max().unwrap_or(1000);
    let rungs = determine_rungs(&study.rungs, target_rounds);
    let mut active_cells: BTreeSet<String> = cells.iter().map(|c| c.cell_id.clone()).collect();
    let mut final_out = None;

    for &rung in &rungs {
        let mut slots = collect_slots(root, store, cells, &active_cells, rung)?;
        if !slots.is_empty() {
            step_fibers(root, store, &broker, cells, &mut slots, rung)?;
        }
        let arm_inputs = collect_arms(root, store, cells, &active_cells, rung)?;
        let report = inc::paired_report(env_id, rung, &arm_inputs)?;
        let should_prune = study.prune == study::StudyPrune::PairedLcb && rung < target_rounds;
        let next_active = inc::active_cells(&report, should_prune, 1.96);
        let pruned: Vec<String> = active_cells.difference(&next_active).cloned().collect();
        observer::print_rung(&report, &next_active, &pruned);
        active_cells = next_active;
        final_out = Some((report, arm_inputs));
    }
    Ok(final_out)
}

pub fn run_study(
    root: &Path,
    store: &inc::IncStore,
    study: &study::CompiledStudy,
    jobs: Option<usize>,
    plot: bool,
) -> Result<(), String> {
    let mut by_env: BTreeMap<String, Vec<study::StudyCell>> = BTreeMap::new();
    for cell in &study.cells {
        by_env
            .entry(cell.env_id.clone())
            .or_default()
            .push(cell.clone());
    }
    for (_, cells) in by_env {
        if let Some((rep, arm_inputs)) = run_rungs(root, store, &cells, study, jobs)? {
            let run_dir =
                observer::write_artifacts(root, &rep, &arm_inputs, &cells[0].experiment, study)?;
            println!("Wrote incremental run: {}", run_dir.display());
            if plot {
                let written = super::plot_web::emit_plots(
                    root,
                    &run_dir,
                    &rep,
                    &arm_inputs,
                    &cells[0].experiment,
                )?;
                for path in written {
                    println!("Wrote benchmark plot: {}", path.display());
                }
            }
        }
    }
    Ok(())
}
