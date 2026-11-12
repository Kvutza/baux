//! Terminal plan/rung formatting, run artifact persistence, and plot data
//! emission.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use baux::artifact::inc::{ArmInput, PairedReport};
use baux::artifact::json;
use baux::artifact::runs;
use baux::artifact::study::CompiledStudy;
use deser::Serialize;

#[derive(Default)]
pub struct PlanStats {
    pub total_cells: usize,
    pub total_trajectories: usize,
    pub cutoff_hits: usize,
    pub suffix_resumes: usize,
    pub cold_runs: usize,
}

pub struct PlanRow {
    pub cell_id: String,
    pub reps: u32,
    pub cutoff: usize,
    pub resume: usize,
    pub cold: usize,
}

#[derive(Serialize)]
struct PlotArmEntry {
    mean: Vec<f64>,
    sem: Vec<f64>,
    paired_sem: Vec<f64>,
    diff_vs_ref_seeds: Vec<f64>,
    diff_mean: f64,
    diff_sem: f64,
    diff_ci95: f64,
}

#[derive(Serialize)]
struct PlotData<'a> {
    experiment: &'a str,
    rounds: u32,
    n_paired_seeds: usize,
    configs: BTreeMap<String, PlotArmEntry>,
}

pub fn print_plan(rows: &[PlanRow], stats: &PlanStats) {
    println!("\nIncremental Execution Plan:");
    println!(
        "{:<48} {:>5} {:>8} {:>8} {:>8}",
        "Cell", "Reps", "Cutoff", "Resume", "Cold"
    );
    println!("{}", "-".repeat(81));
    for r in rows {
        println!(
            "{:<48} {:>5} {:>8} {:>8} {:>8}",
            r.cell_id, r.reps, r.cutoff, r.resume, r.cold
        );
    }
    println!("{}", "-".repeat(81));
    println!(
        "Total: {} cells, {} trajectories ({} cutoff, {} resume, {} cold)\n",
        stats.total_cells,
        stats.total_trajectories,
        stats.cutoff_hits,
        stats.suffix_resumes,
        stats.cold_runs
    );
}

pub fn print_rung(report: &PairedReport, active_cells: &BTreeSet<String>, pruned: &[String]) {
    println!(
        "\n[Rung {}] env={} n={} init_drift={:.2e} active={}",
        report.horizon,
        report.env_id,
        report.common_reps.len(),
        report.max_paired_init_drift,
        active_cells.len()
    );
    println!(
        "{:<36} {:>16} {:>12} {:>9} {:>9}",
        "Arm", "Loss (mean±sem)", "Δ vs Base", "p_Holm", "W/T/L"
    );
    println!("{}", "-".repeat(86));
    for arm in &report.arms {
        let slug = arm
            .cell_id
            .strip_prefix(&format!("{}__", report.env_id))
            .unwrap_or(&arm.cell_id);
        let loss_str = format!("{:.4}±{:.4}", arm.loss_mean, arm.loss_sem);
        let wt_str = format!("{}/{}/{}", arm.wins, arm.ties, arm.losses);
        println!(
            "{:<36} {:>16} {:>+12.4} {:>9.4} {:>9}",
            slug, loss_str, arm.delta_vs_base_mean, arm.p_holm, wt_str
        );
    }
    if !pruned.is_empty() {
        println!("  pruned (paired-lcb): {}", pruned.join(", "));
    }
}

fn mean_sem(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let m = xs.iter().sum::<f64>() / (xs.len() as f64);
    if xs.len() <= 1 {
        return (m, 0.0);
    }
    let var = xs.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / ((xs.len() - 1) as f64);
    (m, (var / (xs.len() as f64)).sqrt())
}

fn arm_entry(arm: &ArmInput, base: &ArmInput, reps: &[u32], horizon: usize) -> PlotArmEntry {
    let mut mean = Vec::with_capacity(horizon);
    let mut sem = Vec::with_capacity(horizon);
    let mut paired_sem = Vec::with_capacity(horizon);
    for t in 0..horizon {
        let vals: Vec<f64> = reps
            .iter()
            .filter_map(|&r| arm.traces.iter().find(|tr| tr.rep == r))
            .map(|tr| tr.loss_curve[t])
            .collect();
        let diffs: Vec<f64> = reps
            .iter()
            .filter_map(|&r| {
                let a = arm.traces.iter().find(|tr| tr.rep == r)?.loss_curve[t];
                let b = base.traces.iter().find(|tr| tr.rep == r)?.loss_curve[t];
                Some(a - b)
            })
            .collect();
        let (m, s) = mean_sem(&vals);
        let (_, ps) = mean_sem(&diffs);
        mean.push(m);
        sem.push(s);
        paired_sem.push(ps);
    }
    let diff_vs_ref_seeds: Vec<f64> = reps
        .iter()
        .filter_map(|&r| {
            let a = arm.traces.iter().find(|tr| tr.rep == r)?.loss_curve[horizon - 1];
            let b = base.traces.iter().find(|tr| tr.rep == r)?.loss_curve[horizon - 1];
            Some(a - b)
        })
        .collect();
    let (diff_mean, diff_sem) = mean_sem(&diff_vs_ref_seeds);
    PlotArmEntry {
        mean,
        sem,
        paired_sem,
        diff_vs_ref_seeds,
        diff_mean,
        diff_sem,
        diff_ci95: 1.96 * diff_sem,
    }
}

pub fn write_artifacts(
    root: &Path,
    report: &PairedReport,
    arm_inputs: &[ArmInput],
    exp: &str,
    study: &CompiledStudy,
) -> Result<PathBuf, String> {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let run_dir = root
        .join(".cache/baux/runs/incremental")
        .join(format!("{exp}-{nonce}"));
    fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;

    let report_raw = json::to_string(report).map_err(|e| e.to_string())?;
    fs::write(run_dir.join("report.json"), report_raw).map_err(|e| e.to_string())?;

    let base_idx = arm_inputs.iter().position(|a| a.baseline).unwrap_or(0);
    let mut configs = BTreeMap::new();
    if !arm_inputs.is_empty() && report.horizon > 0 {
        for arm in arm_inputs {
            let slug = arm
                .cell_id
                .strip_prefix(&format!("{}__", report.env_id))
                .unwrap_or(&arm.cell_id);
            let v = arm_entry(
                arm,
                &arm_inputs[base_idx],
                &report.common_reps,
                report.horizon as usize,
            );
            configs.insert(slug.to_string(), v);
        }
    }
    let plot = PlotData {
        experiment: exp,
        rounds: report.horizon,
        n_paired_seeds: report.common_reps.len(),
        configs,
    };
    let plot_raw = json::to_string(&plot).map_err(|e| e.to_string())?;
    fs::write(run_dir.join("plot_data.json"), plot_raw).map_err(|e| e.to_string())?;

    let resolved = study
        .cells
        .first()
        .map(|c| c.canonical_toml.as_str())
        .unwrap_or_default();
    fs::write(run_dir.join("config.toml"), resolved).map_err(|e| e.to_string())?;

    if let Ok(commit) = runs::jj(root, &["log", "-r", "@", "--no-graph", "-T", "commit_id"])
        && let Ok(exe) = std::env::current_exe()
        && let Ok(mut record) = runs::start_worker(root, &run_dir, resolved, commit.trim(), &exe)
    {
        record.model = exp.to_string();
        record.optimizer = format!("incremental-study({}-arms)", report.arms.len());
        record.protocol = runs::digest(resolved.as_bytes());
        let ok = Ok(());
        let _ = runs::finish(root, &mut record, &ok);
    }
    Ok(run_dir)
}
