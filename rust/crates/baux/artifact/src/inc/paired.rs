use deser::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeedLossTrace {
    pub rep: u32,
    pub seed: u64,
    pub init_loss: f64,
    pub final_loss: f64,
    pub accepts: u32,
    pub elapsed_s: f64,
    pub loss_curve: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmInput {
    pub cell_id: String,
    pub arm_id: String,
    pub baseline: bool,
    pub factors: Vec<(String, String)>,
    pub traces: Vec<SeedLossTrace>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedArmSummary {
    pub cell_id: String,
    pub arm_id: String,
    pub baseline: bool,
    pub n_seeds: usize,
    pub init_mean: f64,
    pub init_sem: f64,
    pub loss_mean: f64,
    pub loss_sem: f64,
    pub loss_median: f64,
    pub loss_iqr: f64,
    pub best_seed_loss: f64,
    pub delta_vs_base_mean: f64,
    pub delta_vs_base_sem: f64,
    pub wins: usize,
    pub ties: usize,
    pub losses: usize,
    pub p_raw: f64,
    pub p_holm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairedReport {
    pub env_id: String,
    pub horizon: u32,
    pub common_reps: Vec<u32>,
    pub max_paired_init_drift: f64,
    pub baseline_cell_id: String,
    pub arms: Vec<PairedArmSummary>,
}

fn mean(s: &[f64]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    s.iter().sum::<f64>() / (s.len() as f64)
}

fn sem(s: &[f64], mean_val: f64) -> f64 {
    let n = s.len();
    if n <= 1 {
        return 0.0;
    }
    let var = s.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / ((n - 1) as f64);
    (var / (n as f64)).sqrt()
}

fn median(s: &[f64]) -> f64 {
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) * 0.5
    }
}

fn iqr(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 0.0;
    }
    let (lower, upper) = if n % 2 == 0 {
        (&sorted[..n / 2], &sorted[n / 2..])
    } else {
        (&sorted[..n / 2], &sorted[n / 2 + 1..])
    };
    median(upper) - median(lower)
}

fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

fn flip_pval(diffs: &[f64]) -> f64 {
    let non_zero: Vec<f64> = diffs.iter().copied().filter(|&d| d.abs() > 1e-12).collect();
    let m = non_zero.len();
    if m == 0 {
        return 1.0;
    }
    let obs_sum = non_zero.iter().sum::<f64>();
    let obs_abs = obs_sum.abs();
    if m <= 20 {
        let total = 1u64 << m;
        let mut count = 0u64;
        let threshold = obs_abs - 1e-12;
        for mask in 0..total {
            let mut s = 0.0f64;
            for (k, &d) in non_zero.iter().enumerate() {
                if (mask & (1 << k)) != 0 {
                    s += d;
                } else {
                    s -= d;
                }
            }
            if s.abs() >= threshold {
                count += 1;
            }
        }
        (count as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        let var_sum = non_zero.iter().map(|&d| d * d).sum::<f64>();
        if var_sum <= 1e-24 {
            return 1.0;
        }
        let z = obs_abs / var_sum.sqrt();
        erfc(z / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
    }
}

fn common_reps(horizon: u32, arms: &[ArmInput]) -> Vec<u32> {
    if arms.is_empty() {
        return Vec::new();
    }
    let h = horizon as usize;
    let mut set: BTreeSet<u32> = arms[0]
        .traces
        .iter()
        .filter(|t| t.loss_curve.len() >= h)
        .map(|t| t.rep)
        .collect();
    for arm in &arms[1..] {
        let arm_reps: BTreeSet<u32> = arm
            .traces
            .iter()
            .filter(|t| t.loss_curve.len() >= h)
            .map(|t| t.rep)
            .collect();
        set.retain(|r| arm_reps.contains(r));
    }
    set.into_iter().collect()
}

fn summarize_arm(
    arm: &ArmInput,
    common_reps: &[u32],
    h_idx: usize,
    base_losses: &[f64],
    is_base: bool,
) -> PairedArmSummary {
    let mut inits = Vec::with_capacity(common_reps.len());
    let mut losses = Vec::with_capacity(common_reps.len());
    let mut diffs = Vec::with_capacity(common_reps.len());
    let (mut wins, mut ties, mut losses_cnt) = (0, 0, 0);
    for (i, &r) in common_reps.iter().enumerate() {
        let trace = arm.traces.iter().find(|t| t.rep == r).unwrap();
        let l = trace.loss_curve[h_idx];
        inits.push(trace.init_loss);
        losses.push(l);
        let d = base_losses[i] - l;
        diffs.push(d);
        if d > 1e-12 {
            wins += 1;
        } else if d < -1e-12 {
            losses_cnt += 1;
        } else {
            ties += 1;
        }
    }
    let mut sorted_losses = losses.clone();
    sorted_losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let init_m = mean(&inits);
    let loss_m = mean(&losses);
    let delta_m = mean(&diffs);
    let p_raw = if is_base { 1.0 } else { flip_pval(&diffs) };
    PairedArmSummary {
        cell_id: arm.cell_id.clone(),
        arm_id: arm.arm_id.clone(),
        baseline: is_base,
        n_seeds: common_reps.len(),
        init_mean: init_m,
        init_sem: sem(&inits, init_m),
        loss_mean: loss_m,
        loss_sem: sem(&losses, loss_m),
        loss_median: median(&sorted_losses),
        loss_iqr: iqr(&sorted_losses),
        best_seed_loss: if losses.is_empty() {
            0.0
        } else {
            losses.iter().copied().fold(f64::INFINITY, f64::min)
        },
        delta_vs_base_mean: delta_m,
        delta_vs_base_sem: sem(&diffs, delta_m),
        wins,
        ties,
        losses: losses_cnt,
        p_raw,
        p_holm: 1.0,
    }
}

pub fn paired_report(
    env_id: &str,
    horizon: u32,
    arms: &[ArmInput],
) -> Result<PairedReport, String> {
    if arms.is_empty() {
        return Err("empty arms list".into());
    }
    if horizon == 0 {
        return Err("horizon must be greater than zero".into());
    }
    let common_reps = common_reps(horizon, arms);
    let h_idx = (horizon - 1) as usize;
    let base_idx = arms.iter().position(|a| a.baseline).unwrap_or(0);
    let base_losses: Vec<f64> = common_reps
        .iter()
        .map(|&r| {
            arms[base_idx]
                .traces
                .iter()
                .find(|t| t.rep == r)
                .unwrap()
                .loss_curve[h_idx]
        })
        .collect();
    let mut max_init_drift = 0.0f64;
    for &r in &common_reps {
        let mut min_i = f64::INFINITY;
        let mut max_i = f64::NEG_INFINITY;
        for arm in arms {
            let init_val = arm.traces.iter().find(|t| t.rep == r).unwrap().init_loss;
            min_i = min_i.min(init_val);
            max_i = max_i.max(init_val);
        }
        max_init_drift = max_init_drift.max(max_i - min_i);
    }
    let mut summaries = Vec::with_capacity(arms.len());
    for (idx, arm) in arms.iter().enumerate() {
        summaries.push(summarize_arm(
            arm,
            &common_reps,
            h_idx,
            &base_losses,
            idx == base_idx,
        ));
    }
    let non_base: Vec<usize> = (0..arms.len()).filter(|&i| i != base_idx).collect();
    let k_hyp = non_base.len();
    if k_hyp > 0 {
        let mut ordered = non_base;
        ordered.sort_by(|&a, &b| {
            summaries[a]
                .p_raw
                .partial_cmp(&summaries[b].p_raw)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut running_max = 0.0f64;
        for (rank, &arm_i) in ordered.iter().enumerate() {
            let mult = (k_hyp - rank) as f64;
            let adj = (summaries[arm_i].p_raw * mult).clamp(0.0, 1.0);
            running_max = running_max.max(adj);
            summaries[arm_i].p_holm = running_max.min(1.0);
        }
    }
    Ok(PairedReport {
        env_id: env_id.to_string(),
        horizon,
        baseline_cell_id: arms[base_idx].cell_id.clone(),
        max_paired_init_drift: max_init_drift,
        common_reps,
        arms: summaries,
    })
}

pub fn active_cells(report: &PairedReport, prune_lcb: bool, z_alpha: f64) -> BTreeSet<String> {
    if !prune_lcb || report.common_reps.len() < 3 {
        return report.arms.iter().map(|a| a.cell_id.clone()).collect();
    }
    let mut active = BTreeSet::new();
    active.insert(report.baseline_cell_id.clone());
    for arm in &report.arms {
        if arm.cell_id == report.baseline_cell_id {
            continue;
        }
        let ucb = arm.delta_vs_base_mean + z_alpha * arm.delta_vs_base_sem;
        if ucb >= 0.0 {
            active.insert(arm.cell_id.clone());
        }
    }
    active
}
