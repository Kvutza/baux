//! Benchmark trajectory series extraction, baseline collapsing, downsampling,
//! and right-margin label collision repulsion.

use baux::artifact::inc::{ArmInput, PairedReport};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkupMode {
    Forester,
    SvgXml,
}

pub struct MarkupBuf {
    pub mode: MarkupMode,
    pub out: String,
}

impl MarkupBuf {
    pub fn new(mode: MarkupMode) -> Self {
        Self {
            mode,
            out: String::with_capacity(16384),
        }
    }

    pub fn push_raw(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn write_attrs(&mut self, attrs: &[(&str, &str)]) {
        for &(k, v) in attrs {
            match self.mode {
                MarkupMode::Forester => {
                    self.out.push('[');
                    self.out.push_str(k);
                    self.out.push_str("]{");
                    self.out.push_str(&sanitize_atom(v));
                    self.out.push('}');
                }
                MarkupMode::SvgXml => {
                    self.out.push(' ');
                    self.out.push_str(k);
                    self.out.push_str("=\"");
                    self.out.push_str(&escape_xml(v));
                    self.out.push('"');
                }
            }
        }
    }

    pub fn leaf(&mut self, tag: &str, attrs: &[(&str, &str)]) {
        match self.mode {
            MarkupMode::Forester => {
                self.out.push_str("\\<");
                self.out.push_str(tag);
                self.out.push('>');
                self.write_attrs(attrs);
                self.out.push_str("{}\n");
            }
            MarkupMode::SvgXml => {
                self.out.push('<');
                self.out.push_str(tag);
                self.write_attrs(attrs);
                self.out.push_str(" />\n");
            }
        }
    }

    pub fn text_elem(&mut self, tag: &str, attrs: &[(&str, &str)], text: &str) {
        match self.mode {
            MarkupMode::Forester => {
                self.out.push_str("\\<");
                self.out.push_str(tag);
                self.out.push('>');
                self.write_attrs(attrs);
                self.out.push('{');
                self.out.push_str(&sanitize_atom(text));
                self.out.push_str("}\n");
            }
            MarkupMode::SvgXml => {
                self.out.push('<');
                self.out.push_str(tag);
                self.write_attrs(attrs);
                self.out.push('>');
                self.out.push_str(&escape_xml(text));
                self.out.push_str("</");
                self.out.push_str(tag);
                self.out.push_str(">\n");
            }
        }
    }

    pub fn elem<F: FnOnce(&mut Self)>(&mut self, tag: &str, attrs: &[(&str, &str)], f: F) {
        match self.mode {
            MarkupMode::Forester => {
                self.out.push_str("\\<");
                self.out.push_str(tag);
                self.out.push('>');
                self.write_attrs(attrs);
                self.out.push_str("{\n");
                f(self);
                self.out.push_str("}\n");
            }
            MarkupMode::SvgXml => {
                self.out.push('<');
                self.out.push_str(tag);
                self.write_attrs(attrs);
                self.out.push_str(">\n");
                f(self);
                self.out.push_str("</");
                self.out.push_str(tag);
                self.out.push_str(">\n");
            }
        }
    }
}

pub fn sanitize_atom(s: &str) -> String {
    s.chars()
        .filter(|c| {
            !matches!(
                c,
                '#' | '%' | '\\' | '{' | '}' | '[' | ']' | '<' | '>' | '&' | '"'
            )
        })
        .collect()
}

pub fn escape_xml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

pub fn fmt_commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[derive(Clone, Debug)]
pub struct PlotSeries {
    pub slug: String,
    pub label: String,
    pub summary_label: String,
    pub short_full: String,
    pub short_tail: String,
    pub color_var: &'static str,
    pub dash: Option<&'static str>,
    pub mean: Vec<f64>,
    pub sem: Vec<f64>,
    pub seed_losses: Vec<f64>,
    pub seed_diffs: Vec<f64>,
    pub final_mean: f64,
    pub final_sem: f64,
    pub diff_mean: f64,
    pub diff_sem: f64,
    pub diff_ci95: f64,
    pub p_holm: f64,
}

#[derive(Clone, Debug)]
pub struct StudyPlotModel {
    pub exp: String,
    pub horizon: usize,
    pub reps: Vec<u32>,
    pub n_seeds: usize,
    pub hero: PlotSeries,
    pub collapsed_base_count: usize,
    pub collapse_note: Option<String>,
    pub non_baseline: Vec<PlotSeries>,
    pub ppo_ref: Option<f64>,
}

const ENN_COLORS: [&str; 4] = [
    "var(--bench-c-enn1)",
    "var(--bench-c-enn2)",
    "var(--bench-c-enn3)",
    "var(--bench-c-enn4)",
];

const ENN_DASHES: [Option<&str>; 4] = [None, Some("6,2.5"), Some("3.2,2.2"), Some("1.4,2.0")];

fn sample_stats(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (idx, &val) in xs.iter().enumerate() {
        let delta = val - mean;
        mean += delta / ((idx + 1) as f64);
        m2 += delta * (val - mean);
    }
    if n <= 1 {
        return (mean, 0.0);
    }
    let sem = (m2 / (((n - 1) * n) as f64)).sqrt();
    (mean, sem)
}

pub fn t_crit(n: usize) -> f64 {
    const T_TABLE: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    if n <= 1 {
        0.0
    } else if n - 1 <= T_TABLE.len() {
        T_TABLE[n - 2]
    } else {
        1.96
    }
}

fn title_word(w: &str) -> String {
    let mut c = w.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn enn_labels(metric: &str) -> (String, String, String, String) {
    let cap = title_word(metric);
    let tail = match metric {
        "wasserstein" => "Wass".to_string(),
        "euclidean" => "Eucl".to_string(),
        "hyperbolic" => "Hyp".to_string(),
        "conformal" => "Conf".to_string(),
        _ => cap.chars().take(4).collect(),
    };
    let lbl = format!("ENN · {cap}");
    let sum = format!("{lbl} (R=1)");
    (lbl, sum, "ENN (R=1)".to_string(), tail)
}

fn arm_labels(slug: &str, is_hero_group: bool) -> (String, String, String, String) {
    if is_hero_group && (slug == "turbo-zero" || slug.starts_with("turbo-zero_")) {
        let lbl = "TuRBO-0 (R=10)".to_string();
        return (lbl.clone(), lbl, "R=10".to_string(), "R=10".to_string());
    }
    if let Some(rest) = slug.strip_prefix("turbo-enn_") {
        return enn_labels(rest);
    }
    if let Some(rest) = slug.strip_prefix("turbo-zero_") {
        let cap = title_word(rest.split('-').next().unwrap_or(rest));
        let lbl = format!("TuRBO-0 · {cap}");
        return (lbl.clone(), lbl, cap.clone(), cap);
    }
    let clean = sanitize_atom(&slug.replace(['_', ':'], " "));
    let short: String = clean.chars().take(8).collect();
    (clean.clone(), clean, short.clone(), short)
}

fn compute_curves(arm: &ArmInput, reps: &[u32], horizon: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut mean = Vec::with_capacity(horizon);
    let mut sem = Vec::with_capacity(horizon);
    for t in 0..horizon {
        let vals: Vec<f64> = reps
            .iter()
            .filter_map(|&r| arm.traces.iter().find(|tr| tr.rep == r))
            .map(|tr| tr.loss_curve[t])
            .collect();
        let (m, s) = sample_stats(&vals);
        mean.push(m);
        sem.push(s);
    }
    let seed_losses = reps
        .iter()
        .filter_map(|&r| arm.traces.iter().find(|tr| tr.rep == r))
        .map(|tr| tr.loss_curve[horizon - 1])
        .collect();
    (mean, sem, seed_losses)
}

fn trajectories_identical(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-9)
}

fn build_series(
    slug: String,
    labels: (String, String, String, String),
    curves: (Vec<f64>, Vec<f64>, Vec<f64>),
    base_losses: &[f64],
    p_holm: f64,
    style: (&'static str, Option<&'static str>),
) -> PlotSeries {
    let (mean, sem, seed_losses) = curves;
    let final_mean = *mean.last().unwrap_or(&0.0);
    let final_sem = *sem.last().unwrap_or(&0.0);
    let seed_diffs: Vec<f64> = seed_losses
        .iter()
        .zip(base_losses)
        .map(|(&l, &b)| l - b)
        .collect();
    let (diff_mean, diff_sem) = sample_stats(&seed_diffs);
    let diff_ci95 = t_crit(seed_diffs.len()) * diff_sem;
    PlotSeries {
        slug,
        label: labels.0,
        summary_label: labels.1,
        short_full: labels.2,
        short_tail: labels.3,
        color_var: style.0,
        dash: style.1,
        mean,
        sem,
        seed_losses,
        seed_diffs,
        final_mean,
        final_sem,
        diff_mean,
        diff_sem,
        diff_ci95,
        p_holm,
    }
}

type RawChallenger = (String, (Vec<f64>, Vec<f64>, Vec<f64>), f64);

fn strip_env(env_id: &str, cid: &str) -> String {
    let prefix = format!("{env_id}__");
    cid.strip_prefix(&prefix).unwrap_or(cid).to_string()
}

fn collect_challengers(
    report: &PairedReport,
    arm_inputs: &[ArmInput],
    base_idx: usize,
    base_mean: &[f64],
) -> (usize, Vec<RawChallenger>) {
    let base_arm = &arm_inputs[base_idx];
    let base_slug = strip_env(&report.env_id, &base_arm.cell_id);
    let horizon = report.horizon as usize;
    let mut collapsed = 1usize;
    let mut challengers = Vec::new();
    for (idx, arm) in arm_inputs.iter().enumerate() {
        if idx == base_idx {
            continue;
        }
        let slug = strip_env(&report.env_id, &arm.cell_id);
        let curves = compute_curves(arm, &report.common_reps, horizon);
        let same_family = arm.arm_id == base_arm.arm_id
            || (base_slug.starts_with("turbo-zero") && slug.starts_with("turbo-zero"));
        if same_family && trajectories_identical(&curves.0, base_mean) {
            collapsed += 1;
        } else {
            let ph = report
                .arms
                .iter()
                .find(|a| a.cell_id == arm.cell_id)
                .map(|a| a.p_holm)
                .unwrap_or(1.0);
            challengers.push((slug, curves, ph));
        }
    }
    challengers.sort_by(|a, b| {
        let la = a.1.0.last().copied().unwrap_or(0.0);
        let lb = b.1.0.last().copied().unwrap_or(0.0);
        la.partial_cmp(&lb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    (collapsed, challengers)
}

pub fn plot_model(
    report: &PairedReport,
    arm_inputs: &[ArmInput],
    exp: &str,
) -> Result<StudyPlotModel, String> {
    let horizon = report.horizon as usize;
    if horizon == 0 || arm_inputs.is_empty() || report.common_reps.is_empty() {
        return Err("cannot plot empty incremental report".into());
    }
    let base_idx = arm_inputs
        .iter()
        .position(|a| a.cell_id == report.baseline_cell_id)
        .or_else(|| arm_inputs.iter().position(|a| a.baseline))
        .unwrap_or(0);
    let base_slug = strip_env(&report.env_id, &arm_inputs[base_idx].cell_id);
    let base_curves = compute_curves(&arm_inputs[base_idx], &report.common_reps, horizon);
    let (collapsed_count, raw_non_base) =
        collect_challengers(report, arm_inputs, base_idx, &base_curves.0);

    let hero_is_group = collapsed_count > 1 || base_slug.starts_with("turbo-zero");
    let hero_labels = arm_labels(&base_slug, hero_is_group);
    let base_losses = base_curves.2.clone();
    let hero = build_series(
        base_slug,
        hero_labels,
        base_curves,
        &base_losses,
        1.0,
        ("var(--bench-c-hero)", None),
    );

    let non_baseline = raw_non_base
        .into_iter()
        .enumerate()
        .map(|(k, (slug, curves, ph))| {
            let labels = arm_labels(&slug, false);
            let style = (ENN_COLORS[k % 4], ENN_DASHES[k % 4]);
            build_series(slug, labels, curves, &base_losses, ph, style)
        })
        .collect();

    let collapse_note = match collapsed_count {
        4 => Some("4 topologies ≡ 0".to_string()),
        c if c > 1 => Some(format!("{c} variants ≡ 0")),
        _ => None,
    };

    Ok(StudyPlotModel {
        exp: sanitize_atom(exp),
        horizon,
        n_seeds: report.common_reps.len(),
        reps: report.common_reps.clone(),
        hero,
        collapsed_base_count: collapsed_count,
        collapse_note,
        non_baseline,
        ppo_ref: None,
    })
}

pub fn scrub_rounds(horizon: usize) -> Vec<usize> {
    if horizon <= 48 {
        return (1..=horizon).collect();
    }
    let anchors = [1, 10, 25, 50, 100, 250, 500, 1000, 3000, 10000];
    let mut rounds: Vec<usize> = anchors
        .into_iter()
        .filter(|&r| r <= horizon)
        .chain(std::iter::once(horizon))
        .collect();
    for k in 0..42 {
        let frac = (k as f64) / 41.0;
        let t = (horizon as f64).powf(frac).round() as usize;
        rounds.push(t.clamp(1, horizon));
    }
    rounds.sort_unstable();
    rounds.dedup();
    rounds
}

pub fn curve_rounds(start_round: usize, horizon: usize) -> Vec<usize> {
    let lo = start_round.max(1).min(horizon);
    if horizon.saturating_sub(lo) <= 160 {
        return (lo..=horizon).collect();
    }
    let mut pts: Vec<usize> = (lo..=lo.saturating_add(24).min(horizon)).collect();
    let l0 = (lo as f64).ln();
    let l1 = (horizon as f64).ln();
    for k in 0..144 {
        let t = (l0 + (l1 - l0) * ((k as f64) / 143.0)).exp().round() as usize;
        pts.push(t.clamp(lo, horizon));
    }
    pts.push(horizon);
    pts.sort_unstable();
    pts.dedup();
    pts
}

pub fn repel_labels(y_true: &[f64], y_min: f64, y_max: f64, min_gap: f64) -> Vec<f64> {
    let m = y_true.len();
    if m == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        y_true[a]
            .partial_cmp(&y_true[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut ys: Vec<f64> = order
        .iter()
        .map(|&i| y_true[i].clamp(y_min, y_max))
        .collect();
    for _ in 0..32 {
        for i in 1..m {
            if ys[i] < ys[i - 1] + min_gap {
                let mid = 0.5 * (ys[i - 1] + ys[i]);
                ys[i - 1] = mid - 0.5 * min_gap;
                ys[i] = mid + 0.5 * min_gap;
            }
        }
        ys[0] = ys[0].max(y_min);
        for i in 1..m {
            ys[i] = ys[i].max(ys[i - 1] + min_gap);
        }
        ys[m - 1] = ys[m - 1].min(y_max);
        for i in (0..m - 1).rev() {
            ys[i] = ys[i].min(ys[i + 1] - min_gap);
        }
        ys[0] = ys[0].max(y_min);
        for i in 1..m {
            ys[i] = ys[i].max(ys[i - 1] + min_gap);
        }
    }
    let mut out = vec![0.0; m];
    for (rank, &orig) in order.iter().enumerate() {
        out[orig] = ys[rank];
    }
    out
}
