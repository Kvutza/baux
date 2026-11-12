//! Pane A anytime trajectory SVG renderer (`full` log-log and `tail` zoomed
//! log-linear views) with collision-free right-margin direct labels.

use super::series::{
    MarkupBuf, PlotSeries, StudyPlotModel, curve_rounds, fmt_commas, repel_labels,
};

const X_LEFT: f64 = 48.0;
const X_RIGHT: f64 = 326.0;
const Y_TOP: f64 = 18.0;
const Y_BOT: f64 = 228.0;

struct RightLabel {
    y_true: f64,
    color: &'static str,
    text: String,
    bold: bool,
}

struct AxisSpec {
    log_min: f64,
    log_max: f64,
    y_min: f64,
    y_max: f64,
    y_log: bool,
}

impl AxisSpec {
    fn x_of(&self, t: usize) -> f64 {
        let span = (self.log_max - self.log_min).max(1e-6);
        let u = (((t.max(1) as f64).log10() - self.log_min) / span).clamp(0.0, 1.0);
        X_LEFT + (X_RIGHT - X_LEFT) * u
    }

    fn y_of(&self, v: f64) -> f64 {
        if self.y_log {
            let lo = self.y_min.max(1e-6);
            let hi = self.y_max.max(lo * 1.01);
            let vc = v.clamp(lo, hi);
            let u = ((vc.log10() - lo.log10()) / (hi.log10() - lo.log10())).clamp(0.0, 1.0);
            Y_BOT - (Y_BOT - Y_TOP) * u
        } else {
            let span = (self.y_max - self.y_min).max(1e-6);
            let u = ((v.clamp(self.y_min, self.y_max) - self.y_min) / span).clamp(0.0, 1.0);
            Y_BOT - (Y_BOT - Y_TOP) * u
        }
    }
}

fn round_tick(t: usize) -> String {
    if t >= 1000 && t.is_multiple_of(1000) {
        format!("{}k", t / 1000)
    } else {
        t.to_string()
    }
}

fn fmt_metric(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.1}")
    } else if v >= 10.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

fn fmt_tick(v: f64, step: f64) -> String {
    if step >= 1.0 && (v - v.round()).abs() < 1e-6 {
        format!("{:.0}", v.round())
    } else if step >= 0.1 {
        format!("{v:.1}")
    } else if step >= 0.01 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

pub fn nice_step(raw: f64) -> f64 {
    let safe = raw.abs().max(1e-6);
    let exp = safe.log10().floor();
    let pow = 10.0f64.powf(exp);
    let frac = safe / pow;
    let mult = if frac <= 1.5 {
        1.0
    } else if frac <= 3.2 {
        2.0
    } else if frac <= 7.0 {
        5.0
    } else {
        10.0
    };
    mult * pow
}

fn linear_ticks(y_min: f64, y_max: f64, target_cnt: usize) -> Vec<(f64, String)> {
    let step = nice_step((y_max - y_min) / (target_cnt.max(2) as f64));
    let k_lo = (y_min / step).ceil() as i64;
    let k_hi = (y_max / step).floor() as i64;
    (k_lo..=k_hi)
        .take(12)
        .map(|k| {
            let v = (k as f64) * step;
            (v, fmt_tick(v, step))
        })
        .collect()
}

fn range_ticks(start_round: usize, horizon: usize) -> Vec<(usize, String)> {
    let cands: &[usize] = if start_round == 1 && horizon >= 1000 {
        &[1, 10, 100, 1000, 10000, 100000]
    } else {
        &[
            1, 2, 5, 10, 20, 50, 100, 300, 1000, 3000, 10000, 30000, 100000,
        ]
    };
    let mut ts: Vec<usize> = cands
        .iter()
        .copied()
        .filter(|&t| t >= start_round && t <= horizon)
        .collect();
    if ts.first().copied() != Some(start_round) {
        ts.insert(0, start_round);
    }
    if ts.last().copied() != Some(horizon) {
        ts.push(horizon);
    }
    ts.into_iter().map(|t| (t, round_tick(t))).collect()
}

fn full_domain(model: &StudyPlotModel) -> (f64, f64, Vec<(f64, String)>) {
    let mut min_v = model
        .hero
        .mean
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let mut max_v = model.hero.mean.iter().copied().fold(0.0f64, f64::max);
    if let Some(best) = model.non_baseline.first() {
        min_v = min_v.min(best.mean.iter().copied().fold(f64::INFINITY, f64::min));
        max_v = max_v.max(best.mean.iter().copied().fold(0.0f64, f64::max));
    }
    if let Some(ppo) = model.ppo_ref {
        min_v = min_v.min(ppo);
        max_v = max_v.max(ppo);
    }
    if model.exp == "tokamak-tokagym" && min_v >= 1.15 && max_v <= 80.0 {
        let ticks = vec![
            (1.3, "1.3".into()),
            (2.0, "2".into()),
            (5.0, "5".into()),
            (10.0, "10".into()),
            (20.0, "20".into()),
            (50.0, "50".into()),
        ];
        return (1.15, 80.0, ticks);
    }
    let y_min = (0.88 * min_v.max(1e-3)).max(1e-4);
    let y_max = (1.15 * max_v.max(y_min * 1.1)).max(y_min * 1.15);
    let log_cands = [
        0.1, 0.2, 0.5, 1.0, 1.3, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
    ];
    let ticks: Vec<(f64, String)> = log_cands
        .into_iter()
        .filter(|&v| v >= y_min * 1.02 && v <= y_max * 0.98)
        .map(|v| (v, fmt_tick(v, if v < 2.0 { 0.1 } else { 1.0 })))
        .collect();
    if ticks.len() >= 3 {
        (y_min, y_max, ticks)
    } else {
        (y_min, y_max, linear_ticks(y_min, y_max, 5))
    }
}

fn tail_domain(model: &StudyPlotModel, start_round: usize) -> (f64, f64, Vec<(f64, String)>) {
    let idx0 = start_round.saturating_sub(1).min(model.horizon - 1);
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for t in idx0..model.horizon {
        min_v = min_v.min((model.hero.mean[t] - model.hero.sem[t]).max(0.0));
        max_v = max_v.max(model.hero.mean[t]);
        for arm in model.non_baseline.iter().take(4) {
            min_v = min_v.min(arm.mean[t]);
            max_v = max_v.max(arm.mean[t]);
        }
    }
    let span = (max_v - min_v).max(0.04);
    let y_min = (min_v - 0.08 * span).max(0.0);
    let y_max = max_v + 0.08 * span;
    (y_min, y_max, linear_ticks(y_min, y_max, 6))
}

fn emit_axes(
    buf: &mut MarkupBuf,
    ax: &AxisSpec,
    x_ticks: &[(usize, String)],
    y_ticks: &[(f64, String)],
) {
    for &(t, ref lbl) in x_ticks {
        let x = format!("{:.2}", ax.x_of(t));
        buf.leaf(
            "line",
            &[
                ("x1", &x),
                ("y1", "18"),
                ("x2", &x),
                ("y2", "228"),
                ("stroke", "var(--bench-grid)"),
                ("stroke-width", "0.7"),
            ],
        );
        buf.leaf(
            "line",
            &[
                ("x1", &x),
                ("y1", "228"),
                ("x2", &x),
                ("y2", "232"),
                ("stroke", "var(--bench-axis)"),
                ("stroke-width", "0.9"),
            ],
        );
        buf.text_elem(
            "text",
            &[
                ("x", &x),
                ("y", "243"),
                ("text-anchor", "middle"),
                ("font-family", "var(--font-mono)"),
                ("font-size", "9.5"),
                ("fill", "var(--fg-muted)"),
            ],
            lbl,
        );
    }
    for &(v, ref lbl) in y_ticks {
        let y = format!("{:.2}", ax.y_of(v));
        buf.leaf(
            "line",
            &[
                ("x1", "48"),
                ("y1", &y),
                ("x2", "326"),
                ("y2", &y),
                ("stroke", "var(--bench-grid)"),
                ("stroke-width", "0.7"),
            ],
        );
        buf.leaf(
            "line",
            &[
                ("x1", "44"),
                ("y1", &y),
                ("x2", "48"),
                ("y2", &y),
                ("stroke", "var(--bench-axis)"),
                ("stroke-width", "0.9"),
            ],
        );
        buf.text_elem(
            "text",
            &[
                ("x", "41"),
                ("y", &y),
                ("text-anchor", "end"),
                ("dominant-baseline", "central"),
                ("font-family", "var(--font-mono)"),
                ("font-size", "9.5"),
                ("fill", "var(--fg-muted)"),
            ],
            lbl,
        );
    }
    buf.leaf(
        "line",
        &[
            ("x1", "48"),
            ("y1", "18"),
            ("x2", "48"),
            ("y2", "228"),
            ("stroke", "var(--bench-axis)"),
            ("stroke-width", "1"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", "48"),
            ("y1", "228"),
            ("x2", "326"),
            ("y2", "228"),
            ("stroke", "var(--bench-axis)"),
            ("stroke-width", "1"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "187"),
            ("y", "262"),
            ("text-anchor", "middle"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "11.5"),
            ("fill", "var(--fg-secondary)"),
        ],
        "Evaluation round t",
    );
    buf.text_elem(
        "text",
        &[
            ("x", "13"),
            ("y", "123"),
            ("transform", "rotate(-90 13 123)"),
            ("text-anchor", "middle"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "11.5"),
            ("fill", "var(--fg-secondary)"),
        ],
        "Tracking error (lower is better)",
    );
}

fn emit_band(buf: &mut MarkupBuf, ax: &AxisSpec, s: &PlotSeries, rounds: &[usize]) {
    let mut pts = String::with_capacity(rounds.len() * 28);
    for &t in rounds {
        let i = t - 1;
        let p = format!("{:.2},{:.2} ", ax.x_of(t), ax.y_of(s.mean[i] + s.sem[i]));
        pts.push_str(&p);
    }
    for &t in rounds.iter().rev() {
        let i = t - 1;
        let p = format!("{:.2},{:.2} ", ax.x_of(t), ax.y_of(s.mean[i] - s.sem[i]));
        pts.push_str(&p);
    }
    buf.leaf(
        "polygon",
        &[
            ("points", pts.trim()),
            ("fill", s.color_var),
            ("fill-opacity", "0.14"),
            ("stroke", "none"),
        ],
    );
}

fn emit_curve(
    buf: &mut MarkupBuf,
    ax: &AxisSpec,
    s: &PlotSeries,
    rounds: &[usize],
    width: &str,
    use_dash: bool,
) {
    let mut pts = String::with_capacity(rounds.len() * 14);
    for &t in rounds {
        let i = t - 1;
        pts.push_str(&format!("{:.2},{:.2} ", ax.x_of(t), ax.y_of(s.mean[i])));
    }
    let mut attrs = vec![
        ("points", pts.trim()),
        ("fill", "none"),
        ("stroke", s.color_var),
        ("stroke-width", width),
        ("stroke-linejoin", "round"),
        ("stroke-linecap", "round"),
    ];
    if use_dash && let Some(d) = s.dash {
        attrs.push(("stroke-dasharray", d));
    }
    buf.leaf("polyline", &attrs);
    let cy = format!("{:.2}", ax.y_of(s.final_mean));
    buf.leaf(
        "circle",
        &[
            ("cx", "326"),
            ("cy", &cy),
            ("r", "2.3"),
            ("fill", s.color_var),
        ],
    );
}

fn emit_labels(buf: &mut MarkupBuf, labels: &[RightLabel], min_gap: f64) {
    let y_trues: Vec<f64> = labels.iter().map(|l| l.y_true).collect();
    let y_rep = repel_labels(&y_trues, Y_TOP + 4.0, Y_BOT - 3.0, min_gap);
    for (item, &yl) in labels.iter().zip(&y_rep) {
        let pts = format!(
            "326.00,{:.2} 329.50,{:.2} 333.00,{:.2}",
            item.y_true, yl, yl
        );
        buf.leaf(
            "polyline",
            &[
                ("points", &pts),
                ("fill", "none"),
                ("stroke", item.color),
                ("stroke-width", "0.8"),
                ("opacity", "0.65"),
            ],
        );
        let y_s = format!("{yl:.2}");
        let fw = if item.bold { "700" } else { "500" };
        buf.text_elem(
            "text",
            &[
                ("x", "335"),
                ("y", &y_s),
                ("dominant-baseline", "central"),
                ("font-family", "var(--font-mono)"),
                ("font-size", "9.2"),
                ("font-weight", fw),
                ("fill", item.color),
            ],
            &item.text,
        );
    }
}

pub fn render_full(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let log_max = (model.horizon.max(2) as f64).log10();
    let (y_min, y_max, y_ticks) = full_domain(model);
    let ax = AxisSpec {
        log_min: 0.0,
        log_max,
        y_min,
        y_max,
        y_log: true,
    };
    let x_ticks = range_ticks(1, model.horizon);
    emit_axes(buf, &ax, &x_ticks, &y_ticks);

    let rounds = curve_rounds(1, model.horizon);
    let mut r_labels = Vec::new();
    if let Some(ppo) = model.ppo_ref {
        let yp = ax.y_of(ppo);
        let ys = format!("{yp:.2}");
        buf.leaf(
            "line",
            &[
                ("x1", "48"),
                ("y1", &ys),
                ("x2", "326"),
                ("y2", &ys),
                ("stroke", "var(--bench-c-ppo)"),
                ("stroke-width", "1.35"),
                ("stroke-dasharray", "4,3"),
            ],
        );
        r_labels.push(RightLabel {
            y_true: yp,
            color: "var(--bench-c-ppo)",
            text: format!("PPO {ppo:.2}"),
            bold: false,
        });
    }
    if let Some(best) = model.non_baseline.first() {
        emit_band(buf, &ax, best, &rounds);
        emit_curve(buf, &ax, best, &rounds, "1.75", false);
        r_labels.push(RightLabel {
            y_true: ax.y_of(best.final_mean),
            color: best.color_var,
            text: format!("{} {}", best.short_full, fmt_metric(best.final_mean)),
            bold: false,
        });
    }
    emit_band(buf, &ax, &model.hero, &rounds);
    emit_curve(buf, &ax, &model.hero, &rounds, "2.1", false);
    r_labels.push(RightLabel {
        y_true: ax.y_of(model.hero.final_mean),
        color: model.hero.color_var,
        text: format!(
            "{} {}",
            model.hero.short_full,
            fmt_metric(model.hero.final_mean)
        ),
        bold: true,
    });
    emit_labels(buf, &r_labels, 13.0);
    buf.leaf(
        "line",
        &[
            ("class", "baux-scrub-line"),
            ("x1", "326"),
            ("y1", "18"),
            ("x2", "326"),
            ("y2", "228"),
            ("stroke", "var(--fg-muted)"),
            ("stroke-width", "1"),
            ("stroke-dasharray", "3,3"),
            ("opacity", "0"),
        ],
    );
}

pub fn tail_start(horizon: usize) -> usize {
    if horizon > 100 {
        100
    } else {
        (horizon / 2).max(1)
    }
}

pub fn render_tail(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let t0 = tail_start(model.horizon);
    let log_min = (t0 as f64).log10();
    let log_max = (model.horizon.max(t0 + 1) as f64).log10();
    let (y_min, y_max, y_ticks) = tail_domain(model, t0);
    let ax = AxisSpec {
        log_min,
        log_max,
        y_min,
        y_max,
        y_log: false,
    };
    let x_ticks = range_ticks(t0, model.horizon);
    emit_axes(buf, &ax, &x_ticks, &y_ticks);

    let rounds = curve_rounds(t0, model.horizon);
    if let Some(best) = model.non_baseline.first() {
        emit_band(buf, &ax, best, &rounds);
    }
    emit_band(buf, &ax, &model.hero, &rounds);

    let mut r_labels = Vec::new();
    for (idx, arm) in model.non_baseline.iter().take(4).enumerate() {
        let w = if idx == 0 { "1.8" } else { "1.5" };
        emit_curve(buf, &ax, arm, &rounds, w, true);
        r_labels.push(RightLabel {
            y_true: ax.y_of(arm.final_mean),
            color: arm.color_var,
            text: format!("{} {}", arm.short_tail, fmt_metric(arm.final_mean)),
            bold: false,
        });
    }
    emit_curve(buf, &ax, &model.hero, &rounds, "2.1", false);
    r_labels.push(RightLabel {
        y_true: ax.y_of(model.hero.final_mean),
        color: model.hero.color_var,
        text: format!(
            "{} {}",
            model.hero.short_tail,
            fmt_metric(model.hero.final_mean)
        ),
        bold: true,
    });
    emit_labels(buf, &r_labels, 12.5);
    buf.leaf(
        "line",
        &[
            ("class", "baux-scrub-line"),
            ("x1", "326"),
            ("y1", "18"),
            ("x2", "326"),
            ("y2", "228"),
            ("stroke", "var(--fg-muted)"),
            ("stroke-width", "1"),
            ("stroke-dasharray", "3,3"),
            ("opacity", "0"),
        ],
    );
}

fn fmt_log(v: f64) -> String {
    if (v - v.round()).abs() < 1e-6 {
        format!("{:.0}", v.round())
    } else {
        format!("{v:.3}")
    }
}

pub fn emit_traj(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let h_str = fmt_commas(model.horizon);
    let log_max_full = fmt_log((model.horizon.max(2) as f64).log10());
    let aria_full = format!("Anytime tracking error from round 1 to {h_str}");
    buf.elem(
        "svg",
        &[
            ("class", "baux-bench-svg"),
            ("data-view", "full"),
            ("data-x-left", "48"),
            ("data-x-right", "326"),
            ("data-log-min", "0"),
            ("data-log-max", &log_max_full),
            ("viewBox", "0 0 432 272"),
            ("role", "img"),
            ("aria-label", &aria_full),
        ],
        |b| render_full(b, model),
    );

    let t0 = tail_start(model.horizon);
    let log_min_tail = fmt_log((t0 as f64).log10());
    let log_max_tail = fmt_log((model.horizon.max(t0 + 1) as f64).log10());
    let aria_tail = format!("Zoomed tail tracking error from round {t0} to {h_str}");
    buf.elem(
        "svg",
        &[
            ("class", "baux-bench-svg"),
            ("data-view", "tail"),
            ("hidden", "hidden"),
            ("data-x-left", "48"),
            ("data-x-right", "326"),
            ("data-log-min", &log_min_tail),
            ("data-log-max", &log_max_tail),
            ("viewBox", "0 0 432 272"),
            ("role", "img"),
            ("aria-label", &aria_tail),
        ],
        |b| render_tail(b, model),
    );
}
