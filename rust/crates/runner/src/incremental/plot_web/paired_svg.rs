//! Pane B paired common-random-number seed difference forest plot
//! (`data-view="paired"`).

use super::series::{MarkupBuf, PlotSeries, StudyPlotModel, fmt_commas};
use super::traj_svg::nice_step;

const X_LEFT: f64 = 120.0;
const X_RIGHT: f64 = 312.0;
const Y_TOP: f64 = 24.0;
const Y_BOT: f64 = 228.0;

struct DeltaScale {
    d_min: f64,
    d_max: f64,
}

impl DeltaScale {
    fn x_of(&self, d: f64) -> f64 {
        let span = (self.d_max - self.d_min).max(1e-6);
        let u = ((d.clamp(self.d_min, self.d_max) - self.d_min) / span).clamp(0.0, 1.0);
        X_LEFT + (X_RIGHT - X_LEFT) * u
    }
}

fn signed_tick(v: f64, step: f64) -> String {
    if v.abs() < 1e-9 {
        return "0".to_string();
    }
    let sign = if v > 0.0 { "+" } else { "-" };
    let av = v.abs();
    if step >= 1.0 && (av - av.round()).abs() < 1e-6 {
        format!("{sign}{:.0}", av.round())
    } else if step >= 0.1 {
        format!("{sign}{av:.1}")
    } else if step >= 0.01 {
        format!("{sign}{av:.2}")
    } else {
        format!("{sign}{av:.3}")
    }
}

fn delta_scale(model: &StudyPlotModel) -> (DeltaScale, Vec<(f64, String)>) {
    let mut lo = 0.0f64;
    let mut hi = 0.0f64;
    for arm in &model.non_baseline {
        lo = lo.min(arm.diff_mean - arm.diff_ci95);
        hi = hi.max(arm.diff_mean + arm.diff_ci95);
        for &d in &arm.seed_diffs {
            lo = lo.min(d);
            hi = hi.max(d);
        }
    }
    let raw_span = (hi - lo).max(0.08);
    let mut d_min = (lo - 0.09 * raw_span).min(-0.04);
    let mut d_max = (hi + 0.09 * raw_span).max(0.04);
    d_min = d_min.min(-0.45 * d_max);
    d_max = d_max.max(-0.55 * d_min);

    let step = nice_step((d_max - d_min) / 4.2);
    let k_lo = (d_min / step).ceil() as i64;
    let k_hi = (d_max / step).floor() as i64;
    let ticks = (k_lo..=k_hi)
        .take(10)
        .map(|k| {
            let v = (k as f64) * step;
            (v, signed_tick(v, step))
        })
        .collect();
    (DeltaScale { d_min, d_max }, ticks)
}

fn emit_frame(
    buf: &mut MarkupBuf,
    sc: &DeltaScale,
    ticks: &[(f64, String)],
    hero_short: &str,
    zero_bot_y: f64,
) {
    let x0 = sc.x_of(0.0);
    let x_left_cue = format!("{:.2}", (x0 - 5.0).max(X_LEFT + 2.0));
    let x_right_cue = format!("{:.2}", (x0 + 5.0).min(X_RIGHT - 2.0));
    let left_cue = format!("← beats {hero_short}");
    buf.text_elem(
        "text",
        &[
            ("x", &x_left_cue),
            ("y", "15"),
            ("text-anchor", "end"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "9.2"),
            ("fill", "var(--fg-muted)"),
        ],
        &left_cue,
    );
    buf.text_elem(
        "text",
        &[
            ("x", &x_right_cue),
            ("y", "15"),
            ("text-anchor", "start"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "9.2"),
            ("fill", "var(--fg-muted)"),
        ],
        "higher error (+Δ) →",
    );
    buf.text_elem(
        "text",
        &[
            ("x", "368"),
            ("y", "15"),
            ("text-anchor", "end"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "8.8"),
            ("fill", "var(--fg-muted)"),
        ],
        "Δ mean",
    );

    let grid_bot = format!("{zero_bot_y:.2}");
    for &(v, ref lbl) in ticks {
        let x = format!("{:.2}", sc.x_of(v));
        if v.abs() > 1e-9 {
            buf.leaf(
                "line",
                &[
                    ("x1", &x),
                    ("y1", "24"),
                    ("x2", &x),
                    ("y2", &grid_bot),
                    ("stroke", "var(--bench-grid)"),
                    ("stroke-width", "0.65"),
                ],
            );
        }
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

    let x0_s = format!("{x0:.2}");
    buf.leaf(
        "line",
        &[
            ("x1", &x0_s),
            ("y1", "22"),
            ("x2", &x0_s),
            ("y2", &grid_bot),
            ("stroke", "var(--bench-zero)"),
            ("stroke-width", "1.1"),
            ("stroke-dasharray", "3,3"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", "120"),
            ("y1", "228"),
            ("x2", "312"),
            ("y2", "228"),
            ("stroke", "var(--bench-axis)"),
            ("stroke-width", "1"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "216"),
            ("y", "262"),
            ("text-anchor", "middle"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "11.5"),
            ("fill", "var(--fg-secondary)"),
        ],
        "Paired error difference Δ vs. baseline",
    );
}

fn emit_anchor(buf: &mut MarkupBuf, model: &StudyPlotModel, x_zero: f64, y_row: f64) {
    let ys = format!("{y_row:.2}");
    buf.leaf(
        "line",
        &[
            ("x1", "120"),
            ("y1", &ys),
            ("x2", "312"),
            ("y2", &ys),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "0.6"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "114"),
            ("y", &ys),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "10.3"),
            ("font-weight", "700"),
            ("fill", "var(--fg-primary)"),
        ],
        &model.hero.label,
    );
    let diamond = format!(
        "{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}",
        x_zero,
        y_row - 4.5,
        x_zero + 4.5,
        y_row,
        x_zero,
        y_row + 4.5,
        x_zero - 4.5,
        y_row
    );
    buf.leaf(
        "polygon",
        &[("points", &diamond), ("fill", "var(--bench-c-hero)")],
    );
    if let Some(ref note) = model.collapse_note {
        let xn = format!("{:.2}", x_zero + 8.0);
        buf.text_elem(
            "text",
            &[
                ("x", &xn),
                ("y", &ys),
                ("dominant-baseline", "central"),
                ("font-family", "var(--font-serif)"),
                ("font-size", "9"),
                ("fill", "var(--fg-muted)"),
            ],
            note,
        );
    }
    buf.text_elem(
        "text",
        &[
            ("x", "368"),
            ("y", &ys),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "9.8"),
            ("fill", "var(--fg-secondary)"),
        ],
        "0.000",
    );
}

fn emit_arm(
    buf: &mut MarkupBuf,
    model: &StudyPlotModel,
    arm: &PlotSeries,
    sc: &DeltaScale,
    y_row: f64,
) {
    let ys = format!("{y_row:.2}");
    buf.leaf(
        "line",
        &[
            ("x1", "120"),
            ("y1", &ys),
            ("x2", "312"),
            ("y2", &ys),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "0.6"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "114"),
            ("y", &ys),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "10.2"),
            ("fill", "var(--fg-primary)"),
        ],
        &arm.label,
    );

    let n = arm.seed_diffs.len();
    let mid_s = ((n.saturating_sub(1)) as f64) * 0.5;
    let j_step = if n <= 12 { 1.15 } else { 11.0 / (n as f64) };
    for (s, &d_s) in arm.seed_diffs.iter().enumerate() {
        let cx = format!("{:.2}", sc.x_of(d_s));
        let cy = format!("{:.2}", y_row + ((s as f64) - mid_s) * j_step);
        let seed_idx = s.to_string();
        let rep_num = model.reps.get(s).copied().unwrap_or(s as u32) + 1;
        let loss_s = arm.seed_losses.get(s).copied().unwrap_or(0.0);
        let base_s = model.hero.seed_losses.get(s).copied().unwrap_or(0.0);
        let summary = format!(
            "Seed {rep_num}  ·  {}: {loss_s:.3} vs {}: {base_s:.3}  (Δ = {d_s:+.3})",
            arm.label, model.hero.short_full
        );
        buf.leaf(
            "circle",
            &[
                ("class", "baux-seed-dot"),
                ("data-seed", &seed_idx),
                ("data-arm", &arm.label),
                ("data-summary", &summary),
                ("cx", &cx),
                ("cy", &cy),
                ("r", "2.9"),
                ("fill", arm.color_var),
                ("fill-opacity", "0.55"),
                ("stroke", arm.color_var),
                ("stroke-width", "0.8"),
            ],
        );
    }

    let x_lo = format!("{:.2}", sc.x_of(arm.diff_mean - arm.diff_ci95));
    let x_hi = format!("{:.2}", sc.x_of(arm.diff_mean + arm.diff_ci95));
    let y_cap_top = format!("{:.2}", y_row - 3.0);
    let y_cap_bot = format!("{:.2}", y_row + 3.0);
    buf.leaf(
        "line",
        &[
            ("x1", &x_lo),
            ("y1", &ys),
            ("x2", &x_hi),
            ("y2", &ys),
            ("stroke", arm.color_var),
            ("stroke-width", "1.8"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", &x_lo),
            ("y1", &y_cap_top),
            ("x2", &x_lo),
            ("y2", &y_cap_bot),
            ("stroke", arm.color_var),
            ("stroke-width", "1.4"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", &x_hi),
            ("y1", &y_cap_top),
            ("x2", &x_hi),
            ("y2", &y_cap_bot),
            ("stroke", arm.color_var),
            ("stroke-width", "1.4"),
        ],
    );
    let xm = format!("{:.2}", sc.x_of(arm.diff_mean));
    buf.leaf(
        "circle",
        &[
            ("cx", &xm),
            ("cy", &ys),
            ("r", "4.0"),
            ("fill", arm.color_var),
            ("stroke", "var(--bg-canvas)"),
            ("stroke-width", "1.1"),
        ],
    );

    let val_str = format!("{:+.3}", arm.diff_mean);
    buf.text_elem(
        "text",
        &[
            ("x", "368"),
            ("y", &ys),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "9.8"),
            ("font-weight", "600"),
            ("fill", "var(--fg-primary)"),
        ],
        &val_str,
    );
}

fn emit_ppo(buf: &mut MarkupBuf, model: &StudyPlotModel, ppo_val: f64, x_zero: f64) {
    buf.leaf(
        "line",
        &[
            ("x1", "16"),
            ("y1", "193"),
            ("x2", "368"),
            ("y2", "193"),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "0.9"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "114"),
            ("y", "211"),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "10.2"),
            ("font-weight", "600"),
            ("fill", "var(--bench-c-ppo)"),
        ],
        "On-Policy PPO",
    );
    let x0_s = format!("{x_zero:.2}");
    buf.leaf(
        "line",
        &[
            ("x1", &x0_s),
            ("y1", "216"),
            ("x2", "305"),
            ("y2", "216"),
            ("stroke", "var(--bench-c-ppo)"),
            ("stroke-width", "1.35"),
            ("stroke-dasharray", "4,3"),
        ],
    );
    buf.leaf(
        "polygon",
        &[
            ("points", "311,216 304.5,213.2 304.5,218.8"),
            ("fill", "var(--bench-c-ppo)"),
        ],
    );
    let delta = ppo_val - model.hero.final_mean;
    let ratio = ppo_val / model.hero.final_mean.max(1e-6);
    let callout = format!("{delta:+.2} ({ratio:.1}× higher error ▸)");
    buf.text_elem(
        "text",
        &[
            ("x", "309"),
            ("y", "204"),
            ("text-anchor", "end"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "8.7"),
            ("fill", "var(--bench-c-ppo)"),
        ],
        &callout,
    );
    let d_str = format!("{delta:+.2}");
    buf.text_elem(
        "text",
        &[
            ("x", "368"),
            ("y", "211"),
            ("text-anchor", "end"),
            ("dominant-baseline", "central"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "9.8"),
            ("font-weight", "600"),
            ("fill", "var(--bench-c-ppo)"),
        ],
        &d_str,
    );
}

pub fn render_paired(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let (sc, ticks) = delta_scale(model);
    let has_ppo = model.ppo_ref.is_some();
    let zero_bot = if has_ppo { 193.0 } else { Y_BOT };
    emit_frame(buf, &sc, &ticks, &model.hero.short_full, zero_bot);

    let bottom_row_y = if has_ppo { 174.0 } else { 204.0 };
    let n_rows = 1 + model.non_baseline.len();
    let y0 = if n_rows == 1 {
        0.5 * (Y_TOP + bottom_row_y)
    } else {
        38.0
    };
    let step = if n_rows <= 1 {
        0.0
    } else {
        ((bottom_row_y - y0) / ((n_rows - 1) as f64)).min(42.0)
    };
    let x_zero = sc.x_of(0.0);
    emit_anchor(buf, model, x_zero, y0);
    for (i, arm) in model.non_baseline.iter().enumerate() {
        let yr = y0 + ((i + 1) as f64) * step;
        emit_arm(buf, model, arm, &sc, yr);
    }
    if let Some(ppo_val) = model.ppo_ref {
        emit_ppo(buf, model, ppo_val, x_zero);
    }
}

pub fn emit_paired(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let h_str = fmt_commas(model.horizon);
    let aria = format!("Paired common-random-number seed differences at round {h_str}");
    buf.elem(
        "svg",
        &[
            ("class", "baux-bench-svg"),
            ("data-view", "paired"),
            ("viewBox", "0 0 376 272"),
            ("role", "img"),
            ("aria-label", &aria),
        ],
        |b| render_paired(b, model),
    );
}
