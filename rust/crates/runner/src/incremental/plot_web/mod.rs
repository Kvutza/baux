//! Deterministic native Rust web-SVG and Forester `.tree` benchmark plot
//! compiler for stabilized incremental studies.

use std::fs;
use std::path::{Path, PathBuf};

use baux::artifact::inc::{ArmInput, PairedReport};

pub mod paired_svg;
pub mod series;
pub mod traj_svg;
pub mod tree;

use paired_svg::render_paired;
use series::{MarkupBuf, MarkupMode, StudyPlotModel, escape_xml, fmt_commas, plot_model};
use traj_svg::render_full;
use tree::render_tree;

const LIGHT_STYLE: &str = ":root {
  --bg-canvas: #ffffff;
  --bg-surface: #f6f7f9;
  --fg-primary: #111111;
  --fg-secondary: #27272a;
  --fg-muted: #64748b;
  --bench-grid: #e5e7eb;
  --bench-axis: #94a3b8;
  --bench-zero: #475569;
  --bench-c-hero: #111827;
  --bench-c-enn1: #1d4ed8;
  --bench-c-enn2: #0f766e;
  --bench-c-enn3: #b45309;
  --bench-c-enn4: #6d28d9;
  --bench-c-ppo: #be123c;
  --font-serif: 'Latin Modern Roman', 'CMU Serif', Georgia, serif;
  --font-mono: 'IBM Plex Mono', 'Latin Modern Mono', ui-monospace, Menlo, monospace;
}";

const DARK_STYLE: &str = ":root {
  --bg-canvas: #0d1117;
  --bg-surface: #161b22;
  --fg-primary: #eef0f4;
  --fg-secondary: #cbd5e1;
  --fg-muted: #8b949e;
  --bench-grid: #21262d;
  --bench-axis: #484f58;
  --bench-zero: #8b949e;
  --bench-c-hero: #f0f6fc;
  --bench-c-enn1: #58a6ff;
  --bench-c-enn2: #2dd4bf;
  --bench-c-enn3: #f59e0b;
  --bench-c-enn4: #c084fc;
  --bench-c-ppo: #fb7185;
  --font-serif: 'Latin Modern Roman', 'CMU Serif', Georgia, serif;
  --font-mono: 'IBM Plex Mono', 'Latin Modern Mono', ui-monospace, Menlo, monospace;
}";

pub fn render_svg(model: &StudyPlotModel, dark: bool) -> String {
    let mut buf = MarkupBuf::new(MarkupMode::SvgXml);
    let css = if dark { DARK_STYLE } else { LIGHT_STYLE };
    let h_str = fmt_commas(model.horizon);
    let aria = escape_xml(&format!(
        "Benchmark trajectory and paired seed differences for {} over {} rounds",
        model.exp, h_str
    ));
    let hdr_b = format!(
        "(b) Paired Δ Error vs. {} at t = {}",
        model.hero.short_full, h_str
    );
    let seeds_lbl = format!("n = {} CRN seeds", model.n_seeds);

    buf.push_raw(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 820 310\" width=\"820\" height=\"310\" role=\"img\" aria-label=\"{aria}\">\n"
    ));
    buf.push_raw(&format!("<style>\n{css}\n</style>\n"));
    buf.leaf(
        "rect",
        &[
            ("x", "0"),
            ("y", "0"),
            ("width", "820"),
            ("height", "310"),
            ("rx", "6"),
            ("fill", "var(--bg-canvas)"),
        ],
    );
    buf.leaf(
        "rect",
        &[
            ("x", "0"),
            ("y", "0"),
            ("width", "820"),
            ("height", "30"),
            ("rx", "6"),
            ("fill", "var(--bg-surface)"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", "0"),
            ("y1", "30"),
            ("x2", "820"),
            ("y2", "30"),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "1"),
        ],
    );
    buf.leaf(
        "line",
        &[
            ("x1", "438"),
            ("y1", "0"),
            ("x2", "438"),
            ("y2", "310"),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "1"),
        ],
    );
    buf.text_elem(
        "text",
        &[
            ("x", "12"),
            ("y", "19"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "12"),
            ("font-weight", "700"),
            ("fill", "var(--fg-primary)"),
        ],
        "(a) Anytime Tracking Error (↓)",
    );
    buf.text_elem(
        "text",
        &[
            ("x", "450"),
            ("y", "19"),
            ("font-family", "var(--font-serif)"),
            ("font-size", "12"),
            ("font-weight", "700"),
            ("fill", "var(--fg-primary)"),
        ],
        &hdr_b,
    );
    buf.text_elem(
        "text",
        &[
            ("x", "808"),
            ("y", "19"),
            ("text-anchor", "end"),
            ("font-family", "var(--font-mono)"),
            ("font-size", "10"),
            ("fill", "var(--fg-muted)"),
        ],
        &seeds_lbl,
    );

    buf.elem("g", &[("transform", "translate(0, 34)")], |g| {
        render_full(g, model);
    });
    buf.elem("g", &[("transform", "translate(442, 34)")], |g| {
        render_paired(g, model);
    });
    buf.leaf(
        "rect",
        &[
            ("x", "0.5"),
            ("y", "0.5"),
            ("width", "819"),
            ("height", "309"),
            ("rx", "6"),
            ("fill", "none"),
            ("stroke", "var(--bench-grid)"),
            ("stroke-width", "1"),
        ],
    );
    buf.push_raw("</svg>\n");
    buf.out
}

fn write_text(path: &Path, content: &str) -> Result<PathBuf, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(path, content).map_err(|e| e.to_string())?;
    Ok(path.to_path_buf())
}

pub fn emit_plots(
    root: &Path,
    run_dir: &Path,
    report: &PairedReport,
    arm_inputs: &[ArmInput],
    exp: &str,
) -> Result<Vec<PathBuf>, String> {
    let model = plot_model(report, arm_inputs, exp)?;
    let tree_doc = render_tree(&model);
    let svg_light = render_svg(&model, false);
    let svg_dark = render_svg(&model, true);

    let mut written = Vec::with_capacity(4);
    written.push(write_text(&run_dir.join("bench_figure.tree"), &tree_doc)?);

    if exp == "tokamak-tokagym" && report.horizon >= 1000 && report.common_reps.len() >= 5 {
        let forest_tree = root.join("forest/trees/baux-0010.tree");
        written.push(write_text(&forest_tree, &tree_doc)?);
    }

    written.push(write_text(
        &run_dir.join("suite_b_plot-light.svg"),
        &svg_light,
    )?);
    written.push(write_text(
        &run_dir.join("suite_b_plot-dark.svg"),
        &svg_dark,
    )?);
    Ok(written)
}
