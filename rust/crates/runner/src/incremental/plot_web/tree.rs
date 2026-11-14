//! Forester `.tree` inline-SVG figure compiler (`baux-0010.tree` /
//! `bench_figure.tree`) adhering strictly to `forest/theme/core.xsl`.

use super::paired_svg::emit_paired;
use super::series::{
    MarkupBuf, MarkupMode, StudyPlotModel, fmt_commas, sanitize_atom, scrub_rounds,
};
use super::traj_svg::{emit_traj, tail_start};

pub fn encode_scrub(model: &StudyPlotModel) -> String {
    let clean_lbl = |s: &str| {
        sanitize_atom(s)
            .chars()
            .filter(|c| !matches!(c, '|' | '~' | ';'))
            .collect::<String>()
    };
    let mut series_refs = Vec::with_capacity(1 + model.non_baseline.len());
    series_refs.push(&model.hero);
    for arm in &model.non_baseline {
        series_refs.push(arm);
    }
    let header: Vec<String> = series_refs.iter().map(|s| clean_lbl(&s.label)).collect();
    let rounds = scrub_rounds(model.horizon);
    let mut rows = Vec::with_capacity(rounds.len());
    for t in rounds {
        let i = t - 1;
        let mut vals = Vec::with_capacity(series_refs.len() * 2);
        for s in &series_refs {
            vals.push(format!("{:.3}", s.mean[i]));
            vals.push(format!("{:.3}", s.sem[i]));
        }
        rows.push(format!("{t}:{}", vals.join(",")));
    }
    format!("{}~{}", header.join("|"), rows.join(";"))
}

pub fn summary_text(model: &StudyPlotModel) -> String {
    let mut parts = Vec::with_capacity(4);
    parts.push(format!("Round t = {}", fmt_commas(model.horizon)));
    parts.push(format!(
        "{}: {:.3} ± {:.3}",
        model.hero.label, model.hero.final_mean, model.hero.final_sem
    ));
    if let Some(best) = model.non_baseline.first() {
        parts.push(format!(
            "{}: {:.3} ± {:.3}",
            best.summary_label, best.final_mean, best.final_sem
        ));
    }
    if let Some(ppo) = model.ppo_ref {
        parts.push(format!("On-Policy PPO: {ppo:.2}"));
    }
    sanitize_atom(&parts.join("  ·  "))
}

fn figure_title(exp: &str) -> &'static str {
    match exp {
        "tokamak-tokagym" => "TCV Rampdown Policy Search Trajectory and Paired Seed Differences",
        "tokamak-tcv" => "TORAX Transport Policy Search Trajectory and Paired Seed Differences",
        "oceananigans" => "Stratified Fluid Control Trajectory and Paired Seed Differences",
        "fuse" => "Tokamak Equilibrium Control Trajectory and Paired Seed Differences",
        _ => "Benchmark Trajectory and Paired Seed Differences",
    }
}

fn horizon_tex(horizon: usize) -> String {
    match horizon {
        100 => "10^2".into(),
        1000 => "10^3".into(),
        10000 => "10^4".into(),
        100000 => "10^5".into(),
        _ => horizon.to_string(),
    }
}

fn regime_labels(horizon: usize) -> (String, String) {
    if horizon == 10000 {
        ("1–10⁴".into(), "Tail 10²–10⁴".into())
    } else {
        let t0 = tail_start(horizon);
        (format!("1–{horizon}"), format!("Tail {t0}–{horizon}"))
    }
}

fn tokagym_caption(model: &StudyPlotModel) -> String {
    let h_mean = format!("{:.3} ± {:.3}", model.hero.final_mean, model.hero.final_sem);
    let mut enn_parts = Vec::new();
    for arm in &model.non_baseline {
        let metric = arm.slug.strip_prefix("turbo-enn_").unwrap_or(&arm.slug);
        let m_clean = sanitize_atom(metric);
        enn_parts.push(format!(
            "\\code{{{:.3} ± {:.3}}} (\\code{{{m_clean}}})",
            arm.final_mean, arm.final_sem
        ));
    }
    let enn_clause = if enn_parts.is_empty() {
        String::new()
    } else {
        format!(
            ", while single-region \\code{{TuRBO-ENN}} (#{{R = 1}}, #{{K = 10}}) reaches {}",
            enn_parts.join(", ")
        )
    };
    let min_ph = model
        .non_baseline
        .iter()
        .map(|a| a.p_holm)
        .fold(1.0f64, f64::min);
    let df = model.n_seeds.saturating_sub(1).max(1);
    format!(
        "Closed-loop TCV rampdown policy search (\\code{{toka_gym}}, #{{D = 136{{,}}712}} policy weights, #{{n = {}}} common-random-number seeds over #{{{}}} rounds). \\strong{{(a)}} Sample mean tracking error (#{{\\pm 1}} SEM shaded) across the full trajectory and zoomed tail (#{{10^2\\text{{--}}10^4}}). Surrogate-free multi-region \\code{{TuRBO-zero}} (#{{R = 10}}) collapses identically across all four sharing topologies (\\code{{origin-only}}, \\code{{box-contain}}, \\code{{ellipsoid-ball}}, \\code{{nearest-center}}) to \\code{{{h_mean}}}{enn_clause}. \\strong{{(b)}} Per-seed paired differences #{{d_s = \\mathcal{{L}}_{{\\mathrm{{arm}},s}} - \\mathcal{{L}}_{{R=10,s}}}} with two-sided #{{1-\\alpha = 0.95}} paired #{{t_{{{df}}}}} intervals (Holm-adjusted sign-flip #{{p_{{\\mathrm{{Holm}}}} \\ge {min_ph:.2}}}).",
        model.n_seeds,
        horizon_tex(model.horizon)
    )
}

fn generic_caption(model: &StudyPlotModel) -> String {
    let h_lbl = sanitize_atom(&model.hero.label);
    let h_stat = format!("{:.3} ± {:.3}", model.hero.final_mean, model.hero.final_sem);
    let df = model.n_seeds.saturating_sub(1).max(1);
    let best_clause = if let Some(best) = model.non_baseline.first() {
        let b_lbl = sanitize_atom(&best.label);
        format!(
            " vs. \\code{{{b_lbl}}} at \\code{{{:.3} ± {:.3}}}",
            best.final_mean, best.final_sem
        )
    } else {
        String::new()
    };
    format!(
        "Paired benchmark evaluation on \\code{{{}}} (#{{n = {}}} common-random-number seeds over #{{{}}} rounds). \\strong{{(a)}} Anytime tracking objective (#{{\\pm 1}} SEM shaded): baseline \\code{{{h_lbl}}} reaches \\code{{{h_stat}}}{best_clause}. \\strong{{(b)}} Per-seed paired differences #{{d_s = \\mathcal{{L}}_{{\\mathrm{{arm}},s}} - \\mathcal{{L}}_{{\\mathrm{{base}},s}}}} with two-sided #{{1-\\alpha = 0.95}} paired #{{t_{{{df}}}}} intervals.",
        model.exp,
        model.n_seeds,
        horizon_tex(model.horizon)
    )
}

fn regime_pills(buf: &mut MarkupBuf, horizon: usize) {
    let (full_lbl, tail_lbl) = regime_labels(horizon);
    buf.elem(
        "div",
        &[
            ("class", "baux-regime-pills"),
            ("role", "group"),
            ("aria-label", "Trajectory Horizon"),
        ],
        |p| {
            p.text_elem(
                "button",
                &[
                    ("type", "button"),
                    ("class", "tcv-pol-btn baux-regime-btn active"),
                    ("data-regime", "full"),
                ],
                &full_lbl,
            );
            p.text_elem(
                "button",
                &[
                    ("type", "button"),
                    ("class", "tcv-pol-btn baux-regime-btn"),
                    ("data-regime", "tail"),
                ],
                &tail_lbl,
            );
        },
    );
}

fn traj_pane(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    buf.elem("div", &[("class", "tcv-pane baux-bench-pane")], |b| {
        b.elem("div", &[("class", "tcv-hud-badge baux-bench-hdr")], |h| {
            h.push_raw("\\<span>{\\strong{(a)} Anytime Tracking Error (#{\\downarrow})}\n");
            regime_pills(h, model.horizon);
        });
        b.elem("div", &[("class", "baux-svg-stage")], |s| {
            emit_traj(s, model);
        });
    });
}

fn paired_pane(buf: &mut MarkupBuf, model: &StudyPlotModel) {
    let h_tex = horizon_tex(model.horizon);
    let base_code = sanitize_atom(&model.hero.short_full);
    let seeds_hint = format!("n = {} CRN seeds", model.n_seeds);
    buf.elem("div", &[("class", "tcv-pane baux-bench-pane")], |b| {
        b.elem("div", &[("class", "tcv-hud-badge baux-bench-hdr")], |h| {
            h.push_raw(&format!(
                "\\<span>{{\\strong{{(b)}} Paired #{{\\Delta}} Error vs. \\code{{{base_code}}} at #{{{h_tex}}}}}\n"
            ));
            h.text_elem("span", &[("class", "tcv-touch-hint")], &seeds_hint);
        });
        b.elem("div", &[("class", "baux-svg-stage")], |s| {
            emit_paired(s, model);
        });
    });
}

pub fn render_tree(model: &StudyPlotModel) -> String {
    let mut buf = MarkupBuf::new(MarkupMode::Forester);
    buf.push_raw(&format!("\\title{{{}}}\n\n", figure_title(&model.exp)));

    let summary = summary_text(model);
    let scrub = encode_scrub(model);
    let caption = if model.exp == "tokamak-tokagym" {
        tokagym_caption(model)
    } else {
        generic_caption(model)
    };

    buf.elem(
        "figure",
        &[
            ("class", "tcv-cad-shell baux-bench-shell"),
            ("data-bench-fig", &model.exp),
            ("data-summary", &summary),
            ("data-scrub", &scrub),
        ],
        |f| {
            f.elem("div", &[("class", "baux-bench-grid")], |g| {
                traj_pane(g, model);
                paired_pane(g, model);
            });
            f.text_elem(
                "div",
                &[("class", "baux-scrub-bar"), ("aria-live", "polite")],
                &summary,
            );
            f.push_raw(&format!(
                "\\<figcaption>[class]{{tcv-fig-caption}}{{{caption}}}\n"
            ));
        },
    );
    buf.out
}
