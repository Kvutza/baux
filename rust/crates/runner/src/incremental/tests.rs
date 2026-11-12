//! Unit tests for incremental broker deduplication, prefix extension, and plan
//! inspection.

use std::path::PathBuf;

use baux::artifact::inc::{ArmInput, IncStore, SeedLossTrace, paired_report};
use baux::artifact::study::{CompiledStudy, StudyCell, StudyPrune};
use baux::artifact::toml::from_str;

use super::engine::{plan_execution, run_study};
use super::plot_web::emit_plots;
use super::plot_web::series::{curve_rounds, repel_labels, scrub_rounds};

fn make_cell(arm_id: &str, rounds: u32, reps: u32) -> StudyCell {
    let raw = r#"
experiment = "tokamak-tcv"
rounds = 10
repetitions = 1

[optimizer]
algorithm = "bayesian-search"
arms = 1
candidates = 4
acquisition = "random"

[optimizer.trust-region]
initial = 0.8
min = 0.00048828125
max = 1.6
"#;
    let val = from_str(raw).expect("parse cell toml");
    StudyCell {
        env_id: "tokamak-tcv".into(),
        arm_id: arm_id.into(),
        cell_id: format!("tokamak-tcv__{arm_id}"),
        baseline: arm_id == "arm_a",
        factors: Vec::new(),
        experiment: "tokamak-tcv".into(),
        rounds,
        repetitions: reps,
        canonical_toml: raw.into(),
        value: val,
    }
}

fn temp_store() -> (PathBuf, IncStore) {
    let dir = std::env::temp_dir().join(format!(
        "baux_inc_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let store = IncStore::open(&dir).expect("open temp inc store");
    (dir, store)
}

#[test]
fn test_singleflight() {
    unsafe {
        std::env::set_var("BAUX_INCREMENTAL_SYNTHETIC", "1");
    }
    let (dir, store) = temp_store();
    let root = PathBuf::from(".");
    let cell_a = make_cell("arm_a", 10, 1);
    let cell_b = make_cell("arm_b", 10, 1);
    let study = CompiledStudy {
        is_multi: true,
        rungs: vec![10],
        prune: StudyPrune::None,
        cells: vec![cell_a, cell_b],
    };

    let res = run_study(&root, &store, &study, Some(1), false);
    assert!(res.is_ok(), "execution should succeed: {:?}", res.err());

    let stats = store.stats().expect("store stats");
    assert_eq!(
        stats.eval_memos, 10,
        "2 identical arms x 10 rounds must run 10 evaluations via Singleflight"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_prefix() {
    unsafe {
        std::env::set_var("BAUX_INCREMENTAL_SYNTHETIC", "1");
    }
    let (dir, store) = temp_store();
    let root = PathBuf::from(".");

    let study_6 = CompiledStudy {
        is_multi: false,
        rungs: vec![6],
        prune: StudyPrune::None,
        cells: vec![make_cell("arm_ext", 6, 1)],
    };
    run_study(&root, &store, &study_6, Some(1), false).expect("exec 6");
    let stats_6 = store.stats().expect("stats 6");
    assert_eq!(stats_6.eval_memos, 6);

    let study_12 = CompiledStudy {
        is_multi: false,
        rungs: vec![12],
        prune: StudyPrune::None,
        cells: vec![make_cell("arm_ext", 12, 1)],
    };
    run_study(&root, &store, &study_12, Some(1), false).expect("exec 12");
    let stats_12 = store.stats().expect("stats 12");
    assert_eq!(
        stats_12.eval_memos, 12,
        "extension should only run 6 new evaluations"
    );

    let study_4 = CompiledStudy {
        is_multi: false,
        rungs: vec![4],
        prune: StudyPrune::None,
        cells: vec![make_cell("arm_ext", 4, 1)],
    };
    run_study(&root, &store, &study_4, Some(1), false).expect("exec 4");
    let stats_4 = store.stats().expect("stats 4");
    assert_eq!(
        stats_4.eval_memos, 12,
        "cutoff should run 0 new evaluations"
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn test_plan() {
    unsafe {
        std::env::set_var("BAUX_INCREMENTAL_SYNTHETIC", "1");
    }
    let (dir, store) = temp_store();
    let root = PathBuf::from(".");
    let study = CompiledStudy {
        is_multi: true,
        rungs: vec![10],
        prune: StudyPrune::None,
        cells: vec![make_cell("arm_plan", 10, 2)],
    };

    let plan_res = plan_execution(&root, &store, &study);
    assert!(plan_res.is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

fn synthetic_arm(
    cell_id: &str,
    arm_id: &str,
    baseline: bool,
    horizon: usize,
    end: f64,
) -> ArmInput {
    let traces = (0..10u32)
        .map(|rep| {
            let offset = ((rep as f64) - 4.5) * 0.015;
            let curve: Vec<f64> = (1..=horizon)
                .map(|t| end + offset + 56.0 / ((t as f64).sqrt()))
                .collect();
            SeedLossTrace {
                rep,
                seed: 1000 + u64::from(rep),
                init_loss: curve[0],
                final_loss: *curve.last().unwrap(),
                accepts: 12,
                elapsed_s: 0.1,
                loss_curve: curve,
            }
        })
        .collect();
    ArmInput {
        cell_id: cell_id.into(),
        arm_id: arm_id.into(),
        baseline,
        factors: Vec::new(),
        traces,
    }
}

const ALLOWED_TAGS: &[&str] = &[
    "figure",
    "figcaption",
    "div",
    "span",
    "button",
    "strong",
    "code",
    "em",
    "svg",
    "g",
    "path",
    "polyline",
    "polygon",
    "line",
    "rect",
    "circle",
    "text",
    "tspan",
    "defs",
    "clipPath",
    "title",
];

const ALLOWED_ATTRS: &[&str] = &[
    "id",
    "class",
    "type",
    "hidden",
    "role",
    "aria-label",
    "aria-live",
    "aria-hidden",
    "focusable",
    "data-bench-fig",
    "data-regime",
    "data-view",
    "data-scrub",
    "data-seed",
    "data-arm",
    "data-summary",
    "data-x-left",
    "data-x-right",
    "data-log-min",
    "data-log-max",
    "viewBox",
    "width",
    "height",
    "x",
    "y",
    "dx",
    "dy",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "d",
    "points",
    "fill",
    "fill-opacity",
    "stroke",
    "stroke-width",
    "stroke-dasharray",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-opacity",
    "opacity",
    "text-anchor",
    "dominant-baseline",
    "font-size",
    "font-weight",
    "font-family",
    "letter-spacing",
    "transform",
    "clip-path",
    "vector-effect",
];

fn assert_forester(tree: &str) {
    assert!(!tree.contains('%'), "bare or escaped % forbidden in .tree");
    assert!(
        !tree.contains("[style]"),
        "style attribute forbidden in .tree"
    );
    let bytes = tree.as_bytes();
    let (mut braces, mut brackets) = (0i32, 0i32);
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'#' {
            assert_eq!(bytes.get(i + 1), Some(&b'{'), "bare # at byte {i}");
        }
        match b {
            b'{' => braces += 1,
            b'}' => braces -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            _ => {}
        }
        assert!(braces >= 0 && brackets >= 0, "unbalanced delimiter at {i}");
    }
    assert_eq!((braces, brackets), (0, 0), "unclosed delimiters in .tree");

    for seg in tree.split("\\<").skip(1) {
        let (tag, rest) = seg.split_once('>').expect("tag close");
        assert!(ALLOWED_TAGS.contains(&tag), "disallowed tag: {tag}");
        let mut rem = rest;
        while let Some(after_open) = rem.strip_prefix('[') {
            let (attr, tail) = after_open.split_once("]{").expect("attr value start");
            assert!(ALLOWED_ATTRS.contains(&attr), "disallowed attr: {attr}");
            let (val, next) = tail.split_once('}').expect("attr value end");
            assert!(
                !val.chars()
                    .any(|c| matches!(c, '#' | '%' | '\\' | '{' | '}' | '[' | ']')),
                "unescaped special char in [{attr}]{{{val}}}"
            );
            rem = next;
        }
    }
}

#[test]
fn test_downsampling() {
    let scrub = scrub_rounds(10_000);
    assert!((40..=55).contains(&scrub.len()));
    for req in [1, 10, 25, 50, 100, 250, 500, 1000, 3000, 10_000] {
        assert!(scrub.contains(&req), "missing milestone {req}");
    }
    let full_pts = curve_rounds(1, 10_000);
    assert!(full_pts.len() <= 180 && full_pts[0] == 1 && *full_pts.last().unwrap() == 10_000);

    let ys = repel_labels(&[198.1, 193.3, 198.7, 211.8, 202.2], 22.0, 225.0, 12.5);
    let mut sorted = ys.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(sorted[0] >= 22.0 && *sorted.last().unwrap() <= 225.0);
    for pair in sorted.windows(2) {
        assert!(pair[1] - pair[0] >= 12.49, "overlap: {:?}", sorted);
    }
}

fn assert_standalone(tree: &str, run_dir: &std::path::Path) {
    for view in ["[data-view]{full}", "[data-view]{tail}"] {
        let sec = tree
            .split(view)
            .nth(1)
            .unwrap()
            .split("\\<svg>")
            .next()
            .unwrap();
        let mut ys: Vec<f64> = sec
            .split("\\<text>[x]{335}[y]{")
            .skip(1)
            .map(|s| s.split_once('}').unwrap().0.parse::<f64>().unwrap())
            .collect();
        assert!(ys.len() >= 3);
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in ys.windows(2) {
            assert!(w[1] - w[0] >= 12.4, "svg label collision in {view}: {ys:?}");
        }
    }
    for name in ["suite_b_plot-light.svg", "suite_b_plot-dark.svg"] {
        let svg = std::fs::read_to_string(run_dir.join(name)).unwrap();
        assert!(
            svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 820 310\"")
        );
        assert!(svg.trim_end().ends_with("</svg>") && svg.contains("<style>"));
        assert_eq!(svg.matches("<g ").count(), svg.matches("</g>").count());
        assert_eq!(
            svg.matches("<text ").count(),
            svg.matches("</text>").count()
        );
    }
}

#[test]
fn test_plots() {
    let dir = temp_store().0;
    let run_dir = dir.join("run");
    let arms = vec![
        synthetic_arm("turbo-zero_origin-only", "turbo-zero", true, 1000, 1.311),
        synthetic_arm("turbo-zero_box-contain", "turbo-zero", false, 1000, 1.311),
        synthetic_arm(
            "turbo-zero_ellipsoid-ball",
            "turbo-zero",
            false,
            1000,
            1.311,
        ),
        synthetic_arm(
            "turbo-zero_nearest-center",
            "turbo-zero",
            false,
            1000,
            1.311,
        ),
        synthetic_arm("turbo-enn_conformal", "turbo-enn", false, 1000, 1.369),
        synthetic_arm("turbo-enn_wasserstein", "turbo-enn", false, 1000, 1.341),
        synthetic_arm("turbo-enn_euclidean", "turbo-enn", false, 1000, 1.352),
        synthetic_arm("turbo-enn_hyperbolic", "turbo-enn", false, 1000, 1.354),
    ];
    let rep = paired_report("tokamak-tokagym", 1000, &arms).expect("paired report");
    let out = emit_plots(&dir, &run_dir, &rep, &arms, "tokamak-tokagym").expect("emit");
    assert_eq!(out.len(), 4);

    let tree = std::fs::read_to_string(dir.join("forest/trees/baux-0010.tree")).unwrap();
    assert_forester(&tree);
    assert!(tree.contains("4 topologies ≡ 0") && tree.contains("On-Policy PPO: 20.59"));
    assert!(tree.contains("[data-scrub]{TuRBO-0 (R=10)|ENN · Wasserstein|ENN · Euclidean|ENN · Hyperbolic|ENN · Conformal~1:"));
    assert_standalone(&tree, &run_dir);

    let gen_arms = vec![
        synthetic_arm("base_arm", "base", true, 50, 2.0),
        synthetic_arm("bad#slug%name{1}", "chal", false, 50, 2.2),
    ];
    let gen_rep = paired_report("oceananigans", 50, &gen_arms).unwrap();
    let gen_dir = dir.join("gen_run");
    let gen_out = emit_plots(&dir, &gen_dir, &gen_rep, &gen_arms, "oceananigans").unwrap();
    assert_eq!(gen_out.len(), 3);
    let gen_tree = std::fs::read_to_string(gen_dir.join("bench_figure.tree")).unwrap();
    assert_forester(&gen_tree);
    assert!(!gen_tree.contains("On-Policy PPO"));

    let _ = std::fs::remove_dir_all(dir);
}
