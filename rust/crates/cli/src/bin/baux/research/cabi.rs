use std::fs;
use std::path::Path;
use std::process::Command;

use crate::cli_cfg::BuildSystem;

const RUNTIME_TARGET: &str = "//rust/crates/runner:baux-runner";

pub(super) fn ensure_cache(root: &Path) -> Result<(), String> {
    let lean_root = root.join("research/lean");
    let mathlib =
        lean_root.join(".lake/packages/mathlib/.lake/build/lib/lean/Mathlib/Data/Real/Basic.olean");
    if mathlib.is_file() {
        return Ok(());
    }
    let mut cache = Command::new("lake");
    cache.current_dir(lean_root).args(["exe", "cache", "get"]);
    crate::execute(cache)
}

pub(super) fn verify_cabi(root: &Path) -> Result<(), String> {
    let archive = root.join("research/lean/.lake/build/lib/libbaux__research_BauxResearch.a");
    if !archive.is_file() {
        return Err(format!(
            "missing compiled Lean 4 C-ABI static archive {}",
            archive.display()
        ));
    }
    let bytes = fs::read(&archive)
        .map_err(|error| format!("cannot read {}: {error}", archive.display()))?;
    for symbol in [
        "baux_accept",
        "baux_action",
        "baux_world",
        "baux_anchored",
        "baux_routed",
        "baux_total",
        "baux_window",
        "baux_cell",
        "baux_slot",
        "baux_sign",
    ] {
        if !bytes
            .windows(symbol.len())
            .any(|window| window == symbol.as_bytes())
        {
            return Err(format!(
                "compiled Lean 4 static archive lacks C-ABI symbol `{symbol}`"
            ));
        }
    }
    let cache = root.join(".cache/baux/cabi");
    fs::create_dir_all(&cache)
        .map_err(|error| format!("cannot create {}: {error}", cache.display()))?;
    let sem_dir = root.join("rust/crates/baux/semantics/src");
    let harness_src = cache.join("cabi_parity.rs");
    let harness_bin = cache.join("cabi_parity");
    fs::write(
        &harness_src,
        format!(
            r#"#![allow(dead_code)]
extern crate self as baux;
pub mod semantics {{
    pub use crate::*;
    pub mod optimizer {{
        pub use crate::*;
        pub mod surrogate {{
            pub use crate::radius::*;
        }}
    }}
}}
#[path = "{sem}/optimizer/calibration.rs"]
pub mod calibration;
#[path = "{sem}/numerical.rs"]
pub mod numerical;
#[path = "{sem}/optimizer/trust_region/controller.rs"]
pub mod radius;
#[path = "{sem}/optimizer/provenance.rs"]
pub mod provenance;
#[path = "{sem}/shape.rs"]
pub mod shape;
#[path = "{sem}/optimizer/param_tree.rs"]
pub mod param_tree;
#[path = "{sem}/optimizer/reuse.rs"]
pub mod reuse;
#[path = "{sem}/neighbor.rs"]
pub mod neighbor;

use param_tree::{{ParamTree, routed_count, total_count}};
use provenance::{{exact_anchored, same_world}};
use radius::{{RadiusAction, accept_bits, decide_action}};
use reuse::{{conditioned_sign, conditioned_slot}};

unsafe extern "C" {{
    fn baux_accept(improvement: u64, margin: u64) -> u8;
    fn baux_action(
        adequate: u8,
        inadequate: u8,
        sharp: u8,
        accepted: u8,
        positive: u8,
        boundary: u8,
        failure: u8,
    ) -> u8;
    fn baux_world(left: u64, right: u64) -> u8;
    fn baux_anchored(is_fixed: u8, from_step: u64, to_step: u64) -> u8;
    fn baux_routed(patch: u64) -> u64;
    fn baux_total(patch: u64) -> u64;
    fn baux_window(planned: u64, prefilled: u64, scored: u64) -> u8;
    fn baux_cell(level: u32, grid: u32) -> u64;
    fn baux_slot(slot: u64, accepted: u8) -> u64;
    fn baux_sign(reversed: u8, accepted: u8) -> u8;
}}

fn main() {{
    let samples: [u64; 12] = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0xc000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_0001,
    ];
    for &imp in &samples {{
        for &mar in &samples {{
            let rust_acc = accept_bits(imp, mar);
            let lean_acc = unsafe {{ baux_accept(imp, mar) != 0 }};
            assert_eq!(rust_acc, lean_acc, "accept_bits mismatch for ({{imp:#x}}, {{mar:#x}})");
        }}
    }}
    for mask in 0u8..128 {{
        let b = |bit: u8| (mask >> bit) & 1 != 0;
        let rust_act = match decide_action(b(0), b(1), b(2), b(3), b(4), b(5), b(6)) {{
            RadiusAction::Hold => 0u8,
            RadiusAction::Expand => 1u8,
            RadiusAction::Shrink => 2u8,
        }};
        let lean_act = unsafe {{
            baux_action(
                u8::from(b(0)),
                u8::from(b(1)),
                u8::from(b(2)),
                u8::from(b(3)),
                u8::from(b(4)),
                u8::from(b(5)),
                u8::from(b(6)),
            )
        }};
        assert_eq!(rust_act, lean_act, "decide_action mismatch at mask {{mask:#09b}}");
    }}
    for &(l, r) in &[(0u64, 0u64), (42, 42), (42, 43), (u64::MAX, u64::MAX - 1)] {{
        assert_eq!(same_world(l, r), unsafe {{ baux_world(l, r) != 0 }});
        for fixed in [false, true] {{
            assert_eq!(
                exact_anchored(fixed, l, r),
                unsafe {{ baux_anchored(u8::from(fixed), l, r) != 0 }}
            );
        }}
    }}
    assert_eq!(routed_count(), unsafe {{ baux_routed(1) }});
    assert_eq!(total_count(), unsafe {{ baux_total(1) }});
    assert_eq!(
        ParamTree::<usize>::total_parameters(1) as u64,
        unsafe {{ baux_total(1) }}
    );
    for &(planned, prefilled, scored) in &[
        (0u64, 0u64, 0u64),
        (4096, 0, 4096),
        (1_048_576, 0, 1_048_576),
        (1_048_576, 2048, 1024),
        (4096, 0, 2048),
    ] {{
        assert_eq!(
            planned > 0 && prefilled == 0 && scored == planned,
            unsafe {{ baux_window(planned, prefilled, scored) != 0 }}
        );
    }}
    assert_eq!(
        neighbor::grid_cell(7.0, 8.0),
        unsafe {{ baux_cell(3, 0) }}
    );
    for slot in 0u64..64 {{
        for acc in [false, true] {{
            assert_eq!(
                conditioned_slot(slot, acc),
                unsafe {{ baux_slot(slot, u8::from(acc)) }}
            );
        }}
    }}
    for rev in [false, true] {{
        for acc in [false, true] {{
            assert_eq!(
                conditioned_sign(rev, acc),
                unsafe {{ baux_sign(u8::from(rev), u8::from(acc)) != 0 }}
            );
        }}
    }}
}}
"#,
            sem = sem_dir.display()
        ),
    )
    .map_err(|error| format!("cannot write {}: {error}", harness_src.display()))?;
    let strip_flag = if cfg!(target_os = "macos") {
        "-C\nlink-arg=-Wl,-dead_strip"
    } else {
        "-C\nlink-arg=-Wl,--gc-sections"
    };
    let mut rustc_bin = Command::new("rustc");
    rustc_bin
        .arg("--edition=2024")
        .arg("--cfg=cabi_parity")
        .arg(&harness_src)
        .arg(format!("-Clink-arg={}", archive.display()));
    for arg in strip_flag.lines() {
        rustc_bin.arg(arg);
    }
    rustc_bin.arg("-o").arg(&harness_bin);
    crate::execute(rustc_bin)?;
    crate::execute(Command::new(&harness_bin))?;
    Ok(())
}

pub(super) fn check_contract(
    root: &Path,
    build_system: BuildSystem,
    target: &str,
) -> Result<(), String> {
    let value = |key: &str| -> Result<&str, String> {
        target
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .map(str::trim)
            .ok_or_else(|| format!("Lean target output lacks `{key}`"))
    };
    let number = |key: &str| -> Result<u64, String> {
        value(key)?
            .parse::<u64>()
            .map_err(|error| format!("invalid Lean target value for `{key}`: {error}"))
    };

    let args = vec![
        "research-contract".to_owned(),
        "performance/configs/mhc4-1m-causal-1000.toml".to_owned(),
        "--parameters".to_owned(),
        number("target.parameters: ")?.to_string(),
        "--context-tokens".to_owned(),
        number("target.context_tokens: ")?.to_string(),
        "--candidates".to_owned(),
        number("target.candidates: ")?.to_string(),
        "--neighbors".to_owned(),
        number("target.neighbors: ")?.to_string(),
        "--target-ms".to_owned(),
        number("target.transaction_target_ms: ")?.to_string(),
        "--ceiling-ms".to_owned(),
        number("target.transaction_ceiling_ms: ")?.to_string(),
        "--dense-flops".to_owned(),
        number("target.dense_flops: ")?.to_string(),
        "--objective".to_owned(),
        value("target.objective: ")?.to_owned(),
        "--sampling".to_owned(),
        value("target.sampling: ")?.to_owned(),
    ];
    build_system.run(root, RUNTIME_TARGET, &args)
}
