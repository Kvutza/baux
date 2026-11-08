//! Trail of Bits-style in-toto Statement v1 (`https://in-toto.io/Statement/v1`)
//! and SLSA Provenance v1.0 (`https://slsa.dev/provenance/v1`) generator and
//! verifier with BLAKE3 artifact digests.

#[path = "attestation/notice.rs"]
pub mod notice;

use crate::artifact::{
    blake3,
    json::{self, Value},
};
use crate::wire_json;
use std::fs;
use std::path::Path;

pub use notice::{CANONICAL_NOTICE, render_notice, verify_notices};

pub const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
pub const PREDICATE_TYPE: &str = "https://slsa.dev/provenance/v1";
pub const BUILD_TYPE: &str = "https://kvutza.github.io/baux/slsa/buildTypes/buck2-lean4-oxcaml/v1";

const SUBJECT_PATHS: &[&str] = &[
    "LICENSE",
    "rust/crates/baux/LICENSE",
    "NOTICE",
    "rust/crates/baux/NOTICE",
    "Cargo.toml",
    "research/extraction/artifact_manifest.toml",
    "research/extraction/BauxBo.llbc",
    "research/upstream/acme.lock.toml",
    "research/upstream/enn.lock.toml",
    "research/upstream/hnet.lock.toml",
    "research/upstream/mhc.lock.toml",
    "research/upstream/muon.lock.toml",
    "research/upstream/polar_express.lock.toml",
    "research/upstream/tokagym.lock.toml",
    "research/upstream/torax.lock.toml",
    "ocaml/baux_benchmarks.opam",
    "ocaml/lib/formulas.ml",
    "ocaml/hardware/distance.ml",
];

fn dependency(
    name: &str,
    uri: &str,
    version: &str,
    license: &str,
    copyright: &str,
    scope: &str,
    citation: &str,
) -> Value {
    wire_json!({
        "name": name,
        "uri": uri,
        "version": version,
        "license": license,
        "copyright": copyright,
        "scope": scope,
        "citation": citation
    })
}

pub fn resolved_dependencies() -> Vec<Value> {
    vec![
        dependency(
            "enn",
            "git+https://github.com/yubo-research/enn@f8f6a5dada9e85d1a196b7281be4b1b070e0a3ef",
            "0.4.1",
            "MIT",
            "Copyright (c) 2025 yubo research",
            "foundational-port",
            "arXiv:2502.10390",
        ),
        dependency(
            "TuRBO-specification",
            "https://arxiv.org/abs/1910.01739",
            "NeurIPS-2019",
            "MIT (BoTorch reference)",
            "Eriksson et al. (2019); Meta Platforms, Inc. (BoTorch)",
            "clean-room-mathematical-specification",
            "arXiv:1910.01739",
        ),
        dependency(
            "MORBO-specification",
            "https://arxiv.org/abs/2109.10964",
            "UAI-2022",
            "MIT (BoTorch reference)",
            "Daulton, Eriksson, Balandat, Bakshy (2022)",
            "clean-room-mathematical-specification",
            "arXiv:2109.10964",
        ),
        dependency(
            "hnet",
            "git+https://github.com/goombalab/hnet@main",
            "2025",
            "Apache-2.0",
            "Copyright (c) 2025 Goomba Lab (Sukjun Hwang, Brandon Wang, Albert Gu)",
            "reference-and-kernel-port",
            "arXiv:2507.07955",
        ),
        dependency(
            "mlx",
            "https://github.com/ml-explore/mlx",
            "2024",
            "MIT",
            "Copyright (c) 2023-2024 Apple Inc.",
            "metal-simd-reduction-idioms",
            "https://github.com/ml-explore/mlx",
        ),
        dependency(
            "Muon",
            "git+https://github.com/KellerJordan/Muon@f98f1cacc0263b04290753e32be8d498c1efc806",
            "f98f1ca",
            "MIT",
            "Copyright (c) 2024 Keller Jordan and contributors",
            "optimizer-reference-and-port",
            "https://github.com/KellerJordan/Muon",
        ),
        dependency(
            "PolarExpress",
            "git+https://github.com/NoahAmsel/PolarExpress@f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe",
            "f717e2c",
            "MIT",
            "Copyright (c) 2025 Noah Amsel, David Persson, Christopher Musco, Robert M. Gower",
            "matrix-sign-polynomial-port",
            "arXiv:2505.16932",
        ),
        dependency(
            "mHC",
            "git+https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections@a0ea62c",
            "a0ea62c",
            "MIT",
            "Copyright (c) 2025 tokenbender; Xie et al. (DeepSeek-AI)",
            "sinkhorn-knopp-manifold-reference",
            "arXiv:2512.24880",
        ),
        dependency(
            "torax",
            "https://github.com/google-deepmind/torax",
            "2.0.0",
            "Apache-2.0",
            "Copyright 2024 DeepMind Technologies Limited",
            "tokamak-pde-transport-benchmark",
            "arXiv:2406.06718",
        ),
        dependency(
            "TokaGym",
            "https://doi.org/10.5281/zenodo.16621120",
            "16621120",
            "MIT",
            "Copyright (c) 2025 Allen M. Wang, Christian Theiler, et al. (MIT / EPFL)",
            "tcv-rampdown-control-benchmark",
            "doi:10.1038/s41467-025-63917-3",
        ),
        dependency(
            "acme",
            "git+https://github.com/google-deepmind/acme@b2886f8",
            "b2886f8",
            "Apache-2.0",
            "Copyright 2018 DeepMind Technologies Limited",
            "rl-interface-reference",
            "arXiv:2006.00979",
        ),
        dependency(
            "oxcaml-hardcaml",
            "https://github.com/oxcaml/oxcaml",
            "5.2.0minus-ox",
            "MIT / LGPL-2.1-only WITH OCaml-LGPL-linking-exception",
            "Copyright (c) Jane Street Group, LLC and contributors",
            "unboxed-simd-and-rtl-hardware-circuits",
            "https://github.com/janestreet/hardcaml",
        ),
        dependency(
            "sleef",
            "https://github.com/shibatch/sleef",
            "3.9.0",
            "BSL-1.0",
            "Copyright (c) Naoki Shibata and contributors",
            "neon-vector-transcendentals",
            "https://sleef.org",
        ),
        dependency(
            "scipy-optimize-benchmarks",
            "https://github.com/scipy/scipy",
            "1.15",
            "BSD-3-Clause",
            "Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers",
            "sfu-reference-test-generator",
            "https://scipy.org",
        ),
        dependency(
            "usearch",
            "https://github.com/unum-cloud/usearch",
            "2.26",
            "Apache-2.0",
            "Copyright (c) 2023 Ash Vardanian / Unum Cloud, Inc.",
            "vendored-native-library",
            "vendor/usearch",
        ),
        dependency(
            "candle",
            "https://github.com/huggingface/candle",
            "0.9.2",
            "Apache-2.0 OR MIT",
            "Copyright (c) 2022-2025 Hugging Face Inc.",
            "statically-linked-crate",
            "Cargo.lock",
        ),
        dependency(
            "pyo3",
            "https://github.com/PyO3/pyo3",
            "0.27",
            "MIT OR Apache-2.0",
            "Copyright (c) PyO3 Project and Contributors",
            "statically-linked-crate",
            "Cargo.lock",
        ),
    ]
}

pub fn build_statement(root: &Path) -> Result<Value, String> {
    let mut subjects = Vec::new();
    for &rel in SUBJECT_PATHS {
        let full = root.join(rel);
        if !full.is_file() {
            continue;
        }
        let digest = blake3::hash_file(&full)?;
        let size = fs::metadata(&full)
            .map_err(|e| format!("{}: {e}", full.display()))?
            .len();
        subjects.push(wire_json!({
            "name": rel,
            "size_bytes": size,
            "digest": {
                "blake3": digest
            }
        }));
    }
    Ok(wire_json!({
        "_type": STATEMENT_TYPE,
        "subject": subjects,
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": BUILD_TYPE,
                "externalParameters": {
                    "repository": "https://github.com/Kvutza/baux",
                    "license": "MIT",
                    "copyright": "Copyright (c) 2026 Kvutza",
                    "hash_algorithm": "blake3",
                    "toolchains": [
                        "buck2",
                        "charon",
                        "aeneas",
                        "lean4",
                        "oxcaml",
                        "hardcaml"
                    ]
                },
                "resolvedDependencies": resolved_dependencies()
            },
            "runDetails": {
                "builder": {
                    "id": "https://github.com/Kvutza/baux/actions/workflows/verify"
                }
            }
        }
    }))
}

pub fn verify_statement(root: &Path) -> Result<Value, String> {
    verify_notices(root)?;
    let path = root.join("research/extraction/provenance.intoto.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let recorded: Value =
        json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let expected = build_statement(root)?;
    if recorded != expected {
        return Err(
            "research/extraction/provenance.intoto.json is out of date with tracked BLAKE3 subjects"
                .into(),
        );
    }
    Ok(recorded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notice_sync() {
        let Ok(root) = std::env::current_dir() else {
            return;
        };
        if root.join("NOTICE").is_file() {
            verify_statement(&root)
                .expect("license, NOTICE, and provenance.intoto.json must match canonical state");
        }
    }
}
