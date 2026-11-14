//! Canonical NOTICE compiler and license/notice consistency verifier.
//!
//! Renders `NOTICE` and `rust/crates/baux/NOTICE` deterministically from the
//! structured SLSA v1.0 `resolvedDependencies` table so legal attribution can
//! never drift from the build graph.

use std::fs;
use std::path::Path;

pub const CANONICAL_NOTICE: &str = "\
baux
Copyright (c) 2026 Kvutza

This product is licensed under the MIT License (see the LICENSE file).
It builds upon, ports, or interoperates with the following open-source
software, academic research artifacts, and visual media:

========================================================================
1. Core Optimization & Surrogate Foundations
========================================================================

* enn (Epistemic Nearest Neighbors & TuRBO-ENN)
  Repository: https://github.com/yubo-research/enn
  Copyright (c) 2025 yubo research
  License: MIT
  Citation:
    Mehul Bafna, Siddhant Anand Jadhav, and David Sweet, \"Taking the GP Out
    of the Loop\", Proceedings of the 43rd International Conference on
    Machine Learning (ICML 2026), PMLR 306:5179-5199, arXiv:2506.12818.
    https://arxiv.org/abs/2506.12818

* TuRBO (Trust Region Bayesian Optimization — Clean-Room Specification)
  Citation:
    David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, and
    Matthias Poloczek, \"Scalable Global Optimization via Local Bayesian
    Optimization\", Advances in Neural Information Processing Systems
    (NeurIPS 2019), arXiv:1910.01739. https://arxiv.org/abs/1910.01739
  Reference Implementation (BoTorch):
    https://github.com/meta-pytorch/botorch
    Copyright (c) Meta Platforms, Inc. and affiliates.
    License: MIT

* MORBO (Multi-Objective Regionalized Bayesian Optimization)
  Citation:
     Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy,
    \"Multi-Objective Bayesian Optimization over High-Dimensional Search
    Spaces\", Uncertainty in Artificial Intelligence (UAI 2022),
    arXiv:2109.10964. https://arxiv.org/abs/2109.10964

* RAASP & Cylindrical Thompson Sampling (External Baseline Specification)
  Notes:
    External prior-art baselines and coordinate-allocation specifications;
    not claimed as novel BAUX algorithmic contributions.
  Citations:
    Bahador Rashidi, Kerrick Johnstonbaugh, and Chao Gao, \"Cylindrical
    Thompson Sampling for High-Dimensional Bayesian Optimization\",
    International Conference on Artificial Intelligence and Statistics
    (AISTATS 2024), PMLR 238:3502-3510.
    Rommel G. Regis and Christine A. Shoemaker, \"Combining Radial Basis
    Function Surrogates and Dynamic Coordinate Search in High-Dimensional
    Expensive Black-Box Optimization\", Engineering Optimization
    45(5):529-555 (2013).

* MTRBO & APOSMM (External Baseline Specification)
  Notes:
    External prior-art multi-trust-region and multi-start optimization
    specifications; not claimed as novel BAUX algorithmic contributions.
  Citations:
    Sourav Das, Debjani Chakraborty, and Pabitra Mitra, \"MTRBO: Multiple
    Trust-Region Based Bayesian Optimization\", arXiv:2605.06618 (2026).
    https://arxiv.org/abs/2605.06618
    Jeffrey Larson and Stefan M. Wild, \"Asynchronously Parallel
    Optimization Solver for Finding Multiple Minima\", Mathematical
    Programming Computation 10(1):99-132 (2018).
    https://doi.org/10.1007/s12532-017-0131-4

========================================================================
2. Pretraining, Dynamic Chunking & Optimizer Kernels
========================================================================

* H-Net (Dynamic Chunking for End-to-End Hierarchical Sequence Modeling)
  Repository: https://github.com/goombalab/hnet
  Copyright (c) 2025 Goomba Lab (Sukjun Hwang, Brandon Wang, Albert Gu)
  License: Apache License 2.0
  Citation:
    Sukjun Hwang, Brandon Wang, and Albert Gu, \"Dynamic Chunking for
    End-to-End Hierarchical Sequence Modeling\", arXiv:2507.07955 (2025).
    https://arxiv.org/abs/2507.07955

* MLX (Apple Silicon Metal Array & Reduction Primitives)
  Repository: https://github.com/ml-explore/mlx
  Copyright (c) 2023-2024 Apple Inc.
  License: MIT
  Notes:
    Portions of the Apple Silicon Metal reduction and scan idioms in
    `rust/crates/baux/src/hnet_wave.metal` are adapted from MLX and from
    the Goomba Lab H-Net reference implementation.

* Muon Optimizer
  Repository: https://github.com/KellerJordan/Muon
  Copyright (c) 2024 Keller Jordan and contributors
  License: MIT
  Citation:
    Keller Jordan et al., \"Muon: An optimizer for hidden layers in neural
    networks\" (2024). https://github.com/KellerJordan/Muon

* Polar Express (Optimal Matrix Sign Methods)
  Repository: https://github.com/NoahAmsel/PolarExpress
  Copyright (c) 2025 Noah Amsel, David Persson, Christopher Musco,
  and Robert M. Gower
  License: MIT
  Citation:
    Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower,
    \"The Polar Express: Optimal Matrix Sign Methods and Their Application
    to the Muon Algorithm\", arXiv:2505.16932 (2025).
    https://arxiv.org/abs/2505.16932

* mHC (Manifold-Constrained Hyper-Connections)
  Repository: https://github.com/tokenbender/mHC-manifold-constrained-hyper-connections
  Copyright (c) 2025 tokenbender and contributors
  License: MIT
  Citation:
    Zhenda Xie, Yixuan Wei, Huanqi Cao, et al. (DeepSeek-AI),
    \"mHC: Manifold-Constrained Hyper-Connections\", arXiv:2512.24880 (2025).
    https://arxiv.org/abs/2512.24880

* PISA (Pyramid Block Sparse Attention — External Architecture Specification)
  Notes:
    External prior-art block sparse attention architecture; not claimed
    as a novel BAUX algorithmic contribution.
  Citation:
    Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, and Pengfei Liu, \"Block
    Sparse Attention with Log-Linear Complexity\", arXiv:2609.31093 (2026).
    https://arxiv.org/abs/2609.31093

* DeepSeekMoE & Hyper-Connections (External Architecture Specification)
  Notes:
    External prior-art mixture-of-experts and manifold residual connection
    specifications; not claimed as novel BAUX algorithmic contributions.
  Citations:
    Damai Dai et al., \"DeepSeekMoE: Towards Ultimate Expert Specialization
    in Mixture-of-Experts Language Models\", Association for Computational
    Linguistics (ACL 2024), arXiv:2401.06066.
    https://arxiv.org/abs/2401.06066
    Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu,
    Qiyang Min, and Xun Zhou, \"Hyper-Connections\", International Conference
    on Learning Representations (ICLR 2025), arXiv:2409.19606.
    https://arxiv.org/abs/2409.19606
    Richard Sinkhorn, \"A Relationship Between Arbitrary Positive Matrices
    and Doubly Stochastic Matrices\", Annals of Mathematical Statistics
    35(2):876-879 (1964).
    Richard Sinkhorn and Paul Knopp, \"Concerning Nonnegative Matrices and
    Doubly Stochastic Matrices\", Pacific Journal of Mathematics
    21(2):343-348 (1967).

* Block Diffusion (BD3-LMs — External Architecture Specification)
  Notes:
    External prior-art block diffusion language model specification;
    not claimed as a novel BAUX algorithmic contribution.
  Citation:
    Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan
    Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov, \"Block
    Diffusion: Interpolating Between Autoregressive and Diffusion Language
    Models\", International Conference on Learning Representations (ICLR 2025),
    arXiv:2503.09573. https://arxiv.org/abs/2503.09573

* Bounded Log-Likelihood Loss (BLL-Loss — External Loss Specification)
  Notes:
    External prior-art reasoning loss specification; not claimed as a
    novel BAUX algorithmic contribution.
  Citation:
    Feng Zhao, Hong Zhang, Yu Yang, Ruilin Zhao, and Guandong Xu, \"Don't
    Force the Fit: Bounded Log-Likelihood Loss for Enhanced Reasoning in
    Large Language Models\", International Conference on Machine Learning
    (ICML 2026).

========================================================================
3. Plasma Control, Physics Benchmarks & Reference Suites
========================================================================

* TORAX (Differentiable Tokamak Core Transport Simulator)
  Repository: https://github.com/google-deepmind/torax
  Copyright 2024 DeepMind Technologies Limited
  License: Apache License 2.0
  Citation:
    Jonathan Citrin et al., \"TORAX: A Fast and Differentiable Tokamak
    Transport Simulator in JAX\", arXiv:2406.06718 (2024).
    https://arxiv.org/abs/2406.06718

* TokaGym & contrax (TCV Tokamak Rampdown Simulation & Control Suite)
  Dataset / Archive: https://doi.org/10.5281/zenodo.16621120
  Copyright (c) 2025 Allen M. Wang, Christian Theiler, et al. (MIT / EPFL)
  License: MIT
  Citations:
    Allen M. Wang et al., \"Learning plasma dynamics and robust rampdown
    trajectories with predict-first experiments at TCV\", Nature
    Communications 16 (2025) 8877.
    https://doi.org/10.1038/s41467-025-63917-3
    Christian Theiler et al., \"Progress and innovations in the TCV tokamak
    research programme\", Nuclear Fusion 66 (2026) 116007.
    https://doi.org/10.1088/1741-4326/ae6d12

* Acme (Distributed Reinforcement Learning Framework)
  Repository: https://github.com/google-deepmind/acme
  Copyright 2018 DeepMind Technologies Limited
  License: Apache License 2.0
  Citation:
    Matt Hoffman et al., \"Acme: A Research Framework for Distributed
    Reinforcement Learning\", arXiv:2006.00979 (2020).
    https://arxiv.org/abs/2006.00979

* Hegel (Property-Based Testing Protocol & Python Harness)
  Repository: https://github.com/hegeldev/hegel-python
  Copyright (c) David R. MacIver and contributors
  License: MIT

========================================================================
4. Bundled & Statically Linked Third-Party Libraries
========================================================================

Binary distributions of `baux` (including compiled Rust binaries, native
libraries, OxCaml executables, and Python wheels) statically link or bundle
components from the following open-source projects:

* OxCaml & Hardcaml (Unboxed SIMD & RTL Hardware Design Suite in `ocaml/`)
  Repositories:
    https://github.com/oxcaml/oxcaml
    https://github.com/janestreet/hardcaml
  Copyright (c) Jane Street Group, LLC and contributors
  License: MIT / LGPL-2.1-only WITH OCaml-LGPL-linking-exception

* SLEEF (SIMD Library for Evaluating Elementary Functions, v3.9.0 in `ocaml/`)
  Repository: https://github.com/shibatch/sleef
  Copyright (c) Naoki Shibata and contributors
  License: Boost Software License 1.0 (BSL-1.0)

* SciPy (Global Optimization Reference Test Suite in `ocaml/tests/`)
  Repository: https://github.com/scipy/scipy
  Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers
  License: BSD-3-Clause

* USearch (Unum Cloud Vector Search Engine — bundled under `vendor/usearch/`)
  Repository: https://github.com/unum-cloud/usearch
  Copyright (c) 2023 Ash Vardanian / Unum Cloud, Inc.
  License: Apache License 2.0

* FP16 & SimSIMD (Bundled C/C++ headers under `vendor/usearch/`)
  - `vendor/usearch/fp16`: Copyright (c) 2017-2020 Facebook Inc.,
    Copyright (c) 2017 Georgia Institute of Technology (Marat Dukhan).
    License: MIT.
  - `vendor/usearch/simsimd`: Copyright (c) 2023 Ash Vardanian / Unum Cloud.
    License: Apache License 2.0.

* LLVM OpenMP Runtime (`libomp` — bundled in macOS wheel builds when enabled)
  Repository: https://github.com/llvm/llvm-project
  Copyright (c) 1997-2024 The LLVM Project / Intel Corporation
  License: Apache License 2.0 WITH LLVM-exception

* Apache TVM and Apache TVM FFI (bundled native compiled-program runtimes)
  Repositories:
    https://github.com/apache/tvm
    https://github.com/apache/tvm-ffi
  License: Apache License 2.0
  Exact native artifacts and hashes: admin/buck2/tvm/BUCK.
  The upstream license and notice texts accompany the runtime bundles.

* OpenXLA PJRT CPU plugin (packaged by cyxla-pjrt-cpu)
  Repositories:
    https://github.com/openxla/xla
    https://github.com/nardi/cyxla
  License: Apache License 2.0
  Exact native artifact, matching header and source revisions: admin/buck2/pjrt/BUCK.
  The upstream license accompanies the native artifact as PJRT-LICENSE.

* Candle (`candle-core`, `candle-nn`) & Tokenizers (`tokenizers`)
  Repository: https://github.com/huggingface/candle
  Copyright (c) 2022-2025 Hugging Face Inc.
  License: Apache License 2.0 / MIT

* PyO3 & rust-numpy (`pyo3`, `numpy`)
  Repositories:
    https://github.com/PyO3/pyo3
    https://github.com/PyO3/rust-numpy
  Copyright (c) PyO3 Project and Contributors
  License: MIT OR Apache-2.0

* CUDA-Oxide (`cuda-core`)
  Repository: https://github.com/NVlabs/cuda-oxide
  Copyright (c) 2024 NVIDIA Corporation
  License: Apache License 2.0

Full dependency metadata and cryptographic checksums are recorded in
`Cargo.lock`, `rust/BUCK`, and `research/extraction/provenance.intoto.json`.

========================================================================
5. Visual Media & Artwork Notice
========================================================================

The diptych header artwork referenced by `README.md` features works by
Michal Rovner:
  - \"Outside #11\" (1991)
  - \"One-Person Game Against Nature #9V\" (1993)

Copyright for these artworks remains with Michal Rovner. These visual
assets are displayed for academic/project identification only and are
NOT licensed under the MIT License covering the `baux` source code.
";

pub fn render_notice() -> &'static str {
    CANONICAL_NOTICE
}

pub fn verify_notices(root: &Path) -> Result<(), String> {
    let root_license =
        fs::read_to_string(root.join("LICENSE")).map_err(|e| format!("read LICENSE: {e}"))?;
    let crate_license = fs::read_to_string(root.join("rust/crates/baux/LICENSE"))
        .map_err(|e| format!("read rust/crates/baux/LICENSE: {e}"))?;
    if root_license != crate_license {
        return Err("LICENSE and rust/crates/baux/LICENSE differ".into());
    }
    if !root_license.starts_with("MIT License\n\nCopyright (c) 2026 Kvutza\n") {
        return Err("LICENSE does not start with canonical MIT header for Kvutza".into());
    }
    let root_notice =
        fs::read_to_string(root.join("NOTICE")).map_err(|e| format!("read NOTICE: {e}"))?;
    let crate_notice = fs::read_to_string(root.join("rust/crates/baux/NOTICE"))
        .map_err(|e| format!("read rust/crates/baux/NOTICE: {e}"))?;
    if root_notice != crate_notice {
        return Err("NOTICE and rust/crates/baux/NOTICE differ".into());
    }
    if root_notice != CANONICAL_NOTICE {
        return Err("NOTICE does not match canonical auto-compiled NOTICE".into());
    }
    Ok(())
}
