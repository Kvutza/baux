"""Generated from Cargo metadata by tools/build-parity --sync. Do not edit."""

BAUX_CRATES = ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "redb", "sha2", "sobol", "thiserror", "tokenizers"]

BAUX_TEST_CRATES = ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"]

BAUX_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "baux-tune",
        "crate": "baux_tune",
        "root": "src/bin/baux-tune/main.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "redb", "sha2", "sobol", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": [],
    },
    {
        "version": "0.0.1",
        "name": "baux-tune-worker",
        "crate": "baux_tune_worker",
        "root": "src/bin/baux-tune-worker/main.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "rand", "rand_distr", "rayon", "redb", "sha2", "sobol", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": [],
    },
    {
        "version": "0.0.1",
        "name": "cuda_parity",
        "crate": "cuda_parity",
        "root": "examples/cuda_parity.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": ["cuda"],
    },
    {
        "version": "0.0.1",
        "name": "enn_posterior",
        "crate": "enn_posterior",
        "root": "examples/enn_posterior.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": [],
    },
    {
        "version": "0.0.1",
        "name": "knn_bench",
        "crate": "knn_bench",
        "root": "examples/knn_bench.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": ["metal"],
    },
    {
        "version": "0.0.1",
        "name": "posterior_bench",
        "crate": "posterior_bench",
        "root": "examples/posterior_bench.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": [],
    },
    {
        "version": "0.0.1",
        "name": "trial_bench",
        "crate": "trial_bench",
        "root": "examples/trial_bench.rs",
        "crate_deps": ["anstream", "anstyle", "charming", "deser", "deser-cbor", "deser-json", "deser-toml", "deser-value", "jiff", "kuva", "memmap2", "ndarray", "numpy", "rand", "rand_chacha", "rand_distr", "rayon", "redb", "sha2", "sobol", "tempfile", "thiserror", "tokenizers"],
        "local_deps": ["baux"],
        "required_features": [],
    },
]

CLI_CRATES = ["clap"]

CLI_TEST_CRATES = ["clap"]

CLI_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "baux",
        "crate": "baux",
        "root": "src/bin/baux/main.rs",
        "crate_deps": ["clap"],
        "local_deps": ["baux", "tooling"],
        "required_features": [],
    },
]

CORPUS_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"]

CORPUS_TEST_CRATES = ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"]

CORPUS_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "baux-corpus",
        "crate": "baux_corpus",
        "root": "src/main.rs",
        "crate_deps": ["arrow-array", "deser", "futures", "opendal", "parquet", "parquet_opendal", "sha2", "tokenizers", "tokio"],
        "local_deps": ["baux"],
        "required_features": [],
    },
]

MODAL_RUNNER_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_TEST_CRATES = ["modal-rs", "tempfile", "tokio"]

MODAL_RUNNER_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "baux-modal",
        "crate": "baux_modal",
        "root": "src/main.rs",
        "crate_deps": ["modal-rs", "tempfile", "tokio"],
        "local_deps": ["baux"],
        "required_features": [],
    },
]

PROPERTIES_CRATES = ["hegeltest", "ndarray"]

PROPERTIES_TEST_CRATES = ["hegeltest", "ndarray"]

PROPERTIES_PROGRAMS = [
]

BAUX_PY_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

BAUX_PY_TEST_CRATES = ["ndarray", "numpy", "pyo3", "rand"]

BAUX_PY_PROGRAMS = [
]

RUNNER_CRATES = ["clap", "sha2"]

RUNNER_TEST_CRATES = ["clap", "sha2"]

RUNNER_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "baux-runner",
        "crate": "baux_runner",
        "root": "src/main.rs",
        "crate_deps": ["clap", "sha2"],
        "local_deps": ["baux", "tooling"],
        "required_features": [],
    },
]

TOOLING_CRATES = ["clap", "deser"]

TOOLING_TEST_CRATES = ["clap", "deser"]

TOOLING_PROGRAMS = [
    {
        "version": "0.0.1",
        "name": "build-graph",
        "crate": "build_graph",
        "root": "src/bin/build-graph.rs",
        "crate_deps": ["clap", "deser"],
        "local_deps": ["baux", "tooling"],
        "required_features": [],
    },
]

PTX_SYNTH_CRATES = []

PTX_SYNTH_TEST_CRATES = []

PTX_SYNTH_PROGRAMS = [
]

RUST_EDITION = "2024"
