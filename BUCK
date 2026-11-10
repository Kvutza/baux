load(
    "//admin/buck2:python.bzl",
    "PYTHON_ABI",
    "PYTHON_REQUIRES",
    "python_extension_suffix",
)
load("//admin/buck2/cuda:defs.bzl", "cuda_oxide")
load("//admin/buck2/wheel:wheel.bzl", "python_wheel")

http_file(
    name = "click-wheel",
    urls = ["https://files.pythonhosted.org/packages/58/50/6c0d534c5f134586a8e1ba4e330569e32f057e33372ae556463212fb4cd3/click-8.5.0-py3-none-any.whl"],
    sha256 = "255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360",
    out = "click-8.5.0-py3-none-any.whl",
)

constraint_setting(name = "profile")

constraint_value(
    name = "release-mode",
    constraint_setting = ":profile",
    visibility = ["PUBLIC"],
)

platform(
    name = "macos-arm64-platform",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
    visibility = ["PUBLIC"],
)

config_setting(
    name = "linux-arm64",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
)

config_setting(
    name = "linux-x86_64",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
)

platform(
    name = "linux-arm64-platform",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
    visibility = ["PUBLIC"],
)

platform(
    name = "linux-x86_64-platform",
    constraint_values = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
    visibility = ["PUBLIC"],
)

config_setting(
    name = "macos-arm64",
    constraint_values = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
)

platform(
    name = "release-linux-arm64-platform",
    deps = [":linux-arm64-platform"],
    constraint_values = [":release-mode"],
)

platform(
    name = "release-linux-x86_64-platform",
    deps = [":linux-x86_64-platform"],
    constraint_values = [":release-mode"],
)

platform(
    name = "release-macos-arm64-platform",
    deps = [":macos-arm64-platform"],
    constraint_values = [":release-mode"],
)

filegroup(
    name = "cargo-workspace",
    srcs = ["Cargo.toml"],
    visibility = ["PUBLIC"],
)

filegroup(
    name = "optimizer-python-source",
    srcs = glob([
        "python/baux/**/*.py",
        "tests/python_api/*.py",
        "tests/interop/*.py",
        "evals/torax/*.py",
        "evals/tokagym/*.py",
        "evals/geometry/*.py",
        "research/upstream/*.lock.toml",
        "rust/crates/baux/src/muon/*.toml",
        "rust/crates/baux/src/hnet/*.toml",
        "rust/crates/baux/src/trust_region/*.toml",
    ]),
    visibility = ["//admin/buck2/tests:optimizer-python", "//admin/buck2/tests:resident-python"],
)

filegroup(
    name = "compiler-source",
    srcs = glob(["python/baux/**/*.py", "tests/compiler/*.py"]),
    visibility = ["//admin/buck2/tests:compiler-python"],
)

filegroup(
    name = "cuda-package",
    srcs = {
        path: path
        for path in glob(["python/baux/**/*.py", "data/recipes/*.toml"]) + [
            "LICENSE",
            "NOTICE",
            "README.md",
            "rust-toolchain.toml",
        ]
    } | {
        "admin/ops/cuda_wheel.py": "//admin:ops/cuda_wheel.py",
    },
    visibility = ["PUBLIC"],
)

cuda_oxide(
    name = "cuda",
    package = ":cuda-package",
    workspace = ":cargo-workspace",
    crates = {
        "corpus": "//rust/crates/corpus:source",
        "cli": "//rust/crates/cli:source",
        "runner": "//rust/crates/runner:source",
        "tooling": "//rust/crates/tooling:source",
        "baux": "//rust/crates/baux:baux-source",
        "baux-py": "//rust/crates/baux-py:python-source",
        "fbt": "//rust/crates/fbt:source",
        "modal-runner": "//rust/crates/modal-runner:modal-source",
        "properties": "//rust/crates/properties:source",
    },
    cuda = "//rust/crates/cuda:source",
    parity = "//admin:ops/bf16_parity.py",
    click = ":click-wheel",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
)

python_wheel(
    name = "wheel-linux-arm64",
    extension = "//rust/crates/baux-py:baux-py[shared]",
    extension_suffix = python_extension_suffix("linux-aarch64"),
    fixture_srcs = glob(["tests/fixtures/optimizer/*.toml", "tests/fixtures/morbo/*.toml"]),
    license = "LICENSE",
    notice = "NOTICE",
    package = "baux",
    platform_tag = "manylinux_2_28_aarch64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["python/baux/**/*.py"]),
    readme = "README.md",
    runtime_libraries = [],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:linux",
    ],
    version = read_config("baux", "release_version", "0.0.5"),
)

python_wheel(
    name = "wheel-linux-x86_64",
    extension = "//rust/crates/baux-py:baux-py[shared]",
    extension_suffix = python_extension_suffix("linux-x86_64"),
    fixture_srcs = glob(["tests/fixtures/optimizer/*.toml", "tests/fixtures/morbo/*.toml"]),
    license = "LICENSE",
    notice = "NOTICE",
    package = "baux",
    platform_tag = "manylinux_2_28_x86_64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["python/baux/**/*.py"]),
    readme = "README.md",
    runtime_libraries = [],
    target_compatible_with = [
        "prelude//cpu/constraints:x86_64",
        "prelude//os/constraints:linux",
    ],
    version = read_config("baux", "release_version", "0.0.5"),
)

python_wheel(
    name = "wheel-macos-arm64",
    extension = "//rust/crates/baux-py:baux-py[shared]",
    extension_suffix = python_extension_suffix("macos-arm64"),
    fixture_srcs = glob(["tests/fixtures/optimizer/*.toml", "tests/fixtures/morbo/*.toml"]),
    license = "LICENSE",
    notice = "NOTICE",
    package = "baux",
    platform_tag = "macosx_11_0_arm64",
    python_abi = PYTHON_ABI,
    python_requires = PYTHON_REQUIRES,
    python_srcs = glob(["python/baux/**/*.py"]),
    readme = "README.md",
    runtime_libraries = [
        "//admin/buck2/native:openmp-build[lib]",
        "//admin/buck2/iree:build[library]",
    ],
    target_compatible_with = [
        "prelude//cpu/constraints:arm64",
        "prelude//os/constraints:macos",
    ],
    version = read_config("baux", "release_version", "0.0.5"),
)

configured_alias(
    name = "wheel-linux-arm64-release",
    actual = ":wheel-linux-arm64",
    platform = ":release-linux-arm64-platform",
)

configured_alias(
    name = "wheel-linux-x86_64-release",
    actual = ":wheel-linux-x86_64",
    platform = ":release-linux-x86_64-platform",
)

configured_alias(
    name = "wheel-macos-arm64-release",
    actual = ":wheel-macos-arm64",
    platform = ":release-macos-arm64-platform",
)

alias(
    name = "wheel",
    actual = select({
        ":linux-arm64": ":wheel-linux-arm64-release",
        ":linux-x86_64": ":wheel-linux-x86_64-release",
        ":macos-arm64": ":wheel-macos-arm64-release",
    }),
    visibility = ["PUBLIC"],
)
