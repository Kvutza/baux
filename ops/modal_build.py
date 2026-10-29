from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/opt/baux")
WORK = Path("/work")
CACHE = Path("/cache")
ARTIFACTS = Path("/artifacts")
IGNORED = {
    ".beads",
    ".buck2-tools",
    ".cache",
    ".git",
    ".jj",
    ".lake",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "archive",
    "buck-out",
    "dist",
    "docs",
    "notebooks",
    "performance",
    "research",
    "results",
    "target",
}
WHEEL_INPUTS = (
    Path(".buckconfig"),
    Path("BUCK"),
    Path("Cargo.toml"),
    Path("LICENSE"),
    Path("NOTICE"),
    Path("README.md"),
    Path("backends/cuda"),
    Path("build/buck2/cuda"),
    Path("build/buck2/python.bzl"),
    Path("build/targets.bzl"),
    Path("build/toolchains"),
    Path("data/recipes"),
    Path("ops/cuda_wheel.py"),
    Path("ops/modal_build.py"),
    Path("rust/crates/baux"),
    Path("rust/crates/baux-py"),
    Path("rust/crates/cli"),
    Path("rust/crates/corpus"),
    Path("rust/crates/modal-runner"),
    Path("rust/crates/properties"),
    Path("rust/crates/ptx-synth"),
    Path("rust/crates/runner"),
    Path("rust/crates/tooling"),
    Path("rust-toolchain.toml"),
    Path("src/baux"),
)

app = modal.App("baux-cuda")
build_cache = modal.Volume.from_name("baux-build", create_if_missing=True)
artifacts = modal.Volume.from_name("baux-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
        add_python="3.14",
    )
    .apt_install(
        "ca-certificates",
        "curl",
        "g++",
        "gcc",
        "git",
        "gnupg",
        "libc6-dev",
        "make",
        "pkg-config",
    )
    .run_commands(
        "curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key "
        "| gpg --dearmor -o /usr/share/keyrings/apt.llvm.org.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] "
        "https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' "
        "> /etc/apt/sources.list.d/llvm.list",
        "apt-get update && apt-get install -y --no-install-recommends "
        "clang-21 libclang-common-21-dev lld-21 llvm-21 llvm-21-dev",
        "curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain none",
        "/root/.cargo/bin/rustup toolchain install nightly-2026-08-28 "
        "--profile minimal --component rust-src --component rustc-dev "
        "--component llvm-tools",
        "curl -fsSL https://github.com/NVlabs/cuda-oxide/archive/"
        "6abfaa091e29a6275c1943895bfbc97efa306e98.tar.gz "
        "| tar -xz --strip-components=1 -C /tmp",
        "/root/.cargo/bin/cargo +nightly-2026-08-28 install "
        "--path /tmp/crates/cargo-oxide --locked",
    )
    # JAX CUDA arrays reach the resident posterior through DLPack.  Match the
    # CUDA 13 container; using the CUDA 12 extra would test a different runtime
    # from the wheel's target environment.
    .uv_pip_install("jax[cuda13]", "torch")
    .uv_pip_install(
        "auditwheel",
        "botorch>=0.18.1,<0.19",
        "ax-platform>=1.3.1,<1.4",
        "certifi>=2026.10.19",
        "optuna>=5.0,<5.1",
        "pip>=26.2.2",
        "pytest",
    )
    .apt_install("patchelf")
    .add_local_dir(
        ROOT,
        REMOTE,
        copy=True,
        ignore=sorted(IGNORED),
    )
)


def run(command: list[str], *, cwd: Path = WORK) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _removepath(path: Path) -> None:
    if not (path.is_symlink() or path.exists()):
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def prepare() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.copytree(REMOTE, WORK, symlinks=True)
    (CACHE / "tools").mkdir(parents=True, exist_ok=True)
    cargo = CACHE / "home" / ".cargo"
    cargo.mkdir(parents=True, exist_ok=True)
    if not (cargo / "bin" / "cargo-oxide").exists():
        shutil.copytree("/root/.cargo/bin", cargo / "bin", dirs_exist_ok=True)
    for name in ("buck-out", ".buck2-tools"):
        _removepath(WORK / name)
    # Mutable compiler/linker outputs stay on local disk. Persist downloads and
    # completed wheel artifacts, not the linker's active output filesystem.
    (WORK / ".buck2-tools").symlink_to(CACHE / "tools", target_is_directory=True)


def environment() -> dict[str, str]:
    values = os.environ.copy()
    values.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "CUDA_TOOLKIT_PATH": "/usr/local/cuda",
            "CUDA_OXIDE_LLC": "/usr/bin/llc-21",
            "LIBCLANG_PATH": "/usr/lib/llvm-21/lib",
            "LLVM_CONFIG_PATH": "/usr/bin/llvm-config-21",
            "RUSTUP_HOME": "/root/.rustup",
            "HOME": "/cache/home",
            "CARGO_HOME": "/cache/home/.cargo",
            "CARGO_PROFILE_RELEASE_LTO": "false",
            "PATH": "/cache/home/.cargo/bin:/usr/lib/llvm-21/bin:/usr/local/cuda/bin:"
            + values["PATH"],
        }
    )
    return values


@app.function(
    image=image,
    cpu=8,
    memory=16_384,
    timeout=3_600,
    volumes={CACHE: build_cache, ARTIFACTS: artifacts},
)
def build(source: str) -> str:
    artifacts.reload()
    directory = ARTIFACTS / source
    existing = sorted(directory.glob("*.whl")) if directory.is_dir() else []
    if len(existing) == 1:
        return existing[0].relative_to(ARTIFACTS).as_posix()
    prepare()
    output = subprocess.check_output(
        [
            "./tools/buck2w",
            "--isolation-dir",
            "cuda",
            "build",
            "//:cuda-wheel",
            "--target-platforms",
            "//:linux-x86_64-platform",
            "--local-only",
            "--num-threads",
            "8",
            "--show-full-simple-output",
        ],
        cwd=WORK,
        env=environment(),
        text=True,
    ).strip()
    wheel = Path(output.splitlines()[-1])
    if not wheel.is_file():
        raise RuntimeError(f"wheel output does not exist: {wheel}")
    repaired = WORK / "repaired"
    subprocess.run(
        [
            "auditwheel",
            "repair",
            "--plat",
            "manylinux_2_28_x86_64",
            "--only-plat",
            "--exclude",
            "libcuda.so.1",
            "--wheel-dir",
            str(repaired),
            str(wheel),
        ],
        cwd=WORK,
        check=True,
    )
    wheels = sorted(repaired.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one audited CUDA wheel")
    wheel = wheels[0]
    destination = ARTIFACTS / source / wheel.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, destination)
    build_cache.commit()
    artifacts.commit()
    return destination.relative_to(ARTIFACTS).as_posix()


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16_384,
    timeout=1_200,
    volumes={ARTIFACTS: artifacts},
)
def verify(artifact: str) -> str:
    artifacts.reload()
    wheel = ARTIFACTS / artifact
    if not wheel.is_file():
        raise RuntimeError(f"wheel artifact does not exist: {artifact}")
    site = Path(sysconfig.get_path("platlib"))
    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--quiet",
            str(wheel),
        ],
        cwd=REMOTE,
    )
    run(["python", "-m", "pip", "check"], cwd=REMOTE)
    env = environment()
    env["PYTHONPATH"] = str(site)
    env["BAUX_TEST_WHEEL_ROOT"] = str(site)
    subprocess.run(
        [
            "python",
            "-c",
            (
                "from baux import capabilities; "
                "row = next(row for row in capabilities(probe=True) if row.name == 'cuda'); "
                "print(row, flush=True); "
                "assert row.available, row.reason"
            ),
        ],
        cwd=REMOTE,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "tests/interop/test_resident.py",
            "tests/interop/test_botorch.py",
            "tests/interop/test_ax.py",
            "tests/test_runtime.py",
        ],
        cwd=REMOTE,
        env=env,
        check=True,
    )
    subprocess.run(["python", "ops/bf16_parity.py"], cwd=REMOTE, env=env, check=True)
    subprocess.run(["python", "ops/cusprbench.py"], cwd=REMOTE, env=env, check=True)
    subprocess.run(["python", "ops/bf16_bench.py"], cwd=REMOTE, env=env, check=True)
    return artifact


def digest() -> str:
    result = hashlib.sha256()
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for name in files:
        relative = Path(name)
        if any(part in IGNORED for part in relative.parts):
            continue
        if not any(
            relative == item or item in relative.parents for item in WHEEL_INPUTS
        ):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        result.update(relative.as_posix().encode())
        result.update(path.read_bytes())
    return result.hexdigest()


@app.local_entrypoint()
def wheel(output_directory: str = "dist") -> None:
    artifact = build.remote(digest())
    verified = verify.remote(artifact)
    destination = Path(output_directory) / Path(verified).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        for chunk in artifacts.read_file(verified):
            stream.write(chunk)
    print(f"CUDA_WHEEL ok=true output={destination}")
