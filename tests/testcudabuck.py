import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundle():
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
    build = (ROOT / "BUCK").read_text()
    for member in workspace["workspace"]["members"]:
        if member.startswith("rust/crates/"):
            name = Path(member).name
            assert f'"{name}": "//{member}:' in build
    rule = (ROOT / "build/buck2/cuda/defs.bzl").read_text()
    assert "srcs = sources" in rule
    assert "for crate, label in crates.items()" in rule


def test_atomic():
    source = (ROOT / "build/buck2/cuda/defs.bzl").read_text(encoding="utf-8")
    lockfile = "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;"
    build = (
        "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target "
        "--device-codegen-crate baux_cuda_device -- -p baux-py -p baux-cuda "
        "-p baux-runner -p baux-corpus "
        "--features cuda --release --locked --manifest-path $ROOT/Cargo.toml;"
    )

    assert lockfile in source
    assert build in source
    assert source.index(lockfile) < source.index(build)
    assert '"cargo oxide build",' not in source
    assert '"--arch sm_75",' not in source
    assert "; --arch sm_75" not in " ".join(source.split())


def test_abi():
    source = (ROOT / "build/buck2/cuda/defs.bzl").read_text()
    assert '"sdk": "toolchains//:sdk-linux-x86_64"' in source
    assert "export RUSTFLAGS=" in source
    assert "export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS=" not in source
    assert "--sysroot=$SYSROOT" in source
    assert "-fuse-ld=bfd" in source
    assert source.index("export RUSTFLAGS=") < source.index("cargo oxide build")


def test_storage():
    source = (ROOT / "ops/modal_build.py").read_text()
    assert '(WORK / "buck-out").symlink_to' not in source
    assert '(WORK / ".buck2-tools").symlink_to' in source
    assert 'Path("ops/modal_build.py")' in source
    assert source.index('.apt_install("patchelf")') < source.index(".add_local_dir(")


def test_validation():
    source = (ROOT / "ops/modal_build.py").read_text()
    assert '"--no-deps"' not in source
    assert '"pip", "check"' in source
    assert '"assert row.available, row.reason"' in source
    assert '"auditwheel"' in source
    assert '"manylinux_2_28_x86_64"' in source
