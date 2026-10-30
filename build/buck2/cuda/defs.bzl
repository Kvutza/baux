def cuda_oxide(
        name,
        package,
        workspace,
        crates,
        cuda,
        parity,
        click,
        python_abi,
        python_requires):
    """Build a CUDA-Oxide CPython wheel and its GPU parity action."""
    wheel = name + "-wheel"
    release_version = read_config("baux", "release_version", "0.1.1")
    filename = "baux-{}+cuda75-{}-{}-manylinux_2_28_x86_64.whl".format(
        release_version,
        python_abi,
        python_abi,
    )
    python_version = "{}.{}".format(python_abi[2], python_abi[3:])
    sources = {
        "click": click,
        "cuda": cuda,
        "package": package,
        "workspace": workspace,
        "sdk": "toolchains//:sdk-linux-x86_64",
    }
    for crate, label in crates.items():
        sources[crate] = label
    native.genrule(
        name = wheel,
        srcs = sources,
        outs = {
            "wheel": [filename],
            "cargo-target": ["target"],
            "spine": ["target/release/baux-cuda"],
            "runner": ["target/release/baux-runner"],
            "corpus": ["target/release/baux-corpus"],
        },
        default_outs = [filename],
        # Cargo owns this incremental cache; Buck keeps it between local runs.
        no_outputs_cleanup = True,
        cacheable = False,
        labels = ["uses_undeclared_inputs"],
        # CUDA-Oxide pins this compiler; the host workspace uses a newer nightly.
        # Keep CUDA compilation on the toolchain recorded by backends/cuda/rust-toolchain.toml.
        env = {"RUSTUP_TOOLCHAIN": "nightly-2026-08-28"},
        cmd = " ".join([
            "set -euo pipefail;",
            "export CUDA_HOME=/usr/local/cuda-13.0",
            "CUDA_PATH=/usr/local/cuda-13.0",
            "CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0;",
            "export CUDA_OXIDE_LLC=/usr/bin/llc-21",
            "LIBCLANG_PATH=/usr/lib/llvm-21/lib",
            "LLVM_CONFIG_PATH=/usr/bin/llvm-config-21;",
            "export CARGO_PROFILE_RELEASE_LTO=false;",
            "export PATH=$HOME/.cargo/bin:/usr/lib/llvm-21/bin:/usr/local/cuda-13.0/bin:$PATH;",
            "python3 --version | grep -q '^Python {}\\.';".format(python_version),
            "OUT_DIR=$PWD/$OUT;",
            "OUT_FILE=$OUT_DIR/{};".format(filename),
            "CLICK=$PWD/$(location {});".format(click),
            "SDK=$PWD/$(location toolchains//:sdk-linux-x86_64);",
            "SYSROOT=$SDK/x86_64-conda-linux-gnu/sysroot;",
            "GCC_INSTALL=$SDK/lib/gcc/x86_64-conda-linux-gnu/8.5.0;",
            "export CC=/usr/bin/clang-21 CXX=/usr/bin/clang++-21;",
            "export CFLAGS=\"--no-default-config --sysroot=$SYSROOT --gcc-install-dir=$GCC_INSTALL\";",
            "export CXXFLAGS=\"$CFLAGS\";",
            # cargo-oxide preserves RUSTFLAGS inside CARGO_ENCODED_RUSTFLAGS;
            # target-specific rustflags are shadowed by that encoded variable.
            "export RUSTFLAGS=\"-C linker=/usr/bin/clang++-21",
            "-C link-arg=--no-default-config",
            "-C link-arg=--sysroot=$SYSROOT",
            "-C link-arg=--gcc-install-dir=$GCC_INSTALL",
            "-C link-arg=-fuse-ld=bfd",
            "-C link-arg=-L$SDK/lib\";",
            "ROOT=$TMP/repo;",
            "mkdir -p $ROOT/rust/crates $ROOT/backends/cuda;",
            "cp -R $(location {})/. $ROOT/;".format(package),
            "cp -R $(location {})/. $ROOT/;".format(workspace),
        ] + [
            "cp -R $(location {}) $ROOT/rust/crates/{};".format(label, crate)
            for crate, label in crates.items()
        ] + [
            "cp -R $(location {})/. $ROOT/backends/cuda/;".format(cuda),
            "python3 -c '",
            "p=\"$ROOT/rust/crates/baux/Cargo.toml\";",
            "t=open(p).read()",
            ".replace(\"cuda-core = { workspace = true, optional = true }\",",
            "\"baux-cuda = { path = \\\"../../../backends/cuda\\\", optional = true }\\n\" +",
            "\"cuda-core = { workspace = true, optional = true }\")",
            ".replace(\"cuda = [\\\"dep:cuda-core\\\"]\",",
            "\"cuda = [\\\"dep:baux-cuda\\\", \\\"dep:cuda-core\\\"]\")",
            ";",
            "open(p,\"w\").write(t)';",
            "cd $ROOT/backends/cuda;",
            "cargo generate-lockfile --manifest-path $ROOT/Cargo.toml;",
            "cargo oxide build --arch sm_75 --cargo-target-dir $OUT_DIR/target",
            "--device-codegen-crate baux_cuda_device --",
            "-p baux-py -p baux-cuda -p baux-runner -p baux-corpus",
            "--features cuda --release --locked --manifest-path $ROOT/Cargo.toml;",
            "cd $ROOT;",
            "find rust/crates/baux/src rust/crates/baux/semantics/src",
            "rust/crates/baux/neighbor-index/src rust/crates/baux/artifact/src",
            "backends/cuda/src backends/cuda/device/src",
            "rust/crates/modal-runner/src rust/crates/runner/src/cuda_tune.rs",
            "-type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 |",
            "cut -d' ' -f1 > $TMP/source.sha256;",
            "PYTHONPATH=$CLICK python3 ops/cuda_wheel.py",
            "$OUT_DIR/target/release/libbaux_rust.so",
            "$OUT_DIR/target/release/baux-runner",
            "$OUT_DIR/target/release/baux-corpus",
            "$OUT_DIR/target/release/baux-cuda",
            "$TMP/source.sha256",
            "$TMP/wheel --root $ROOT --version {} --python-abi {} --python-requires '{}';".format(
                release_version,
                python_abi,
                python_requires,
            ),
            "mkdir -p ${OUT_FILE%/*};",
            "cp $TMP/wheel/{} $OUT_FILE".format(filename),
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
    native.genrule(
        name = name + "-parity",
        srcs = [parity],
        out = "bf16-parity.txt",
        cmd = " ".join([
            "set -euo pipefail;",
            "python3 -m pip install --quiet --no-deps",
            "--target $TMP/site $(location :{});".format(wheel),
            "PYTHONPATH=$TMP/site python3 $SRCDIR/ops/bf16_parity.py > $OUT",
        ]),
        target_compatible_with = [
            "prelude//cpu/constraints:x86_64",
            "prelude//os/constraints:linux",
        ],
        visibility = ["PUBLIC"],
    )
