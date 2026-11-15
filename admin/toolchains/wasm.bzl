load(
    "@prelude//cxx:cxx_toolchain_types.bzl",
    "BinaryUtilitiesInfo",
    "CCompilerInfo",
    "CvtresCompilerInfo",
    "CxxCompilerInfo",
    "CxxInternalTools",
    "CxxPlatformInfo",
    "CxxToolchainInfo",
    "DepTrackingMode",
    "LinkerInfo",
    "LinkerType",
    "PicBehavior",
    "RcCompilerInfo",
    "RuntimeDependencyHandling",
    "ShlibInterfacesMode",
)
load("@prelude//cxx:headers.bzl", "HeaderMode")
load("@prelude//linking:link_info.bzl", "LinkStyle")
load("@prelude//linking:lto.bzl", "LtoMode")
load("@prelude//rust:rust_toolchain.bzl", "PanicRuntime", "RustToolchainInfo")
load(":rust_pins.bzl", "RUST_DATE", "RUST_EDITION")

_EMSCRIPTEN_REV = "a4d4afb626c5010f6ccda4638b8d77579a63782e"

_EMSCRIPTEN_ARCHIVES = {
    "macos-arm64": (
        "https://storage.googleapis.com/webassembly/emscripten-releases-builds/mac/{}/wasm-binaries-arm64.tar.xz".format(_EMSCRIPTEN_REV),
        "7a9a15845257629b7602d15bdf7633a8e10472b0fa9b3d9ee7149938aa2f2039",
    ),
    "macos-x86_64": (
        "https://storage.googleapis.com/webassembly/emscripten-releases-builds/mac/{}/wasm-binaries.tar.xz".format(_EMSCRIPTEN_REV),
        "2092aa4bef3b9f88d3f343b042a417ba617d4e04454656d8f2e101ba53f854e8",
    ),
    "linux-x86_64": (
        "https://storage.googleapis.com/webassembly/emscripten-releases-builds/linux/{}/wasm-binaries.tar.xz".format(_EMSCRIPTEN_REV),
        "b188249ecb939dadc679aaf2d3d9afd0fe19ab942f91b7bc926b4f252915dd1a",
    ),
    "linux-arm64": (
        "https://storage.googleapis.com/webassembly/emscripten-releases-builds/linux/{}/wasm-binaries-arm64.tar.xz".format(_EMSCRIPTEN_REV),
        "4aedc8ca641b40d9bd82d85b1dc3458fe1afc9a132da06a09384a5f89c058969",
    ),
}

def wasm_archives():
    native.http_archive(
        name = "rust-std-nightly-wasm32-unknown-emscripten",
        urls = ["https://static.rust-lang.org/dist/{}/rust-std-nightly-wasm32-unknown-emscripten.tar.xz".format(RUST_DATE)],
        sha256 = "919d66dfec3b023f85be716dd169e81bfb48634a9b5cbdac91003754fad1990b",
        strip_prefix = "rust-std-nightly-wasm32-unknown-emscripten/rust-std-wasm32-unknown-emscripten",
        visibility = ["PUBLIC"],
    )
    native.http_archive(
        name = "pyodide-xbuildenv",
        urls = ["https://github.com/pyodide/pyodide/releases/download/0.27.5/xbuildenv-0.27.5.tar.bz2"],
        sha256 = "0953d4859ee5116e83aaf5d0c673efe49670c750fe8dfb29b8cca5f15318b5af",
        visibility = ["PUBLIC"],
    )
    for platform_name, (url, sha256) in _EMSCRIPTEN_ARCHIVES.items():
        native.http_archive(
            name = "emscripten-sdk-" + platform_name,
            urls = [url],
            sha256 = sha256,
            visibility = ["PUBLIC"],
        )
    native.alias(
        name = "emscripten-sdk",
        actual = select({
            "prelude//os:macos": select({"prelude//cpu:arm64": ":emscripten-sdk-macos-arm64", "prelude//cpu:x86_64": ":emscripten-sdk-macos-x86_64"}),
            "prelude//os:linux": select({"prelude//cpu:arm64": ":emscripten-sdk-linux-arm64", "prelude//cpu:x86_64": ":emscripten-sdk-linux-x86_64"}),
        }),
        visibility = ["PUBLIC"],
    )

def _pyodide_python_include_impl(ctx: AnalysisContext):
    xbuildenv = ctx.attrs.xbuildenv[DefaultInfo].default_outputs[0]
    out = ctx.actions.declare_output("include", dir = True)
    copy = ctx.actions.write(
        "copy_include.sh",
        '#!/bin/sh\nset -eu\nmkdir -p "$2"\ncp -R "$1/xbuildenv/pyodide-root/cpython/installs/python-3.12.7/include/python3.12/." "$2"/\n',
        is_executable = True,
    )
    ctx.actions.run([copy, xbuildenv, out.as_output()], category = "pyodide_include")
    return [DefaultInfo(default_output = out)]

_pyodide_python_include = rule(
    impl = _pyodide_python_include_impl,
    attrs = {
        "xbuildenv": attrs.dep(),
    },
)

def _wasm_rust_sysroot_impl(ctx: AnalysisContext):
    host_sysroot = ctx.attrs.host_sysroot[DefaultInfo].default_outputs[0]
    wasm_std = ctx.attrs.wasm_std[DefaultInfo].default_outputs[0]
    sysroot = ctx.actions.declare_output("sysroot", dir = True)
    merge = ctx.actions.write(
        "merge_wasm_sysroot.sh",
        '#!/bin/sh\nset -eu\nmkdir -p "$3"\ncp -R "$1"/. "$3"/\ncp -R "$2"/. "$3"/\n',
        is_executable = True,
    )
    ctx.actions.run([merge, host_sysroot, wasm_std, sysroot.as_output()], category = "wasm_rust_sysroot")
    return [DefaultInfo(default_output = sysroot)]

_wasm_rust_sysroot = rule(
    impl = _wasm_rust_sysroot_impl,
    attrs = {
        "host_sysroot": attrs.dep(),
        "wasm_std": attrs.dep(),
    },
)

def _wasm_rust_toolchain_impl(ctx: AnalysisContext):
    sysroot = ctx.attrs.sysroot[DefaultInfo].default_outputs[0]
    return [
        DefaultInfo(default_output = sysroot),
        RustToolchainInfo(
            compiler = RunInfo(args = [sysroot.project("bin/rustc")]),
            rustdoc = RunInfo(args = [sysroot.project("bin/rustdoc")]),
            clippy_driver = RunInfo(args = [sysroot.project("bin/clippy-driver")]),
            sysroot_path = sysroot,
            rustc_target_triple = "wasm32-unknown-emscripten",
            default_edition = RUST_EDITION,
            panic_runtime = PanicRuntime("abort"),
            nightly_features = False,
            rustc_flags = [
                "-Copt-level=3",
                "-Cdebug-assertions=off",
                "-Coverflow-checks=off",
                "-Cpanic=abort",
                "-Ctarget-feature=+simd128",
                "-Cstrip=debuginfo",
                "-Zremap-cwd-prefix=.",
            ],
            doctests = False,
        ),
    ]

_wasm_rust_toolchain = rule(
    impl = _wasm_rust_toolchain_impl,
    attrs = {
        "sysroot": attrs.exec_dep(),
    },
    is_toolchain_rule = True,
)

def _wasm_cxx_toolchain_impl(ctx: AnalysisContext):
    sdk = ctx.attrs.sdk[DefaultInfo].default_outputs[0]
    python = ctx.attrs.python[DefaultInfo].default_outputs[0]
    py_bin = python.project("bin/python3")
    wrapper = ctx.attrs.wrapper

    def _wrap(mode: str) -> RunInfo:
        return RunInfo(args = cmd_args(py_bin, wrapper, sdk, mode, hidden = [python, sdk]))

    cc = _wrap("cc")
    cxx = _wrap("cxx")
    return [
        DefaultInfo(),
        CxxToolchainInfo(
            internal_tools = ctx.attrs._cxx_internal_tools[CxxInternalTools],
            linker_info = LinkerInfo(
                linker = _wrap("ld"),
                linker_flags = [],
                post_linker_flags = [],
                archiver = _wrap("ar"),
                archiver_type = "llvm",
                archiver_supports_argfiles = True,
                generate_linker_maps = False,
                lto_mode = LtoMode("none"),
                type = LinkerType("wasm"),
                link_binaries_locally = True,
                link_libraries_locally = True,
                archive_objects_locally = True,
                use_archiver_flags = True,
                static_dep_runtime_ld_flags = [],
                static_pic_dep_runtime_ld_flags = [],
                shared_dep_runtime_ld_flags = [],
                independent_shlib_interface_linker_flags = [],
                shlib_interfaces = ShlibInterfacesMode("disabled"),
                link_style = LinkStyle("static_pic"),
                link_weight = 1,
                binary_extension = "wasm",
                object_file_extension = "o",
                shared_library_name_default_prefix = "",
                shared_library_name_format = "{}.so",
                shared_library_versioned_name_format = "{}.so.{}" ,
                static_library_extension = "a",
                force_full_hybrid_if_capable = False,
                is_pdb_generated = False,
                link_ordering = None,
            ),
            bolt_enabled = False,
            binary_utilities_info = BinaryUtilitiesInfo(
                nm = _wrap("nm"),
                objcopy = _wrap("objcopy"),
                objdump = _wrap("objdump"),
                ranlib = _wrap("ranlib"),
                strip = _wrap("strip"),
                dwp = None,
            ),
            cxx_compiler_info = CxxCompilerInfo(
                compiler = cxx,
                preprocessor_flags = [],
                compiler_flags = ["-fPIC", "-O3"],
                compiler_type = "clang",
                supports_two_phase_compilation = False,
                supports_content_based_paths = False,
            ),
            c_compiler_info = CCompilerInfo(
                compiler = cc,
                preprocessor_flags = [],
                compiler_flags = ["-fPIC", "-O3"],
                compiler_type = "clang",
                supports_content_based_paths = False,
            ),
            as_compiler_info = CCompilerInfo(
                compiler = cc,
                compiler_type = "clang",
                supports_content_based_paths = False,
            ),
            asm_compiler_info = CCompilerInfo(
                compiler = cc,
                compiler_type = "clang",
            ),
            cvtres_compiler_info = CvtresCompilerInfo(
                compiler = None,
                preprocessor_flags = [],
                compiler_flags = [],
                compiler_type = "clang",
            ),
            rc_compiler_info = RcCompilerInfo(
                compiler = None,
                preprocessor_flags = [],
                compiler_flags = [],
                compiler_type = "clang",
            ),
            header_mode = HeaderMode("symlink_tree_only"),
            cpp_dep_tracking_mode = DepTrackingMode("show_headers"),
            pic_behavior = PicBehavior("supported"),
            llvm_link = RunInfo(args = cmd_args(sdk.project("install/bin/llvm-link"), hidden = [sdk])),
            use_dep_files = True,
            runtime_dependency_handling = RuntimeDependencyHandling("no_symlink"),
        ),
        CxxPlatformInfo(name = "wasm32-emscripten"),
    ]

_wasm_cxx_toolchain = rule(
    impl = _wasm_cxx_toolchain_impl,
    attrs = {
        "sdk": attrs.exec_dep(),
        "python": attrs.exec_dep(),
        "wrapper": attrs.source(),
        "_cxx_internal_tools": attrs.exec_dep(
            providers = [CxxInternalTools],
            default = "prelude//cxx/tools:internal_tools",
        ),
    },
    is_toolchain_rule = True,
)

def _toolchain_alias_impl(ctx: AnalysisContext):
    actual = ctx.attrs.actual
    providers = [actual[DefaultInfo]]
    for p in (RustToolchainInfo, CxxToolchainInfo, CxxPlatformInfo):
        if p in actual:
            providers.append(actual[p])
    return providers

toolchain_alias = rule(
    impl = _toolchain_alias_impl,
    attrs = {
        "actual": attrs.toolchain_dep(),
    },
    is_toolchain_rule = True,
)

def wasm_toolchains(host_sysroot):
    native.alias(
        name = "wasm-python",
        actual = select({
            "prelude//os:macos": ":native-unpack-python-aarch64-apple-darwin",
            "prelude//os:linux": select({
                "prelude//cpu:x86_64": ":native-unpack-python-x86_64-unknown-linux-gnu",
                "prelude//cpu:arm64": ":native-unpack-python-aarch64-unknown-linux-gnu",
            }),
        }),
    )
    _pyodide_python_include(
        name = "pyodide-python-include",
        xbuildenv = ":pyodide-xbuildenv",
        visibility = ["PUBLIC"],
    )
    _wasm_rust_sysroot(
        name = "rust-wasm-sysroot",
        host_sysroot = host_sysroot,
        wasm_std = ":rust-std-nightly-wasm32-unknown-emscripten",
        visibility = ["PUBLIC"],
    )
    _wasm_rust_toolchain(
        name = "rust-wasm",
        sysroot = ":rust-wasm-sysroot",
        visibility = ["PUBLIC"],
    )
    _wasm_cxx_toolchain(
        name = "cxx-wasm",
        sdk = ":emscripten-sdk",
        python = ":wasm-python",
        wrapper = "emcc_wrapper.py",
        visibility = ["PUBLIC"],
    )

def _to_wasm_impl(platform: PlatformInfo, refs: struct) -> PlatformInfo:
    _ = platform
    return refs.wasm_platform[PlatformInfo]

_wasm_transition = transition(
    impl = _to_wasm_impl,
    refs = {
        "wasm_platform": "root//:wasm32-emscripten-platform",
    },
)

def _baux_py_wasm_impl(ctx: AnalysisContext):
    src = ctx.attrs.src[DefaultInfo]
    sub = src.sub_targets.get("shared")
    artifact = sub[DefaultInfo].default_outputs[0] if sub else src.default_outputs[0]
    out = ctx.actions.copy_file(ctx.attrs.out, artifact)
    return [DefaultInfo(default_output = out)]

baux_py_wasm = rule(
    impl = _baux_py_wasm_impl,
    attrs = {
        "src": attrs.transition_dep(cfg = _wasm_transition),
        "out": attrs.string(),
    },
)
