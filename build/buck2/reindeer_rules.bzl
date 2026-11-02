load("@prelude//rust:cargo_package.bzl", "cargo")
load("@prelude//rust:cargo_buildscript.bzl", _buildscript_run = "buildscript_run")
load("//build/buck2:python.bzl", "PYTHON_VERSION")

_PYO3_CONFIG = {
    "3.14": "696d706c656d656e746174696f6e3d43507974686f6e0a76657273696f6e3d332e31340a7368617265643d747275650a706f696e7465725f77696474683d36340a6275696c645f666c6167733d0a73757070726573735f6275696c645f7363726970745f6c696e6b5f6c696e65733d66616c73650a",
    "3.15": "696d706c656d656e746174696f6e3d43507974686f6e0a76657273696f6e3d332e31350a7368617265643d747275650a706f696e7465725f77696474683d36340a6275696c645f666c6167733d0a73757070726573735f6275696c645f7363726970745f6c696e6b5f6c696e65733d66616c73650a",
}

_FAST_DEV_FLAGS = select({
    "root//:release-mode": [],
    "DEFAULT": ["-Zthreads=8", "-Ccodegen-units=16"],
})

def app_rust_library(name, rustc_flags = None, **kwargs):
    flags = (rustc_flags or []) + _FAST_DEV_FLAGS
    # Final cdylib links use LTO; proc macros must never receive -Clto.
    if kwargs.get("dlopen_enable", False):
        flags = flags + select({
            "root//:release-mode": ["-Clto=thin"],
            "DEFAULT": [],
        })
    cargo.rust_library(
        name = name,
        rustc_flags = flags,
        **kwargs
    )

def app_rust_binary(name, rustc_flags = None, **kwargs):
    cargo.rust_binary(
        name = name,
        rustc_flags = (rustc_flags or []) + _FAST_DEV_FLAGS,
        **kwargs
    )

def reindeer_rust_library(
        name,
        crate,
        rustc_flags = None,
        visibility = None,
        **kwargs):
    if crate == "cuda_bindings":
        srcs = list(kwargs.pop("srcs", [])) + native.glob(["fixups/cuda-bindings/cuda/fixtures/*.rs"])
        env = dict(kwargs.pop("env", {}))
        env["OUT_DIR"] = "../../../../../../../../rust/fixups/cuda-bindings/cuda/fixtures"
        env["CUTILE_RESOLVED_CUDA_TOOLKIT_PATH"] = "/usr/local/cuda"
        kwargs["srcs"] = srcs
        kwargs["env"] = env
    elif crate in ("cuda_device", "deser", "rand"):
        rustc_flags = (rustc_flags or []) + [
            "-Zalways-encode-mir",
            "-Csymbol-mangling-version=v0",
            "-Cdebug-assertions=off",
            "-Copt-level=3",
        ]
    cargo.rust_library(
        name = name,
        crate = crate,
        rustc_flags = rustc_flags or [],
        visibility = visibility,
        **kwargs
    )

def reindeer_rust_binary(name, rustc_flags = None, **kwargs):
    cargo.rust_binary(
        name = name,
        rustc_flags = rustc_flags or [],
        **kwargs
    )

def buildscript_run(package_name, env = None, **kwargs):
    resolved_env = dict(env or {})
    if package_name == "pyo3-ffi":
        resolved_env["PYO3_CROSS"] = "1"
        resolved_env["PYO3_CROSS_PYTHON_VERSION"] = PYTHON_VERSION
    if package_name in ["numpy", "pyo3"]:
        resolved_env["DEP_PYTHON_PYO3_CONFIG"] = _PYO3_CONFIG[PYTHON_VERSION]
    _buildscript_run(
        package_name = package_name,
        env = resolved_env,
        **kwargs
    )
