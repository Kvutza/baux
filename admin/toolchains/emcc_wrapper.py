#!/usr/bin/env python3
"""Hermetic Emscripten SDK wrapper for Buck2 C/C++ and Rust wasm32 actions."""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile

_BARE_WASM_LD_FLAGS = frozenset(
    {
        "--allow-undefined",
        "--gc-sections",
        "--no-entry",
        "--no-gc-sections",
        "--no-whole-archive",
        "--strip-debug",
        "--whole-archive",
    }
)

_LLVM_BINUTILS_MODES = frozenset(
    {
        "ar",
        "nm",
        "objcopy",
        "objdump",
        "ranlib",
    }
)


def _resolve_install_dir(sdk_dir: str) -> str:
    sdk_abs = os.path.abspath(sdk_dir)
    install_dir = os.path.join(sdk_abs, "install")
    return install_dir if os.path.isdir(install_dir) else sdk_abs


def _find_node() -> str | None:
    candidates = [
        shutil.which("node"),
        shutil.which("nodejs"),
        os.path.expanduser("~/.local/bin/node"),
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _expand_rsp_args(args: list[str]) -> list[str]:
    expanded: list[str] = []
    for arg in args:
        if arg.startswith("@") and len(arg) > 1 and os.path.isfile(arg[1:]):
            with open(arg[1:], encoding="utf-8") as fh:
                tokens = shlex.split(fh.read(), posix=True)
            expanded.extend(_expand_rsp_args(tokens))
        else:
            expanded.append(arg)
    return expanded


def _run_strip(args: list[str]) -> int:
    dst: str | None = None
    srcs: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-o" and i + 1 < len(args):
            dst = args[i + 1]
            i += 2
            continue
        if arg.startswith("-o") and len(arg) > 2:
            dst = arg[2:]
            i += 1
            continue
        if arg.startswith("-"):
            i += 1
            continue
        srcs.append(arg)
        i += 1

    # Rust passes -Cstrip=debuginfo and emcc links with -Wl,--strip-debug -g0, so
    # copying avoids llvm-strip crashes on wasm32-unknown-emscripten .rlib files.
    if dst and srcs:
        src = srcs[0]
        dst_dir = os.path.dirname(os.path.abspath(dst))
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile(src, dst)
        if os.access(src, os.X_OK):
            os.chmod(dst, 0o755)
    return 0


def _filter_ld_args(raw_args: list[str]) -> list[str]:
    args = _expand_rsp_args(raw_args)
    filtered: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        # Binaryen 117 wasm-opt (triggered by -O1/-O2/-O3/-Os/-Oz) rejects LLVM 22
        # bulk-memory-opt feature sections emitted by rustc 2026-09-29.
        if arg in {
            "-O0",
            "-O1",
            "-O2",
            "-O3",
            "-Os",
            "-Oz",
            "-fwasm-exceptions",
            "-shared",
        }:
            i += 1
            continue
        if arg.startswith("-fuse-ld="):
            i += 1
            continue
        if arg == "-fuse-ld" and i + 1 < len(args):
            i += 2
            continue
        if arg.startswith("-sSIDE_MODULE="):
            i += 1
            continue
        if arg == "-s" and i + 1 < len(args) and args[i + 1].startswith("SIDE_MODULE="):
            i += 2
            continue
        if arg in _BARE_WASM_LD_FLAGS:
            filtered.append(f"-Wl,{arg}")
            i += 1
            continue
        filtered.append(arg)
        i += 1

    # -sSIDE_MODULE=2 runs --gc-sections cleanly while preserving EXPORTED_FUNCTIONS
    # and avoiding __cpp_exception tag imports incompatible with Pyodide JS EH.
    return [
        "-sSIDE_MODULE=2",
        "-sWASM_BIGINT",
        "-sERROR_ON_UNDEFINED_SYMBOLS=0",
        "-Wl,--no-entry",
        "-Wl,--strip-debug",
        "-g0",
        *filtered,
    ]


def _quote_rsp_arg(arg: str) -> str:
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _run_emcc(install_dir: str, mode: str, args: list[str]) -> int:
    emcc_py = os.path.join(
        install_dir,
        "emscripten",
        "em++.py" if mode == "cxx" else "emcc.py",
    )
    if mode == "ld":
        final_args = _filter_ld_args(args)
    else:
        final_args = _expand_rsp_args(args)

    with tempfile.TemporaryDirectory(prefix="baux_em_") as tmpdir:
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        src_cache = os.path.join(install_dir, "emscripten", "cache")
        src_sysroot = os.path.join(src_cache, "sysroot")
        if os.path.isdir(src_sysroot):
            os.symlink(src_sysroot, os.path.join(cache_dir, "sysroot"))
        src_stamp = os.path.join(src_cache, "sysroot_install.stamp")
        if os.path.isfile(src_stamp):
            shutil.copyfile(src_stamp, os.path.join(cache_dir, "sysroot_install.stamp"))

        node_js = _find_node()
        em_config = os.path.join(tmpdir, "em_config")
        with open(em_config, "w", encoding="utf-8") as fh:
            fh.write(f"EMSCRIPTEN_ROOT = {os.path.join(install_dir, 'emscripten')!r}\n")
            fh.write(f"LLVM_ROOT = {os.path.join(install_dir, 'bin')!r}\n")
            fh.write(f"BINARYEN_ROOT = {install_dir!r}\n")
            fh.write(f"CACHE = {cache_dir!r}\n")
            fh.write(f"NODE_JS = {node_js!r}\n")
            fh.write("JS_ENGINES = [NODE_JS] if NODE_JS else []\n")

        env = os.environ.copy()
        for stale in (
            "EM_CACHE_IS_LOCKED",
            "EMCC_CFLAGS",
            "_PYTHON_SYSCONFIGDATA_NAME",
        ):
            env.pop(stale, None)
        env["EM_CONFIG"] = em_config
        env["EM_CACHE"] = cache_dir
        env["EMSDK_PYTHON"] = sys.executable
        env["EMCC_CORES"] = "1"
        env["EMCC_SKIP_SANITY_CHECK"] = "1"

        cmd = [sys.executable, "-E", emcc_py]
        if len(final_args) > 48:
            rsp_path = os.path.join(tmpdir, "args.rsp")
            with open(rsp_path, "w", encoding="utf-8") as fh:
                for arg in final_args:
                    fh.write(_quote_rsp_arg(arg) + "\n")
            cmd.append(f"@{rsp_path}")
        else:
            cmd.extend(final_args)

        return subprocess.run(cmd, env=env).returncode


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.stderr.write("usage: emcc_wrapper.py <sdk_dir> <mode> [args...]\n")
        return 2

    sdk_dir = argv[1]
    mode = argv[2]
    rest = argv[3:]

    if mode == "strip":
        return _run_strip(rest)

    install_dir = _resolve_install_dir(sdk_dir)
    if mode in _LLVM_BINUTILS_MODES:
        tool = os.path.join(install_dir, "bin", f"llvm-{mode}")
        os.execv(tool, [tool, *rest])

    if mode in {"cc", "cxx", "ld"}:
        return _run_emcc(install_dir, mode, rest)

    sys.stderr.write(f"emcc_wrapper.py: unsupported mode {mode!r}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
