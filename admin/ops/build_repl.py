"""Stage verified Pyodide WASM assets and deterministic ``baux_py.zip``."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "pyodide_manifest.json"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "baux" / "pyodide" / "v0.27.5"
FORBIDDEN_PATH_MARKERS = (b"/Users/", b"/home/runner/")
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SKIP_DIR_NAMES = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dylib"}
WASM_EXT_FILENAME = "baux_rust.cpython-312-wasm32-emscripten.so"
WASM_EXT_ARCNAME = f"baux/{WASM_EXT_FILENAME}"
COMPANION_THEME_SCRIPTS = (
    ("repl_worker_js", "repl_worker.js"),
    ("webgpu_bridge_js", "webgpu_bridge.js"),
    ("webgpu_worker_js", "webgpu_worker.js"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def download_verified(
    url: str, dst: Path, expected_size: int, expected_sha256: str
) -> None:
    if not url.startswith("https://cdn.jsdelivr.net/pyodide/"):
        raise ValueError(f"Refusing non-allowlisted Pyodide URL: {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    h = hashlib.sha256()
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "baux-pages-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
            size += len(chunk)
    actual_sha = h.hexdigest()
    if size != expected_size or actual_sha != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Verification failed for {dst.name}: size {size} != {expected_size} "
            f"or sha256 {actual_sha} != {expected_sha256}"
        )
    tmp.replace(dst)


def stage_pyodide(
    manifest: dict[str, Any],
    cache_dir: Path,
    *,
    offline: bool = False,
) -> dict[str, Path]:
    base_url = str(manifest["base_url"])
    files_spec = manifest["files"]
    resolved: dict[str, Path] = {}
    for name, meta in files_spec.items():
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(f"Unsafe filename in Pyodide manifest: {name!r}")
        expected_size = int(meta["size"])
        expected_sha = str(meta["sha256"]).lower()
        target = cache_dir / name
        needs_fetch = True
        if target.is_file():
            actual_size, actual_sha = sha256_file(target)
            if actual_size == expected_size and actual_sha == expected_sha:
                needs_fetch = False
        if needs_fetch:
            if offline:
                raise FileNotFoundError(
                    f"Missing or invalid cached Pyodide file {target} in --offline mode"
                )
            download_verified(f"{base_url}{name}", target, expected_size, expected_sha)
        resolved[name] = target
    return resolved


def trim_lockfile(
    lock_path: Path, allowed_roots: tuple[str, ...] = ("numpy",)
) -> bytes:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    all_pkgs: dict[str, Any] = data.get("packages", {})
    keep: set[str] = set()
    stack = list(allowed_roots)
    while stack:
        pkg = stack.pop()
        if pkg in keep or pkg not in all_pkgs:
            continue
        keep.add(pkg)
        for dep in all_pkgs[pkg].get("depends", []):
            stack.append(str(dep))
    trimmed = {
        "info": data.get("info", {}),
        "packages": {k: all_pkgs[k] for k in sorted(keep)},
    }
    return json.dumps(trimmed, separators=(",", ":"), sort_keys=False).encode("utf-8")


def _checkbytes(label: str, payload: bytes) -> None:
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker in payload:
            raise RuntimeError(f"Host path marker {marker!r} leaked into {label}")


def _resolve_existing_so(repo_root: Path, candidate: Path, origin: str) -> Path:
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        rel = repo_root / candidate
        if rel.is_file():
            return rel
    raise FileNotFoundError(
        f"WASM extension specified via {origin} not found: {candidate}"
    )


def resolve_wasm_so(repo_root: Path, wasm_so: Path | None = None) -> Path:
    """Locate compiled Emscripten C-API extension `.so`, failing loudly if absent."""
    if wasm_so is not None:
        return _resolve_existing_so(repo_root, wasm_so, "--wasm-so")

    env_val = os.environ.get("BAUX_WASM_SO", "").strip()
    if env_val:
        return _resolve_existing_so(repo_root, Path(env_val), "BAUX_WASM_SO")

    direct_candidates = (
        repo_root / ".cache" / "baux" / "wasm" / WASM_EXT_FILENAME,
        repo_root
        / "rust"
        / "target"
        / "wasm32-unknown-emscripten"
        / "release"
        / WASM_EXT_FILENAME,
        repo_root
        / "rust"
        / "target"
        / "wasm32-unknown-emscripten"
        / "release"
        / "libbaux_rust.so",
    )
    for cand in direct_candidates:
        if cand.is_file():
            return cand

    buck_out = repo_root / "buck-out"
    if buck_out.is_dir():
        exact_matches = sorted(
            (p for p in buck_out.rglob(WASM_EXT_FILENAME) if p.is_file()),
            key=lambda p: p.as_posix(),
        )
        if exact_matches:
            return exact_matches[0]
        target_matches = sorted(
            (
                p
                for p in buck_out.rglob("*.so")
                if p.is_file() and "baux-py-wasm" in p.as_posix()
            ),
            key=lambda p: p.as_posix(),
        )
        if target_matches:
            return target_matches[0]

    raise FileNotFoundError(
        f"Missing compiled {WASM_EXT_FILENAME}. Pass --wasm-so PATH, set BAUX_WASM_SO, "
        "or build //:baux-py-wasm into buck-out or .cache/baux/wasm/."
    )


def _load_verified_wasm_so(
    repo_root: Path, wasm_so: Path | None = None
) -> tuple[Path, bytes]:
    so_path = resolve_wasm_so(repo_root, wasm_so)
    payload = so_path.read_bytes()
    if not payload:
        raise RuntimeError(f"WASM extension file is empty: {so_path}")
    _checkbytes(WASM_EXT_ARCNAME, payload)
    return so_path, payload


def _build_pyzip_with_so(repo_root: Path, so_payload: bytes) -> bytes:
    py_root = repo_root / "python" / "baux"
    if not py_root.exists():
        raise FileNotFoundError(f"Required browser REPL source not found: {py_root}")

    entries: dict[str, bytes] = {}
    for src in sorted(py_root.rglob("*")):
        if not src.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in src.parts):
            continue
        if src.suffix in SKIP_SUFFIXES:
            continue
        rel = Path("baux") / src.relative_to(py_root)
        arcname = rel.as_posix()
        payload = src.read_bytes()
        _checkbytes(arcname, payload)
        entries[arcname] = payload

    _checkbytes(WASM_EXT_ARCNAME, so_payload)
    entries[WASM_EXT_ARCNAME] = so_payload

    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for arcname in sorted(entries):
            zi = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, entries[arcname])
    return buf.getvalue()


def build_pyzip(repo_root: Path, wasm_so: Path | None = None) -> bytes:
    _, so_payload = _load_verified_wasm_so(repo_root, wasm_so)
    return _build_pyzip_with_so(repo_root, so_payload)


def _stage_companion_scripts(
    repo_root: Path, assets_dirs: list[Path]
) -> dict[str, dict[str, Any]]:
    companion_entries: dict[str, dict[str, Any]] = {}
    theme_dir = repo_root / "forest" / "theme"

    for manifest_key, filename in COMPANION_THEME_SCRIPTS:
        src = theme_dir / filename
        if not src.is_file():
            continue
        blob = src.read_bytes()
        _checkbytes(filename, blob)
        for assets_dir in assets_dirs:
            (assets_dir / filename).write_bytes(blob)
        companion_entries[manifest_key] = {
            "path": filename,
            "size": len(blob),
            "sha256": sha256_bytes(blob),
        }

    oxcaml_candidates = (
        repo_root / "admin" / "ui" / "_build" / "default" / "oxcaml_eval.js",
        theme_dir / "oxcaml_eval.js",
    )
    oxcaml_src = next((p for p in oxcaml_candidates if p.is_file()), None)
    if oxcaml_src is not None:
        oxcaml_blob = oxcaml_src.read_bytes()
        _checkbytes("oxcaml_eval.js", oxcaml_blob)
        for assets_dir in assets_dirs:
            (assets_dir / "oxcaml_eval.js").write_bytes(oxcaml_blob)
        companion_entries["oxcaml_eval_js"] = {
            "path": "oxcaml_eval.js",
            "size": len(oxcaml_blob),
            "sha256": sha256_bytes(oxcaml_blob),
        }
    elif assets_dirs and all((d / "oxcaml_eval.js").is_file() for d in assets_dirs):
        oxcaml_blob = (assets_dirs[0] / "oxcaml_eval.js").read_bytes()
        _checkbytes("oxcaml_eval.js", oxcaml_blob)
        companion_entries["oxcaml_eval_js"] = {
            "path": "oxcaml_eval.js",
            "size": len(oxcaml_blob),
            "sha256": sha256_bytes(oxcaml_blob),
        }

    return companion_entries


def stage_assets(
    *,
    repo_root: Path,
    manifest_path: Path,
    cache_dir: Path,
    assets_dirs: list[Path],
    offline: bool = False,
    zip_only: bool = False,
    wasm_so: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, so_payload = _load_verified_wasm_so(repo_root, wasm_so)
    zip_bytes = _build_pyzip_with_so(repo_root, so_payload)
    zip_sha = sha256_bytes(zip_bytes)

    pyodide_entries: dict[str, dict[str, Any]] = {}
    staged_pyodide_blobs: dict[str, bytes] = {}

    if not zip_only:
        cached = stage_pyodide(manifest, cache_dir, offline=offline)
        for name, src_path in cached.items():
            if name == "pyodide-lock.json":
                blob = trim_lockfile(src_path, ("numpy",))
            else:
                blob = src_path.read_bytes()
            if name.endswith((".js", ".json")):
                _checkbytes(name, blob)
            staged_pyodide_blobs[name] = blob
            pyodide_entries[name] = {
                "size": len(blob),
                "sha256": sha256_bytes(blob),
            }

    for assets_dir in assets_dirs:
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "baux_py.zip").write_bytes(zip_bytes)
        if not zip_only:
            pyo_dir = assets_dir / "pyodide"
            pyo_dir.mkdir(parents=True, exist_ok=True)
            for name, blob in staged_pyodide_blobs.items():
                (pyo_dir / name).write_bytes(blob)

    companion_entries = _stage_companion_scripts(repo_root, assets_dirs)

    repl_manifest = {
        "schema_version": 1,
        "pyodide_version": manifest["pyodide_version"],
        "baux_py_zip": {
            "path": "baux_py.zip",
            "size": len(zip_bytes),
            "sha256": zip_sha,
        },
        "wasm_extension": {
            "path": WASM_EXT_ARCNAME,
            "size": len(so_payload),
            "sha256": sha256_bytes(so_payload),
        },
        "companion_files": companion_entries,
        "pyodide_files": pyodide_entries,
    }
    repl_manifest_bytes = (
        json.dumps(repl_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    for assets_dir in assets_dirs:
        (assets_dir / "repl_manifest.json").write_bytes(repl_manifest_bytes)

    return repl_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--wasm-so",
        type=Path,
        default=None,
        help="Path to compiled baux_rust.cpython-312-wasm32-emscripten.so",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        action="append",
        dest="assets_dirs",
        required=True,
        help="Destination assets/ directory (can be supplied multiple times)",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--zip-only", action="store_true")
    args = parser.parse_args(argv)

    summary = stage_assets(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        cache_dir=args.cache_dir.resolve(),
        assets_dirs=[p.resolve() for p in args.assets_dirs],
        offline=args.offline,
        zip_only=args.zip_only,
        wasm_so=args.wasm_so,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
