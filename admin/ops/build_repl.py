"""Stage verified Pyodide WASM assets and deterministic ``baux_py.zip``."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
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


def build_pyzip(repo_root: Path) -> bytes:
    py_root = repo_root / "python" / "baux"
    wasm_rust = repo_root / "forest" / "py" / "rust_wasm.py"
    wasm_oxcaml = repo_root / "forest" / "py" / "oxcaml_wasm.py"
    for req in (py_root, wasm_rust, wasm_oxcaml):
        if not req.exists():
            raise FileNotFoundError(f"Required browser REPL source not found: {req}")

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

    rust_payload = wasm_rust.read_bytes()
    _checkbytes("baux/baux_rust.py", rust_payload)
    entries["baux/baux_rust.py"] = rust_payload

    oxcaml_payload = wasm_oxcaml.read_bytes()
    _checkbytes("baux/benchmarks/_oxcaml_wasm.py", oxcaml_payload)
    entries["baux/benchmarks/_oxcaml_wasm.py"] = oxcaml_payload

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


def stage_assets(
    *,
    repo_root: Path,
    manifest_path: Path,
    cache_dir: Path,
    assets_dirs: list[Path],
    offline: bool = False,
    zip_only: bool = False,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    zip_bytes = build_pyzip(repo_root)
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

    repl_manifest = {
        "schema_version": 1,
        "pyodide_version": manifest["pyodide_version"],
        "baux_py_zip": {
            "path": "baux_py.zip",
            "size": len(zip_bytes),
            "sha256": zip_sha,
        },
        "pyodide_files": pyodide_entries,
    }
    repl_manifest_bytes = (
        json.dumps(repl_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    worker_src = repo_root / "forest" / "theme" / "repl_worker.js"
    for assets_dir in assets_dirs:
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "baux_py.zip").write_bytes(zip_bytes)
        (assets_dir / "repl_manifest.json").write_bytes(repl_manifest_bytes)
        if worker_src.is_file():
            shutil.copy2(worker_src, assets_dir / "repl_worker.js")
        if not zip_only:
            pyo_dir = assets_dir / "pyodide"
            pyo_dir.mkdir(parents=True, exist_ok=True)
            for name, blob in staged_pyodide_blobs.items():
                (pyo_dir / name).write_bytes(blob)

    return repl_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
