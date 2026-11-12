"""Unified static analysis, workflow hardening, and dependency security runner.

Executes Trail of Bits security checks locally and in CI:
1. GitHub Actions workflow audit via `uvx zizmor --pedantic .github/workflows/`.
2. Custom static analysis rules via `uvx semgrep` (`.semgrep/baux-web-security.yml`)
   across `admin/ops/`, `forest/theme/`, `forest/js/`, and `admin/ui/baux_ui.ml`
   (excluding `node_modules` and minified `forester.js`).
3. Locked Python dependency vulnerability audit via `uvx pip-audit --strict`
   against `uv.lock` (with `--offline` graceful handling for airgapped/offline runs).
4. Optional assembled GitHub Pages bundle verification via `verify_pages.py`
   when `--bundle-dir` is provided.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "admin/ops"))
from verify_pages import verify_bundle

WORKFLOWS_DIR = ROOT / ".github/workflows"
SEMGREP_CONFIG = ROOT / ".semgrep/baux-web-security.yml"

NETWORK_ERROR_MARKERS = (
    "ConnectionError",
    "NewConnectionError",
    "NameResolutionError",
    "Max retries exceeded",
    "Temporary failure in name resolution",
    "nodename nor servname provided",
    "Failed to establish a new connection",
    "Network is unreachable",
)


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str  # "PASS", "FAIL", "SKIP"
    detail: str


def resolve_binary(name: str) -> str:
    """Resolve `uv` or `uvx` on PATH or standard user-local install locations."""
    found = shutil.which(name)
    if found:
        return found
    for candidate in (
        Path.home() / ".local/bin" / name,
        Path.home() / ".cargo/bin" / name,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        f"Required executable {name!r} not found on PATH or ~/.local/bin/{name}"
    )


def semgrep_targets() -> list[str]:
    """Collect scan targets across admin/ops, tracked Forester theme/JS, and OCaml UI."""
    targets: list[str] = []
    for rel in ("admin/ops", "forest/theme", "forest/js"):
        if (ROOT / rel).is_dir():
            targets.append(rel)
    if (ROOT / "admin/ui/baux_ui.ml").is_file():
        targets.append("admin/ui/baux_ui.ml")

    return targets


def run_zizmor(uvx_bin: str) -> GateResult:
    print("=== [1/4] GitHub Actions Workflow Hardening (zizmor --pedantic) ===")
    if not WORKFLOWS_DIR.is_dir():
        msg = f"Missing workflows directory: {WORKFLOWS_DIR}"
        print(f"✗ {msg}\n")
        return GateResult("zizmor", "FAIL", msg)

    cmd = [uvx_bin, "zizmor", "--pedantic", str(WORKFLOWS_DIR.relative_to(ROOT))]
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    print()
    if proc.returncode == 0:
        return GateResult("zizmor", "PASS", ".github/workflows/ passed pedantic audit")
    return GateResult("zizmor", "FAIL", f"zizmor exited with code {proc.returncode}")


def run_semgrep(uvx_bin: str) -> GateResult:
    print("=== [2/4] Static Security Analysis (Semgrep custom rules) ===")
    if not SEMGREP_CONFIG.is_file():
        msg = f"Missing Semgrep config: {SEMGREP_CONFIG}"
        print(f"✗ {msg}\n")
        return GateResult("semgrep", "FAIL", msg)

    env = os.environ.copy()
    env["SEMGREP_SEND_METRICS"] = "off"

    cfg_rel = str(SEMGREP_CONFIG.relative_to(ROOT))
    validate_cmd = [
        uvx_bin,
        "semgrep",
        "scan",
        "--metrics=off",
        "--validate",
        "--config",
        cfg_rel,
    ]
    val_proc = subprocess.run(validate_cmd, cwd=str(ROOT), env=env, check=False)
    if val_proc.returncode != 0:
        print()
        return GateResult(
            "semgrep",
            "FAIL",
            f"Rule validation failed (exit {val_proc.returncode})",
        )

    targets = semgrep_targets()
    scan_cmd = [
        uvx_bin,
        "semgrep",
        "scan",
        "--metrics=off",
        "--no-git-ignore",
        "--error",
        "--config",
        cfg_rel,
        "--exclude",
        "node_modules",
        "--exclude",
        "forester.js",
        "--exclude",
        "__pycache__",
        "--exclude",
        "_build",
        "--exclude",
        "fonts",
        *targets,
    ]
    scan_proc = subprocess.run(scan_cmd, cwd=str(ROOT), env=env, check=False)
    print()
    if scan_proc.returncode == 0:
        return GateResult(
            "semgrep",
            "PASS",
            f"0 findings across {len(targets)} target path specs",
        )
    return GateResult(
        "semgrep", "FAIL", f"Semgrep reported findings (exit {scan_proc.returncode})"
    )


def pip_audit(
    uv_bin: str,
    uvx_bin: str,
    *,
    strict: bool,
    offline: bool,
    graceful_offline: bool,
    ignore_vulns: list[str],
) -> GateResult:
    print("=== [3/4] Dependency Vulnerability Audit (pip-audit) ===")
    if offline:
        msg = "Offline mode requested (--offline); skipping OSV network query"
        print(f"↷ {msg}\n")
        return GateResult("pip-audit", "SKIP", msg)

    ignore_args: list[str] = []
    for vuln_id in ignore_vulns:
        ignore_args.extend(["--ignore-vuln", vuln_id])

    strict_args = ["--strict"] if strict else []

    with tempfile.TemporaryDirectory(prefix="baux-sec-audit-") as tmp_dir:
        req_file = Path(tmp_dir) / "locked-requirements.txt"
        uv_lock = ROOT / "uv.lock"

        if uv_lock.is_file():
            export_cmd = [
                uv_bin,
                "export",
                "--frozen",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(req_file),
            ]
            exp_proc = subprocess.run(
                export_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if exp_proc.returncode != 0:
                err_txt = (exp_proc.stderr or exp_proc.stdout).strip()
                print(err_txt)
                print()
                return GateResult(
                    "pip-audit",
                    "FAIL",
                    f"uv export failed (exit {exp_proc.returncode})",
                )
            audit_cmd = [
                uvx_bin,
                "pip-audit",
                *strict_args,
                "--no-deps",
                "--disable-pip",
                "-r",
                str(req_file),
                *ignore_args,
            ]
        else:
            audit_cmd = [uvx_bin, "pip-audit", *strict_args, *ignore_args]

        proc = subprocess.run(
            audit_cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    combined = "\n".join(
        part.strip() for part in (proc.stdout, proc.stderr) if part and part.strip()
    )
    if combined:
        print(combined)
    print()

    if proc.returncode == 0:
        return GateResult("pip-audit", "PASS", "0 known vulnerabilities in uv.lock")

    if graceful_offline and any(m in combined for m in NETWORK_ERROR_MARKERS):
        msg = "Network unreachable during OSV lookup; skipped via --graceful-offline"
        print(f"↷ {msg}\n")
        return GateResult("pip-audit", "SKIP", msg)

    return GateResult(
        "pip-audit",
        "FAIL",
        f"pip-audit exited with code {proc.returncode}",
    )


def bundle_audit(bundle_dir: Path | None) -> GateResult:
    print("=== [4/4] Public Pages Bundle Verification (verify_pages.py) ===")
    if bundle_dir is None:
        msg = "No --bundle-dir provided; bundle check skipped"
        print(f"↷ {msg}\n")
        return GateResult("verify_pages", "SKIP", msg)

    resolved = bundle_dir.resolve()
    try:
        verify_bundle(resolved)
    except Exception as exc:
        print(f"✗ Bundle verification failed: {exc}\n")
        return GateResult("verify_pages", "FAIL", str(exc))
    return GateResult("verify_pages", "PASS", f"Verified bundle at {resolved}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BAUX Trail of Bits security audits (zizmor, semgrep, pip-audit)."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional assembled Pages bundle directory to validate via verify_pages.py",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pip-audit in --strict mode (default: enabled)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network-dependent OSV vulnerability queries for offline execution",
    )
    parser.add_argument(
        "--graceful-offline",
        action="store_true",
        help="Downgrade network-unreachability errors in pip-audit to SKIP instead of FAIL",
    )
    parser.add_argument(
        "--ignore-vuln",
        action="append",
        default=[],
        metavar="VULN_ID",
        help="Ignore a specific CVE/GHSA/PYSEC ID in pip-audit (repeatable)",
    )
    parser.add_argument(
        "--skip-zizmor",
        action="store_true",
        help="Skip zizmor workflow audit",
    )
    parser.add_argument(
        "--skip-semgrep",
        action="store_true",
        help="Skip Semgrep static analysis",
    )
    parser.add_argument(
        "--skip-pip-audit",
        action="store_true",
        help="Skip pip-audit dependency check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uv_bin = resolve_binary("uv")
    uvx_bin = resolve_binary("uvx")

    results: list[GateResult] = []

    if args.skip_zizmor:
        results.append(GateResult("zizmor", "SKIP", "Skipped via --skip-zizmor"))
    else:
        results.append(run_zizmor(uvx_bin))

    if args.skip_semgrep:
        results.append(GateResult("semgrep", "SKIP", "Skipped via --skip-semgrep"))
    else:
        results.append(run_semgrep(uvx_bin))

    if args.skip_pip_audit:
        results.append(GateResult("pip-audit", "SKIP", "Skipped via --skip-pip-audit"))
    else:
        results.append(
            pip_audit(
                uv_bin,
                uvx_bin,
                strict=args.strict,
                offline=args.offline,
                graceful_offline=args.graceful_offline,
                ignore_vulns=args.ignore_vuln,
            )
        )

    results.append(bundle_audit(args.bundle_dir))

    print("=== Security Audit Summary ===")
    failed = False
    for res in results:
        icon = {"PASS": "✓", "SKIP": "↷", "FAIL": "✗"}.get(res.status, "?")
        print(f"  {icon} [{res.status:4s}] {res.name:14s} — {res.detail}")
        if res.status == "FAIL":
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
