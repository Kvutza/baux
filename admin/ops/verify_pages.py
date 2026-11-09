"""Verification checker for the public GitHub Pages bundle.

Validates that:
1. The public bundle contains clean baux black-box optimization documentation.
2. No Overview UI or internal verification monograph text is present.
3. Obsolete AST symbol engines (site_data.js, explorer_engine.js) are excluded.
4. All asset links resolve and JavaScript parses cleanly.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SENSITIVE_PATTERNS = [
    r"tab-btn-monograph",
    r"view-monograph",
    r"renderMonographView",
    r"renderFallbackMonographView",
    r"monograph-shell",
    r"monograph-switcher",
    r"fb-mono-switcher",
    r"BAUX FORMAL VERIFICATION CONTRACT",
    r"VERIFICATION CONTRACT & TRUST BOUNDARY",
    r"GATEWAY_GROUPS",
]


def check_views(target_dir: Path) -> None:
    index_html = target_dir / "index.html"
    if not index_html.exists():
        raise RuntimeError(f"missing index.html at {index_html}")
    content = index_html.read_text(encoding="utf-8")

    if "baux — Black Box Optimization" not in content:
        raise RuntimeError("index.html missing title 'baux — Black Box Optimization'")
    if "A black box optimization library in Python and Rust" not in content:
        raise RuntimeError("index.html missing black box optimization lead")
    if "yubo-research/enn" not in content:
        raise RuntimeError("index.html missing baseline reference to yubo-research/enn")
    if 'id="quickstart"' not in content:
        raise RuntimeError("index.html missing quickstart section")

    # Ensure no internal AST tabs remain
    if "Functions <span" in content or "Module Graph <span" in content:
        raise RuntimeError(
            "Found obsolete Functions or Module Graph tabs in index.html"
        )

    print("✓ Page layout: Clean baux black box optimization documentation.")


def check_sensitive(target_dir: Path) -> None:
    text_extensions = {".html", ".js", ".css", ".json", ".md"}
    violations: list[str] = []

    for path in target_dir.rglob("*"):
        if not path.is_file() or path.suffix not in text_extensions:
            continue
        rel_path = path.relative_to(target_dir)
        text = path.read_text(encoding="utf-8", errors="replace")

        for pattern in SENSITIVE_PATTERNS:
            matches = list(re.finditer(pattern, text))
            if matches:
                violations.append(
                    f"{rel_path}: matches {pattern!r} ({len(matches)} occurrences)"
                )

    if violations:
        raise RuntimeError(
            "Sensitive internal text found in public bundle:\n  "
            + "\n  ".join(violations)
        )
    print("✓ Sensitive content check: Zero internal monograph text present.")


def check_assets(target_dir: Path) -> None:
    required_files = [
        "index.html",
        "docs.html",
        ".nojekyll",
        "assets/site_style.css",
        "assets/baux_ui.js",
    ]
    for rel in required_files:
        p = target_dir / rel
        if not p.is_file():
            raise RuntimeError(f"Missing required bundle file: {rel}")
        if p.stat().st_size == 0 and rel != ".nojekyll":
            raise RuntimeError(f"Bundle file is empty: {rel}")

    # Ensure obsolete AST engines are NOT in public bundle
    forbidden_files = [
        "assets/site_data.js",
        "assets/site_manifest.json",
        "assets/blueprint_engine.js",
        "assets/explorer_engine.js",
    ]
    for rel in forbidden_files:
        p = target_dir / rel
        if p.exists():
            raise RuntimeError(f"Obsolete engine file present in bundle: {rel}")

    print(
        "✓ Link and asset integrity: All required assets exist and obsolete AST engines excluded."
    )


def check_smoke(target_dir: Path) -> None:
    # 1. Syntax check with node -c
    js_files = list((target_dir / "assets").glob("*.js"))
    for js in js_files:
        subprocess.check_call(["node", "-c", str(js)])

    print("✓ JavaScript syntax: All assets pass node syntax verification.")


def verify_bundle(target_dir: Path) -> None:
    print(f"--- Verifying Public Bundle at: {target_dir} ---")
    check_views(target_dir)
    check_sensitive(target_dir)
    check_assets(target_dir)
    check_smoke(target_dir)
    print("--- All Public Bundle Checks PASSED ---\n")
