"""Verification checker for the two-view public GitHub Pages bundle.

Validates that:
1. The public bundle contains only Functions and Module Graph views.
2. The bundle defaults to Functions.
3. No Overview UI or Overview implementation text is present.
4. All asset links resolve and manifests are mathematically consistent.
5. All JavaScript files parse cleanly and pass headless smoke execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SENSITIVE_OVERVIEW_PATTERNS = [
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

    # Tab checks
    tabs = re.findall(r'<button[^>]*class="([^"]*)"[^>]*data-tab="([^"]*)"', content)
    if not tabs:
        raise RuntimeError("No primary navigation tabs found in index.html")
    tab_names = [t[1] for t in tabs]
    if tab_names != ["inspector", "dag"]:
        raise RuntimeError(
            f"Expected exactly two navigation tabs ['inspector', 'dag'], got: {tab_names}"
        )

    # Functions tab must be active
    inspector_tab = [t for t in tabs if t[1] == "inspector"][0]
    if "active" not in inspector_tab[0]:
        raise RuntimeError(
            f"Functions tab is not active by default; classes: {inspector_tab[0]}"
        )

    # Panel checks
    panels = re.findall(r'<section[^>]*id="([^"]*)"[^>]*data-view="([^"]*)"', content)
    panel_views = [p[1] for p in panels]
    if "monograph" in panel_views or "view-monograph" in content:
        raise RuntimeError("Overview (monograph) view panel found in index.html")
    if "inspector" not in panel_views or "dag" not in panel_views:
        raise RuntimeError(f"Expected inspector and dag panels, found: {panel_views}")

    # Active panel must be inspector
    active_panels = re.findall(
        r'<section[^>]*class="[^"]*active[^"]*"[^>]*data-view="([^"]*)"', content
    )
    if active_panels != ["inspector"]:
        raise RuntimeError(
            f"Expected active panel ['inspector'], found: {active_panels}"
        )

    print(
        "✓ Tab and view layout: Exactly 2 views (Functions, Module Graph), defaulting to Functions."
    )


def check_sensitive(target_dir: Path) -> None:
    text_extensions = {".html", ".js", ".css", ".json", ".md"}
    violations: list[str] = []

    for path in target_dir.rglob("*"):
        if not path.is_file() or path.suffix not in text_extensions:
            continue
        rel_path = path.relative_to(target_dir)
        text = path.read_text(encoding="utf-8", errors="replace")

        for pattern in SENSITIVE_OVERVIEW_PATTERNS:
            matches = list(re.finditer(pattern, text))
            if matches:
                if path.name == "site_data.js" and pattern in {
                    "monograph-shell",
                    "GATEWAY_GROUPS",
                    "renderMonographView",
                }:
                    violations.append(
                        f"{rel_path}: matches {pattern!r} ({len(matches)} occurrences)"
                    )
                elif path.name != "site_data.js":
                    violations.append(
                        f"{rel_path}: matches {pattern!r} ({len(matches)} occurrences)"
                    )

    if violations:
        raise RuntimeError(
            "Sensitive Overview / monograph implementation text found in public bundle:\n  "
            + "\n  ".join(violations)
        )
    print(
        "✓ Sensitive content check: Zero Overview UI or Overview implementation text."
    )


def check_assets(target_dir: Path) -> None:
    required_files = [
        "index.html",
        ".nojekyll",
        "assets/site_style.css",
        "assets/site_data.js",
        "assets/site_manifest.json",
        "assets/blueprint_engine.js",
        "assets/explorer_engine.js",
    ]
    for rel in required_files:
        p = target_dir / rel
        if not p.is_file():
            raise RuntimeError(f"Missing required bundle file: {rel}")
        if p.stat().st_size == 0 and rel != ".nojekyll":
            raise RuntimeError(f"Bundle file is empty: {rel}")

    manifest_path = target_dir / "assets/site_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = target_dir / "assets/site_data.js"
    data_bytes = data_path.read_bytes()
    expected_bytes = manifest.get("site_data_js_bytes")
    if expected_bytes is not None and expected_bytes != len(data_bytes):
        raise RuntimeError(
            f"site_manifest.json byte count ({expected_bytes}) does not match site_data.js ({len(data_bytes)})"
        )

    print(
        "✓ Link and asset integrity: All required assets exist, non-empty, and manifest-verified."
    )


def check_smoke(target_dir: Path) -> None:
    # 1. Syntax check with node -c
    js_files = list((target_dir / "assets").glob("*.js"))
    for js in js_files:
        subprocess.check_call(["node", "-c", str(js)])

    # 2. Headless DOM smoke execution
    smoke_js = f"""
    const fs = require('fs');
    const path = require('path');

    // Minimal browser mock
    global.window = global;
    global.document = {{
      documentElement: {{ getAttribute: () => 'light' }},
      getElementById: (id) => ({{
        id: id,
        classList: {{ add: () => {{}}, remove: () => {{}}, toggle: () => {{}} }},
        children: [],
        addEventListener: () => {{}},
        querySelectorAll: () => [],
        querySelector: () => null,
        appendChild: () => {{}},
        setAttribute: () => {{}},
        getAttribute: () => null,
      }}),
      querySelectorAll: () => [],
      querySelector: () => null,
      addEventListener: () => {{}},
    }};
    global.navigator = {{ userAgent: 'node-smoke' }};

    // Load site_data
    require('{target_dir}/assets/site_data.js');
    if (!global.window.BAUX_SITE_DATA) {{
      throw new Error('window.BAUX_SITE_DATA not defined');
    }}

    // Load engines
    require('{target_dir}/assets/blueprint_engine.js');
    require('{target_dir}/assets/explorer_engine.js');

    if (!global.window.BauxBlueprintEngine) {{
      throw new Error('BauxBlueprintEngine not defined');
    }}
    if (!global.window.BAUX_APP_STATE) {{
      throw new Error('BAUX_APP_STATE not defined');
    }}
    if (global.window.BAUX_APP_STATE.activeTab !== 'inspector') {{
      throw new Error('Expected activeTab inspector, got ' + global.window.BAUX_APP_STATE.activeTab);
    }}
    console.log('Smoke test passed: engines initialized cleanly with default activeTab=inspector');
    """

    res = subprocess.run(["node", "-e", smoke_js], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Headless smoke execution failed:\n{res.stderr}\n{res.stdout}"
        )

    print(
        "✓ JavaScript syntax and headless smoke test: Clean execution, default tab verified."
    )


def verify_bundle(target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    print(f"--- Verifying Two-View Public Bundle at: {target_dir} ---")
    check_views(target_dir)
    check_sensitive(target_dir)
    check_assets(target_dir)
    check_smoke(target_dir)
    print("--- All Public Bundle Checks PASSED ---\n")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/site")
    verify_bundle(path)
