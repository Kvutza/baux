"""Build, verify, and publish the clean two-view GitHub Pages bundle.

Generates the public bundle from docs/site/:
- Preserves only Functions and Module Graph views, defaulting to Functions.
- Strips Overview UI and Overview implementation text.
- Validates bundle via admin/ops/verify_pages.py.
- Deploys cleanly to the isolated gh-pages branch.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_SITE = ROOT / "docs/site"
sys.path.insert(0, str(ROOT / "admin/ops"))
from verify_pages import verify_bundle


def transform_html(src_html: str) -> str:
    # 1. Remove monograph nav tab and activate inspector tab
    html = re.sub(
        r'\s*<button type="button" class="nav-tab active" data-tab="monograph" id="tab-btn-monograph">\s*Overview\s*</button>',
        "",
        src_html,
    )
    html = html.replace(
        '<button type="button" class="nav-tab" data-tab="inspector" id="tab-btn-inspector">',
        '<button type="button" class="nav-tab active" data-tab="inspector" id="tab-btn-inspector">',
    )

    # 2. Remove monograph section and activate inspector section
    html = re.sub(
        r'\s*<section class="view-panel active" id="view-monograph" data-view="monograph"></section>',
        "",
        html,
    )
    html = html.replace(
        '<section class="view-panel" id="view-inspector" data-view="inspector"></section>',
        '<section class="view-panel active" id="view-inspector" data-view="inspector"></section>',
    )

    return html


def transform_blueprint(src_js: str) -> str:
    # 1. Remove GATEWAY_GROUPS array definition (lines 11-48)
    js = re.sub(
        r"var GATEWAY_GROUPS\s*=\s*\[[\s\S]*?\n\s*\];\n",
        "",
        src_js,
    )

    # 2. Remove VIEW I renderMonographView and its monograph-specific sub-functions (lines 422 to 884)
    # Target from /* --- 3. VIEW I to function bindSharedSymbolAndModuleButtons
    js = re.sub(
        r"/\*\s*-+\s*3\.\s*VIEW I[\s\S]*?(?=function bindSharedSymbolAndModuleButtons)",
        "",
        js,
    )

    # 3. Remove renderMonographView() call from init()
    js = re.sub(
        r"\s*renderMonographView\(\);",
        "",
        js,
    )

    # 4. Remove renderMonographView from api object
    js = re.sub(
        r"\s*renderMonographView:\s*renderMonographView,",
        "",
        js,
    )

    return js


def transform_explorer(src_js: str) -> str:
    # 1. Default activeTab to inspector
    js = re.sub(
        r'activeTab:\s*"monograph"',
        'activeTab: "inspector"',
        src_js,
    )

    # 2. In normalizeTabName, map monograph to inspector
    js = re.sub(
        r'if \(tabName === "reference" \|\| tabName === "lean"\) return "api";',
        'if (tabName === "reference" || tabName === "lean") return "api";\n    if (tabName === "monograph") return "inspector";',
        js,
    )

    # 3. In activateTab, remove monograph from viewIds
    js = js.replace(
        'const viewIds = ["monograph", "dag", "inspector", "api"];',
        'const viewIds = ["dag", "inspector", "api"];',
    )

    # 4. Remove the tabName === "monograph" branch in activateTab
    js = re.sub(
        r'if \(tabName === "monograph"\)\s*\{[\s\S]*?\}\s*else if \(tabName === "dag"\)',
        'if (tabName === "dag")',
        js,
    )

    # 5. Remove renderFallbackMonographView function body
    js = re.sub(
        r"// Fallback Verification Contract View[\s\S]*?(?=// Fallback Sugiyama Layered DAG View)",
        "",
        js,
    )

    # 6. Final activateTab default
    js = js.replace(
        'activateTab(appState.activeTab || "monograph");',
        'activateTab(appState.activeTab || "inspector");',
    )

    return js


def assemble_bundle(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_assets = dest_dir / "assets"
    dest_assets.mkdir(parents=True, exist_ok=True)

    # Copy and transform static assets
    for name in ["site_data.js", "site_manifest.json"]:
        src = DOCS_SITE / "assets" / name
        if src.exists():
            shutil.copy2(src, dest_assets / name)

    # Sanitize site_style.css
    raw_css = (DOCS_SITE / "assets/site_style.css").read_text(encoding="utf-8")
    clean_css = raw_css.replace("(#view-monograph)", "")
    (dest_assets / "site_style.css").write_text(clean_css, encoding="utf-8")

    # Ensure .nojekyll exists
    (dest_dir / ".nojekyll").touch()

    # Transform index.html
    raw_html = (DOCS_SITE / "index.html").read_text(encoding="utf-8")
    public_html = transform_html(raw_html)
    (dest_dir / "index.html").write_text(public_html, encoding="utf-8")

    # Transform blueprint_engine.js
    raw_blueprint = (DOCS_SITE / "assets/blueprint_engine.js").read_text(
        encoding="utf-8"
    )
    public_blueprint = transform_blueprint(raw_blueprint)
    (dest_assets / "blueprint_engine.js").write_text(public_blueprint, encoding="utf-8")

    # Transform explorer_engine.js
    raw_explorer = (DOCS_SITE / "assets/explorer_engine.js").read_text(encoding="utf-8")
    public_explorer = transform_explorer(raw_explorer)
    (dest_assets / "explorer_engine.js").write_text(public_explorer, encoding="utf-8")

    # Copy docs.html if present
    docs_html = DOCS_SITE / "docs.html"
    if docs_html.exists():
        shutil.copy2(docs_html, dest_dir / "docs.html")

    # Preserve logos if present in dest or source
    for logo in ["logo-light.svg", "logo-dark.svg"]:
        src_logo = DOCS_SITE / "assets" / logo
        if src_logo.exists():
            shutil.copy2(src_logo, dest_assets / logo)


def deploy_pages(ws_dir: Path) -> None:
    print(f"--- Deploying gh-pages from workspace: {ws_dir} ---")
    # 1. jj describe
    subprocess.check_call(
        ["jj", "describe", "-m", "kvutza/pages"],
        cwd=str(ws_dir),
    )
    # 2. jj bookmark set
    subprocess.check_call(
        ["jj", "bookmark", "set", "gh-pages", "-r", "@"],
        cwd=str(ws_dir),
    )
    # 3. jj git push
    subprocess.check_call(
        ["jj", "git", "push", "--bookmark", "gh-pages"],
        cwd=str(ws_dir),
    )
    print("✓ Successfully pushed kvutza/pages to origin/gh-pages!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("/tmp/gh-pages-ws"),
        help="Target directory or workspace for the public bundle",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the bundle to origin/gh-pages using jj",
    )
    parser.add_argument(
        "--skip-check-local",
        action="store_true",
        help="Skip ./baux research site --check freshness check",
    )
    args = parser.parse_args()

    if not args.skip_check_local:
        print("Checking that local docs/site is fresh...")
        subprocess.check_call(["./baux", "research", "site", "--check"], cwd=str(ROOT))

    print(f"Assembling two-view public bundle into {args.target_dir}...")
    assemble_bundle(args.target_dir)

    print("Verifying assembled bundle...")
    verify_bundle(args.target_dir)

    if args.deploy:
        deploy_pages(args.target_dir)


if __name__ == "__main__":
    main()
