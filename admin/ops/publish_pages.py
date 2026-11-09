"""Build, verify, and publish the clean baux documentation and guide bundle.

Generates the public bundle from docs/site/:
- Generates clean index.html and docs.html.
- Includes compiled OCaml frontend (assets/baux_ui.js).
- Validates bundle via admin/ops/verify_pages.py.
- Deploys cleanly to the isolated gh-pages branch.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS_SITE = ROOT / "docs/site"
sys.path.insert(0, str(ROOT / "admin/ops"))
from verify_pages import verify_bundle


def build_ui() -> None:
    ui_target = DOCS_SITE / "assets/baux_ui.js"
    ui_src = ROOT / "admin/ui/baux_ui.ml"
    if ui_src.exists():
        needs_build = (
            not ui_target.exists() or ui_src.stat().st_mtime > ui_target.stat().st_mtime
        )
        if needs_build:
            print("Building OCaml frontend (baux_ui.js) with dune and js_of_ocaml...")
            try:
                subprocess.check_call(
                    ["opam", "exec", "--", "dune", "build", "./baux_ui.js"],
                    cwd=str(ROOT / "admin/ui"),
                )
                built_js = ROOT / "admin/ui/_build/default/baux_ui.js"
                if built_js.exists():
                    ui_target.unlink(missing_ok=True)
                    shutil.copyfile(built_js, ui_target)
                    ui_target.chmod(0o644)
                    print("✓ Compiled and copied baux_ui.js to docs/site/assets/")
            except Exception as e:
                print(f"Notice: OCaml UI build via dune skipped or failed ({e})")


def assemble_bundle(dest_dir: Path) -> None:
    build_ui()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_assets = dest_dir / "assets"
    dest_assets.mkdir(parents=True, exist_ok=True)

    # Remove obsolete engines if present from prior workspace state
    for obsolete in [
        "site_data.js",
        "site_manifest.json",
        "blueprint_engine.js",
        "explorer_engine.js",
    ]:
        ob_path = dest_assets / obsolete
        if ob_path.exists():
            ob_path.unlink()

    # Copy static assets
    src_ui = DOCS_SITE / "assets/baux_ui.js"
    if src_ui.exists():
        shutil.copy2(src_ui, dest_assets / "baux_ui.js")

    # Sanitize and copy site_style.css
    raw_css = (DOCS_SITE / "assets/site_style.css").read_text(encoding="utf-8")
    clean_css = raw_css.replace("(#view-monograph)", "")
    (dest_assets / "site_style.css").write_text(clean_css, encoding="utf-8")

    # Ensure .nojekyll exists
    (dest_dir / ".nojekyll").touch()

    # Copy index.html and docs.html
    shutil.copy2(DOCS_SITE / "index.html", dest_dir / "index.html")
    shutil.copy2(DOCS_SITE / "docs.html", dest_dir / "docs.html")

    # Preserve logos if present
    for logo in ["logo-light.svg", "logo-dark.svg"]:
        src_logo = DOCS_SITE / "assets" / logo
        if src_logo.exists():
            shutil.copy2(src_logo, dest_assets / logo)


def deploy_pages(ws_dir: Path) -> None:
    print(f"--- Deploying gh-pages from workspace: {ws_dir} ---")
    subprocess.check_call(
        ["jj", "describe", "-m", "kvutza/pages"],
        cwd=str(ws_dir),
    )
    subprocess.check_call(
        ["jj", "bookmark", "set", "gh-pages", "-r", "@"],
        cwd=str(ws_dir),
    )
    subprocess.check_call(
        ["jj", "git", "push", "--bookmark", "gh-pages"],
        cwd=str(ws_dir),
    )
    print("✓ Successfully pushed kvutza/pages to origin/gh-pages!")


def ensure_workspace(ws_dir: Path) -> None:
    """Ensure the target directory is an active jj workspace tracking gh-pages."""
    if not (ws_dir / ".jj").exists():
        if ws_dir.exists():
            shutil.rmtree(ws_dir)
        subprocess.check_call(
            ["jj", "workspace", "add", str(ws_dir), "-r", "gh-pages"],
            cwd=str(ROOT),
        )


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

    if args.deploy:
        ensure_workspace(args.target_dir)

    print(f"Assembling public documentation bundle into {args.target_dir}...")
    assemble_bundle(args.target_dir)

    print("Verifying assembled bundle...")
    verify_bundle(args.target_dir)

    if args.deploy:
        deploy_pages(args.target_dir)


if __name__ == "__main__":
    main()
