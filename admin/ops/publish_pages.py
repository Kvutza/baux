"""Build, verify, and publish the clean baux documentation and guide bundle.

Generates a pure runtime static bundle from tracked `forest/` and `admin/ui/`:
- Compiles Forester trees (`forest/trees/*.tree`) + XSLT theme (`forest/theme/*.xsl`) to static HTML.
- Compiles the OCaml frontend (`admin/ui/baux_ui.ml` -> `assets/baux_ui.js`).
- Copies strictly runtime web assets (zero `.xml`, `.xsl`, npm manifests, or `/forest/` duplicates).
- Stamps SHA-256 `?v=` cache-busters and SHA-384 Subresource Integrity digests.
- Validates the assembled bundle via `admin/ops/verify_pages.py`.
- Optionally deploys to `origin/gh-pages` via a transient, auto-cleaned JJ workspace.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOREST_DIR = ROOT / "forest"
FOREST_OUT = FOREST_DIR / "output"
sys.path.insert(0, str(ROOT / "admin/ops"))
from verify_pages import verify_bundle

_SRI_ASSETS = (
    "style.css",
    "katex.min.css",
    "forester.js",
    "assets/baux_ui.js",
)
_VERSIONED_ASSETS = _SRI_ASSETS
_RUNTIME_ROOT_FILES = (
    "style.css",
    "katex.min.css",
    "forester.js",
    "forest.json",
    "favicon.ico",
    "favicon-16x16.png",
    "favicon-32x32.png",
    "apple-touch-icon.png",
    "android-chrome-192x192.png",
    "android-chrome-512x512.png",
)
_RUNTIME_ROOT_DIRS = (
    "fonts",
    "LICENSES",
)
_ASSET_TAG_RE = re.compile(
    r"<(link|script)\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>",
    flags=re.IGNORECASE,
)


def compute_sri_sha384(data: bytes) -> str:
    """Compute a W3C Subresource Integrity sha384-<base64> digest."""
    digest = base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")
    return f"sha384-{digest}"


def stamp_html_sri_and_versions(
    html_text: str,
    sri_map: dict[str, str],
    version_map: dict[str, str] | None = None,
) -> str:
    """Idempotently stamp ?v=<sha256> cache-busters and integrity="sha384-..." on local tags."""
    versions = version_map or {}

    def _rewrite_tag(match: re.Match[str]) -> str:
        tag_str = match.group(0)
        tag_name = match.group(1).lower()

        if tag_str.endswith("/>"):
            prefix = tag_str[:-2]
            close_tok = " />" if prefix.endswith(" ") else "/>"
            core = prefix.rstrip()
        else:
            close_tok = ">"
            core = tag_str[:-1].rstrip()

        if tag_name == "link":
            rel_m = re.search(
                r"""\brel\s*=\s*(?:"([^"]*)"|'([^']*)')""",
                core,
                flags=re.IGNORECASE,
            )
            if not rel_m:
                return tag_str
            rel_val = rel_m.group(1) if rel_m.group(1) is not None else rel_m.group(2)
            if "stylesheet" not in rel_val.lower().split():
                return tag_str
            attr_name = "href"
        else:
            attr_name = "src"

        url_pattern = rf"""(\b{attr_name}\s*=\s*)(?:"([^"]*)"|'([^']*)')"""
        url_m = re.search(url_pattern, core, flags=re.IGNORECASE)
        if not url_m:
            return tag_str

        raw_url = url_m.group(2) if url_m.group(2) is not None else url_m.group(3)
        base_url = raw_url.split("#", 1)[0].split("?", 1)[0]
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", base_url) or base_url.startswith(
            "//"
        ):
            return tag_str

        clean_rel = re.sub(r"^(?:\./|\.\./|/)+", "", base_url)
        if clean_rel not in sri_map and clean_rel not in versions:
            return tag_str

        if clean_rel in versions:
            new_url = f"{base_url}?v={versions[clean_rel]}"
            core = (
                core[: url_m.start()]
                + f'{url_m.group(1)}"{new_url}"'
                + core[url_m.end() :]
            )

        if clean_rel in sri_map:
            core = re.sub(
                r"""(?:\s+|(?<=["']))integrity\s*=\s*(?:"[^"]*"|'[^']*')""",
                "",
                core,
                flags=re.IGNORECASE,
            )
            core = f'{core} integrity="{sri_map[clean_rel]}"'

        return f"{core}{close_tok}"

    return _ASSET_TAG_RE.sub(_rewrite_tag, html_text)


def stamp_bundle_sri(bundle_dir: Path, fallback_dir: Path | None = None) -> None:
    """Compute SHA-256 ?v= and SHA-384 SRI digests for local assets and stamp all HTML files."""
    if not bundle_dir.exists():
        return

    sri_map: dict[str, str] = {}
    version_map: dict[str, str] = {}
    search_roots = [bundle_dir]
    if fallback_dir is not None and fallback_dir != bundle_dir:
        search_roots.append(fallback_dir)

    for rel_asset in _SRI_ASSETS:
        payload: bytes | None = None
        for root in search_roots:
            candidate = root / rel_asset
            if candidate.is_file():
                payload = candidate.read_bytes()
                break
        if payload is not None:
            sri_map[rel_asset] = compute_sri_sha384(payload)
            if rel_asset in _VERSIONED_ASSETS:
                version_map[rel_asset] = (
                    hashlib.sha256(payload).hexdigest()[:12] if payload else "1"
                )
        elif rel_asset in _VERSIONED_ASSETS:
            version_map[rel_asset] = "1"

    skip_parts = {".jj", ".git", "node_modules"}
    for html_file in bundle_dir.rglob("*.html"):
        rel_parts = html_file.relative_to(bundle_dir).parts
        if skip_parts.intersection(rel_parts):
            continue
        txt = html_file.read_text(encoding="utf-8", errors="replace")
        new_txt = stamp_html_sri_and_versions(txt, sri_map, version_map)
        if new_txt != txt:
            html_file.write_text(new_txt, encoding="utf-8")


def build_forest() -> None:
    forest_toml = FOREST_DIR / "forest.toml"
    if not forest_toml.is_file():
        raise FileNotFoundError(f"Missing Forester manifest: {forest_toml}")

    print("Building documentation forest via Forester (OCaml)...")
    if FOREST_OUT.exists():
        shutil.rmtree(FOREST_OUT)

    subprocess.check_call(
        [
            "opam",
            "exec",
            "--switch=forester",
            "--",
            "forester",
            "build",
            "forest.toml",
        ],
        cwd=str(FOREST_DIR),
    )

    theme_dir = FOREST_DIR / "theme"
    for ext in ("*.xsl", "*.css", "*.js"):
        for asset in theme_dir.glob(ext):
            shutil.copy2(asset, FOREST_OUT / asset.name)

    for xml in sorted(FOREST_OUT.glob("*.xml")):
        if xml.name.startswith("unstable-"):
            continue
        html = xml.with_suffix(".html")
        with open(html, "w", encoding="utf-8") as out:
            subprocess.check_call(
                ["xsltproc", "default.xsl", xml.name],
                cwd=str(FOREST_OUT),
                stdout=out,
            )

    route_aliases = (
        ("baux-0001.html", "index.html"),
        ("baux-0007.html", "docs.html"),
        ("baux-0004.html", "simulation.html"),
    )
    for src_name, dst_name in route_aliases:
        src_html = FOREST_OUT / src_name
        if not src_html.is_file():
            raise FileNotFoundError(f"Expected compiled tree HTML missing: {src_html}")
        shutil.copyfile(src_html, FOREST_OUT / dst_name)

    print("✓ Compiled Forester trees to static HTML in forest/output")


def build_ui() -> None:
    ui_src = ROOT / "admin/ui/baux_ui.ml"
    if not ui_src.is_file():
        raise FileNotFoundError(f"Missing OCaml UI source: {ui_src}")

    print("Building OCaml frontend (baux_ui.js) with dune and js_of_ocaml...")
    subprocess.check_call(
        [
            "opam",
            "exec",
            "--switch=5.2.0+ox",
            "--",
            "dune",
            "build",
            "./baux_ui.js",
        ],
        cwd=str(ROOT / "admin/ui"),
    )
    built_js = ROOT / "admin/ui/_build/default/baux_ui.js"
    if not built_js.is_file():
        raise FileNotFoundError(f"Compiled OCaml bundle not found: {built_js}")

    payload = built_js.read_bytes()
    forest_assets = FOREST_OUT / "assets"
    forest_assets.mkdir(parents=True, exist_ok=True)
    forest_ui = forest_assets / "baux_ui.js"
    forest_ui.unlink(missing_ok=True)
    forest_ui.write_bytes(payload)
    forest_ui.chmod(0o644)
    print("✓ Compiled baux_ui.js into forest/output/assets/")


def _clean_bundle_dir(dest_dir: Path) -> None:
    """Remove all existing files in dest_dir while preserving VCS metadata (.jj / .git)."""
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        return
    for entry in dest_dir.iterdir():
        if entry.name in {".jj", ".git"}:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def assemble_bundle(dest_dir: Path) -> None:
    build_forest()
    build_ui()

    _clean_bundle_dir(dest_dir)
    dest_assets = dest_dir / "assets"
    dest_assets.mkdir(parents=True, exist_ok=True)

    for html_file in sorted(FOREST_OUT.glob("*.html")):
        if html_file.name.startswith("unstable-"):
            continue
        shutil.copy2(html_file, dest_dir / html_file.name)

    for rel_name in _RUNTIME_ROOT_FILES:
        src = FOREST_OUT / rel_name
        if not src.is_file():
            raise FileNotFoundError(f"Missing runtime Forester asset: {src}")
        shutil.copy2(src, dest_dir / rel_name)

    for dir_name in _RUNTIME_ROOT_DIRS:
        src_dir = FOREST_OUT / dir_name
        if src_dir.is_dir():
            shutil.copytree(src_dir, dest_dir / dir_name)

    src_ui = FOREST_OUT / "assets/baux_ui.js"
    dst_ui = dest_assets / "baux_ui.js"
    shutil.copy2(src_ui, dst_ui)
    dst_ui.chmod(0o644)

    (dest_dir / ".nojekyll").touch()

    for html_dir in (FOREST_OUT, dest_dir):
        stamp_bundle_sri(html_dir, fallback_dir=dest_dir)


def deploy_pages() -> None:
    ws_name = "gh-pages-deploy"
    subprocess.run(
        ["jj", "workspace", "forget", ws_name],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    with tempfile.TemporaryDirectory(prefix="baux-gh-pages-") as tmp_str:
        ws_dir = Path(tmp_str) / "ws"
        try:
            subprocess.check_call(
                [
                    "jj",
                    "workspace",
                    "add",
                    str(ws_dir),
                    "--name",
                    ws_name,
                    "-r",
                    "gh-pages",
                ],
                cwd=str(ROOT),
            )
            subprocess.check_call(
                ["jj", "new", "gh-pages"],
                cwd=str(ws_dir),
            )
            print(f"Assembling clean public bundle in transient workspace {ws_dir}...")
            assemble_bundle(ws_dir)
            print("Verifying assembled bundle...")
            verify_bundle(ws_dir)

            print("--- Deploying clean runtime bundle to origin/gh-pages ---")
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
            print("✓ Successfully pushed clean runtime bundle to origin/gh-pages!")
        finally:
            subprocess.run(
                ["jj", "workspace", "forget", ws_name],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT / ".cache/pages-bundle",
        help="Target directory for local bundle assembly (default: .cache/pages-bundle)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Assemble, verify, and deploy to origin/gh-pages via a transient JJ workspace",
    )
    parser.add_argument(
        "--skip-check-local",
        action="store_true",
        help="Deprecated no-op flag retained for CLI compatibility",
    )
    args = parser.parse_args()

    if args.deploy:
        deploy_pages()
        return

    print(f"Assembling public documentation bundle into {args.target_dir}...")
    assemble_bundle(args.target_dir)

    print("Verifying assembled bundle...")
    verify_bundle(args.target_dir)


if __name__ == "__main__":
    main()
