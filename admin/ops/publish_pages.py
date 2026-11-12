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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "admin/ops"))
from verify_pages import verify_bundle

_SRI_ASSETS = (
    "style.css",
    "katex.min.css",
    "forester.js",
    "assets/baux_ui.js",
)
_VERSIONED_ASSETS = _SRI_ASSETS
_ASSET_TAG_RE = re.compile(
    r"<(link|script)\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>",
    flags=re.IGNORECASE,
)


def sri_sha384(data: bytes) -> str:
    """Compute a W3C Subresource Integrity sha384-<base64> digest."""
    digest = base64.b64encode(hashlib.sha384(data).digest()).decode("ascii")
    return f"sha384-{digest}"


def stamp_html(
    html_text: str,
    sri_map: dict[str, str],
    version_map: dict[str, str] | None = None,
) -> str:
    """Idempotently stamp ?v=<sha256> cache-busters and integrity="sha384-..." on local tags."""
    versions = version_map or {}

    def rewrite_tag(match: re.Match[str]) -> str:
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

    return _ASSET_TAG_RE.sub(rewrite_tag, html_text)


def stamp_sri(bundle_dir: Path, fallback_dir: Path | None = None) -> None:
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
            sri_map[rel_asset] = sri_sha384(payload)
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
        new_txt = stamp_html(txt, sri_map, version_map)
        if new_txt != txt:
            html_file.write_text(new_txt, encoding="utf-8")


def assemble_bundle(dest_dir: Path) -> None:
    """Delegate static site & Playground bundle assembly to `./baux pages build`."""
    subprocess.check_call(
        [str(ROOT / "baux"), "pages", "build", "--bundle-dir", str(dest_dir)],
        cwd=str(ROOT),
    )


def deploy_pages() -> None:
    """Delegate verified bundle build and gh-pages push to `./baux pages deploy`."""
    subprocess.check_call(
        [str(ROOT / "baux"), "pages", "deploy"],
        cwd=str(ROOT),
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
