"""Verification checker for the public GitHub Pages bundle.

Validates that:
1. The public bundle contains clean baux black-box optimization documentation.
2. No Overview UI or internal verification monograph text is present.
3. Obsolete AST symbol engines (site_data.js, explorer_engine.js) are excluded.
4. All asset links resolve and JavaScript parses cleanly.
"""

from __future__ import annotations

import base64
import hashlib
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

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

    content_lower = content.lower()
    if "<title>baux" not in content_lower and "baux" not in content_lower:
        raise RuntimeError("index.html missing title 'baux'")
    if "a black box optimization library" not in content_lower:
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
        "simulation.html",
        "style.css",
        "katex.min.css",
        "forester.js",
        "forest.json",
        ".nojekyll",
        "assets/baux_ui.js",
    ]
    for rel in required_files:
        p = target_dir / rel
        if not p.is_file():
            raise RuntimeError(f"Missing required bundle file: {rel}")
        if p.stat().st_size == 0 and rel != ".nojekyll":
            raise RuntimeError(f"Bundle file is empty: {rel}")

    # Ensure obsolete AST engines, duplicate /forest/ mirrors, and build-time artifacts are excluded
    forbidden_paths = [
        "forest",
        "node_modules",
        "javascript-source",
        "package.json",
        "package-lock.json",
        "bundle-js.sh",
        "baux_api_data.json",
        "assets/site_style.css",
        "assets/site_data.js",
        "assets/site_manifest.json",
        "assets/blueprint_engine.js",
        "assets/explorer_engine.js",
    ]
    for rel in forbidden_paths:
        p = target_dir / rel
        if p.exists():
            raise RuntimeError(f"Obsolete or build-only path present in bundle: {rel}")

    print(
        "✓ Link and asset integrity: All required runtime assets exist and build/legacy bloat excluded."
    )


_FORBIDDEN_FILE_SUFFIXES = {
    ".map",
    ".env",
    ".pem",
    ".key",
    ".sqlite",
    ".xml",
    ".xsl",
    ".sh",
    ".ml",
    ".toml",
}
_TEXT_FILE_SUFFIXES = {
    ".html",
    ".htm",
    ".js",
    ".mjs",
    ".cjs",
    ".css",
    ".json",
    ".md",
    ".xml",
    ".xsl",
    ".svg",
    ".txt",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
_VCS_DIRS = {".jj", ".git"}


def _sri_sha384(payload: bytes) -> str:
    digest = base64.b64encode(hashlib.sha384(payload).digest()).decode("ascii")
    return f"sha384-{digest}"


def _validate_csp(rel_html: Path, csp_contents: list[str]) -> list[str]:
    if not csp_contents:
        return [f'{rel_html}: missing <meta http-equiv="Content-Security-Policy">']

    errors: list[str] = []
    for csp in csp_contents:
        directives: dict[str, list[str]] = {}
        for raw_dir in csp.split(";"):
            parts = raw_dir.strip().split()
            if not parts:
                continue
            directives.setdefault(parts[0].lower(), []).extend(parts[1:])

        default_vals = [v.lower() for v in directives.get("default-src", [])]
        if default_vals != ["'none'"]:
            errors.append(
                f"{rel_html}: CSP must set default-src 'none' (got {directives.get('default-src')!r})"
            )

        script_vals = [v.lower() for v in directives.get("script-src", [])]
        if "'self'" not in script_vals:
            errors.append(
                f"{rel_html}: CSP script-src must include 'self' (got {directives.get('script-src')!r})"
            )
        if "'unsafe-inline'" in script_vals:
            errors.append(f"{rel_html}: CSP script-src forbids 'unsafe-inline'")
        if "'unsafe-eval'" in script_vals:
            errors.append(f"{rel_html}: CSP script-src forbids 'unsafe-eval'")

        base_vals = [v.lower() for v in directives.get("base-uri", [])]
        if not base_vals or not set(base_vals).issubset({"'self'", "'none'"}):
            errors.append(
                f"{rel_html}: CSP must include base-uri ('self' or 'none', got {directives.get('base-uri')!r})"
            )

        form_vals = [v.lower() for v in directives.get("form-action", [])]
        if form_vals != ["'none'"]:
            errors.append(
                f"{rel_html}: CSP must set form-action 'none' (got {directives.get('form-action')!r})"
            )

    return errors


class _SecurityHTMLParser(HTMLParser):
    def __init__(self, rel_html: Path, html_path: Path, target_root: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.rel_html = rel_html
        self.html_path = html_path
        self.target_root = target_root
        self.violations: list[str] = []
        self.csp_contents: list[str] = []
        self._in_script = False
        self._script_chunks: list[str] = []
        self._script_line = 0

    def _verify_sri(
        self, tag_name: str, raw_ref: str, integrity: str, line: int
    ) -> None:
        tokens = integrity.split()
        sha384_tokens = [
            t for t in tokens if t.startswith("sha384-") and len(t) > len("sha384-")
        ]
        if not sha384_tokens:
            self.violations.append(
                f"{self.rel_html}:{line}: <{tag_name}> referencing {raw_ref!r} missing non-empty sha384-... integrity attribute"
            )
            return

        parsed = urlsplit(raw_ref)
        if parsed.scheme or parsed.netloc:
            self.violations.append(
                f"{self.rel_html}:{line}: <{tag_name}> references non-local asset {raw_ref!r}"
            )
            return

        clean_path = unquote(parsed.path)
        if not clean_path:
            self.violations.append(
                f"{self.rel_html}:{line}: <{tag_name}> has empty asset path in {raw_ref!r}"
            )
            return

        if clean_path.startswith("/"):
            asset_path = (self.target_root / clean_path.lstrip("/")).resolve()
        else:
            asset_path = (self.html_path.parent / clean_path).resolve()

        if not asset_path.is_relative_to(self.target_root):
            self.violations.append(
                f"{self.rel_html}:{line}: <{tag_name}> asset path escapes bundle root: {raw_ref!r}"
            )
            return
        if not asset_path.is_file():
            self.violations.append(
                f"{self.rel_html}:{line}: <{tag_name}> local asset not found: {raw_ref!r}"
            )
            return

        expected = _sri_sha384(asset_path.read_bytes())
        if any(tok != expected for tok in sha384_tokens):
            self.violations.append(
                f"{self.rel_html}:{line}: SRI sha384 mismatch for {raw_ref!r} (expected {expected}, got {integrity!r})"
            )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        line = self.getpos()[0]
        tag_lower = tag.lower()
        attr_map: dict[str, str] = {}
        for k, v in attrs:
            key = k.lower()
            if key not in attr_map:
                attr_map[key] = v if v is not None else ""
            if key.startswith("on"):
                self.violations.append(
                    f"{self.rel_html}:{line}: forbidden inline event handler attribute {k!r} on <{tag_lower}>"
                )
            if key in ("href", "src") and v is not None:
                val_norm = re.sub(r"[\x00-\x20]+", "", v).lower()
                if val_norm.startswith(("javascript:", "vbscript:", "file:", "blob:")):
                    self.violations.append(
                        f"{self.rel_html}:{line}: forbidden URL scheme in {k}={v!r}"
                    )
                elif val_norm.startswith("data:"):
                    allowed_raster_data = (
                        tag_lower == "img"
                        and key == "src"
                        and val_norm.startswith(
                            (
                                "data:image/png;",
                                "data:image/jpeg;",
                                "data:image/webp;",
                                "data:image/gif;",
                            )
                        )
                    )
                    if not allowed_raster_data:
                        self.violations.append(
                            f"{self.rel_html}:{line}: forbidden data: URI in {k}={v!r}"
                        )
                elif val_norm.startswith("http://"):
                    self.violations.append(
                        f"{self.rel_html}:{line}: cleartext http:// URL in {k}={v!r}"
                    )

        if tag_lower == "a" and attr_map.get("target", "").strip().lower() == "_blank":
            rel_tokens = set(attr_map.get("rel", "").lower().split())
            if not {"noopener", "noreferrer"}.issubset(rel_tokens):
                self.violations.append(
                    f'{self.rel_html}:{line}: <a target="_blank"> must include both noopener and noreferrer in rel (got {attr_map.get("rel", "")!r})'
                )

        if (
            tag_lower == "meta"
            and attr_map.get("http-equiv", "").strip().lower()
            == "content-security-policy"
        ):
            self.csp_contents.append(attr_map.get("content", ""))

        if tag_lower == "script":
            self._in_script = True
            self._script_chunks = []
            self._script_line = line
            src = attr_map.get("src", "").strip()
            if "src" not in attr_map or not src:
                self.violations.append(
                    f"{self.rel_html}:{line}: <script> tag lacks non-empty src attribute"
                )
            else:
                self._verify_sri(
                    "script", src, attr_map.get("integrity", "").strip(), line
                )

        if tag_lower == "link":
            rel_tokens_list = attr_map.get("rel", "").lower().split()
            if "stylesheet" in rel_tokens_list:
                href = attr_map.get("href", "").strip()
                if not href:
                    self.violations.append(
                        f'{self.rel_html}:{line}: <link rel="stylesheet"> missing href'
                    )
                else:
                    self._verify_sri(
                        "link", href, attr_map.get("integrity", "").strip(), line
                    )

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._script_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_script:
            if "".join(self._script_chunks).strip():
                self.violations.append(
                    f"{self.rel_html}:{self._script_line}: <script> contains inline executable script content"
                )
            self._in_script = False
            self._script_chunks = []

    def close(self) -> None:
        super().close()
        if self._in_script:
            if "".join(self._script_chunks).strip():
                self.violations.append(
                    f"{self.rel_html}:{self._script_line}: unclosed <script> contains inline executable script content"
                )
            self._in_script = False


def check_security(target_dir: Path) -> None:
    target_root = target_dir.resolve()
    violations: list[str] = []
    html_files: list[Path] = []
    local_user_markers = ("/Users/", "/home/runner/")

    for path in sorted(target_dir.rglob("*")):
        rel_path = path.relative_to(target_dir)
        if _VCS_DIRS.intersection(rel_path.parts):
            continue
        if not path.is_file():
            continue

        name_lower = path.name.lower()
        suffix_lower = path.suffix.lower()
        if (
            suffix_lower in _FORBIDDEN_FILE_SUFFIXES
            or name_lower == ".env"
            or name_lower.startswith(".env.")
        ):
            violations.append(f"{rel_path}: forbidden file type in public bundle")

        raw = path.read_bytes()
        text: str | None = None
        if suffix_lower in _TEXT_FILE_SUFFIXES:
            text = raw.decode("utf-8", errors="replace")
        elif b"\x00" not in raw:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = None

        if text is not None:
            for marker in local_user_markers:
                if marker in text:
                    violations.append(
                        f"{rel_path}: leaks local build-machine path ({marker}...)"
                    )

        if suffix_lower == ".html":
            html_files.append(path)

    if not html_files:
        raise RuntimeError(f"No HTML files found in bundle directory: {target_dir}")

    for html_path in html_files:
        rel_html = html_path.relative_to(target_dir)
        html_text = html_path.read_text(encoding="utf-8", errors="replace")
        parser = _SecurityHTMLParser(rel_html, html_path.resolve(), target_root)
        parser.feed(html_text)
        parser.close()
        violations.extend(_validate_csp(rel_html, parser.csp_contents))
        violations.extend(parser.violations)

    if violations:
        raise RuntimeError(
            "Security verification failed for public bundle:\n  "
            + "\n  ".join(violations)
        )

    print(
        "✓ Security gates: CSP, zero inline scripts, SHA-384 SRI, link/scheme hygiene, and secret hygiene verified."
    )


def check_smoke(target_dir: Path) -> None:
    # 1. Syntax check with node -c
    js_files = list((target_dir / "assets").glob("*.js"))
    if (target_dir / "forester.js").is_file():
        js_files.append(target_dir / "forester.js")
    for js in js_files:
        subprocess.check_call(["node", "-c", str(js)])

    print("✓ JavaScript syntax: All assets pass node syntax verification.")


def verify_bundle(target_dir: Path) -> None:
    print(f"--- Verifying Public Bundle at: {target_dir} ---")
    check_views(target_dir)
    check_sensitive(target_dir)
    check_assets(target_dir)
    check_security(target_dir)
    check_smoke(target_dir)
    print("--- All Public Bundle Checks PASSED ---\n")
