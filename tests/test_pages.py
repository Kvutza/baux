"""Unit test for two-view GitHub Pages bundle verification and generation."""

import re
import shutil
import tempfile
from pathlib import Path

from admin.ops.publish_pages import assemble_bundle, stamp_sri
from admin.ops.verify_pages import check_security, verify_bundle

ROOT = Path(__file__).resolve().parent.parent

_VALID_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
    "base-uri 'self'; form-action 'none'"
)


def test_bundle():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        # Seed a stale file to verify assemble_bundle wipes non-runtime leftovers
        (dest / "stale.xml").write_text("<stale/>", encoding="utf-8")
        assemble_bundle(dest)
        verify_bundle(dest)
        assert not (dest / "forest").exists()
        assert not list(dest.rglob("*.xml"))
        assert not list(dest.rglob("*.xsl"))


def test_snippets():
    import html

    index_path = ROOT / "forest/output/index.html"
    if not index_path.is_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            assemble_bundle(Path(tmpdir))
    index_html = index_path.read_text(encoding="utf-8")
    snippets = [
        html.unescape(s)
        for s in re.findall(r"<pre[^>]*>(.*?)</pre>", index_html, flags=re.DOTALL)
    ]
    python_snippets = [s for s in snippets if "import baux" in s]
    assert len(python_snippets) >= 4
    for snippet in python_snippets:
        ns: dict = {}
        exec(snippet, ns, ns)

    docs_path = ROOT / "forest/output/baux-0007.html"
    if not docs_path.is_file():
        docs_path = ROOT / ".cache/pages-bundle/docs.html"
    if not docs_path.is_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            assemble_bundle(bundle_dir)
            cand = bundle_dir / "docs.html"
            if not cand.is_file():
                cand = bundle_dir / "baux-0007.html"
            docs_html = cand.read_text(encoding="utf-8")
    else:
        docs_html = docs_path.read_text(encoding="utf-8")

    docs_snippets = [
        html.unescape(s)
        for s in re.findall(r"<pre[^>]*>(.*?)</pre>", docs_html, flags=re.DOTALL)
        if "import baux" in s
    ]
    assert len(docs_snippets) >= 6
    for snippet in docs_snippets:
        ns: dict = {}
        exec(snippet, ns, ns)


def write_synthetic(dest: Path) -> None:
    (dest / "assets").mkdir(parents=True, exist_ok=True)
    (dest / "style.css").write_text("body { margin: 0; }\n", encoding="utf-8")
    (dest / "katex.min.css").write_text(
        ".katex { font-size: 1em; }\n", encoding="utf-8"
    )
    (dest / "forester.js").write_text("export const ok = true;\n", encoding="utf-8")
    (dest / "assets/baux_ui.js").write_text("console.log('baux');\n", encoding="utf-8")
    (dest / "index.html").write_text(
        f"""<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta http-equiv="Content-Security-Policy" content="{_VALID_CSP}" />
  <link rel="stylesheet" href="style.css"></link>
  <link rel="stylesheet" href="katex.min.css" />
  <script type="module" src="forester.js"></script>
  <script type="text/javascript" src="assets/baux_ui.js" defer="defer"></script>
  <title>Baux</title>
</head>
<body>
  <a href="https://github.com/Kvutza/baux" target="_blank" rel="noopener noreferrer">GitHub</a>
</body>
</html>
""",
        encoding="utf-8",
    )
    stamp_sri(dest)


def assert_fails(bundle_dir: Path, expected_substr: str) -> None:
    try:
        check_security(bundle_dir)
    except RuntimeError as exc:
        assert expected_substr.lower() in str(exc).lower(), (
            f"Expected {expected_substr!r} in error message, got: {exc}"
        )
    else:
        raise AssertionError(
            f"Expected check_security() to fail with {expected_substr!r}, but it passed"
        )


def test_security():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "valid"
        write_synthetic(base)

        # Verify SRI + ?v= cache-buster stamping is idempotent across all 4 local assets
        stamped_once = (base / "index.html").read_text(encoding="utf-8")
        assert 'integrity="sha384-' in stamped_once
        for asset_name in (
            "style.css?v=",
            "katex.min.css?v=",
            "forester.js?v=",
            "assets/baux_ui.js?v=",
        ):
            assert asset_name in stamped_once, (
                f"Expected ?v= cache-buster on {asset_name}"
            )
        stamp_sri(base)
        assert (base / "index.html").read_text(encoding="utf-8") == stamped_once
        check_security(base)

        def fresh_case(name: str) -> Path:
            case_dir = Path(tmpdir) / name
            shutil.copytree(base, case_dir)
            return case_dir

        # (i) Missing CSP
        c1 = fresh_case("missing_csp")
        html1 = (c1 / "index.html").read_text(encoding="utf-8")
        (c1 / "index.html").write_text(
            re.sub(r"<meta[^>]*Content-Security-Policy[^>]*>", "", html1),
            encoding="utf-8",
        )
        assert_fails(c1, "Content-Security-Policy")

        # (ii) 'unsafe-inline' in script-src (and 'unsafe-eval')
        c2 = fresh_case("unsafe_inline_csp")
        html2 = (c2 / "index.html").read_text(encoding="utf-8")
        (c2 / "index.html").write_text(
            html2.replace("script-src 'self'", "script-src 'self' 'unsafe-inline'"),
            encoding="utf-8",
        )
        assert_fails(c2, "unsafe-inline")

        c2b = fresh_case("unsafe_eval_csp")
        (c2b / "index.html").write_text(
            html2.replace("script-src 'self'", "script-src 'self' 'unsafe-eval'"),
            encoding="utf-8",
        )
        assert_fails(c2b, "unsafe-eval")

        # (iii) Inline <script>alert(1)</script>
        c3 = fresh_case("inline_script")
        html3 = (c3 / "index.html").read_text(encoding="utf-8")
        (c3 / "index.html").write_text(
            html3.replace("</body>", "<script>alert(1)</script></body>"),
            encoding="utf-8",
        )
        assert_fails(c3, "<script>")

        # (iv) Tampered JS failing SRI sha384 check
        c4 = fresh_case("tampered_js")
        (c4 / "assets/baux_ui.js").write_text(
            "console.log('tampered!');\n", encoding="utf-8"
        )
        assert_fails(c4, "sha384")

        # (v) href="javascript:alert(1)"
        c5 = fresh_case("javascript_href")
        html5 = (c5 / "index.html").read_text(encoding="utf-8")
        (c5 / "index.html").write_text(
            html5.replace("https://github.com/Kvutza/baux", "javascript:alert(1)"),
            encoding="utf-8",
        )
        assert_fails(c5, "javascript:")

        # (vi) target="_blank" missing noreferrer
        c6 = fresh_case("missing_noreferrer")
        html6 = (c6 / "index.html").read_text(encoding="utf-8")
        (c6 / "index.html").write_text(
            html6.replace('rel="noopener noreferrer"', 'rel="noopener"'),
            encoding="utf-8",
        )
        assert_fails(c6, "noreferrer")

        # Secret, build-artifact & filesystem hygiene checks (.map, .env, .xml, .xsl, local path)
        c7 = fresh_case("sourcemap_leak")
        (c7 / "assets/baux_ui.js.map").write_text("{}", encoding="utf-8")
        assert_fails(c7, "forbidden file type")

        c7b = fresh_case("xml_build_leak")
        (c7b / "baux-0001.xml").write_text("<tree/>", encoding="utf-8")
        assert_fails(c7b, "forbidden file type")

        c7c = fresh_case("xsl_build_leak")
        (c7c / "tree.xsl").write_text("<xsl:stylesheet/>", encoding="utf-8")
        assert_fails(c7c, "forbidden file type")

        c8 = fresh_case("dotenv_leak")
        (c8 / ".env").write_text("SECRET=1\n", encoding="utf-8")
        assert_fails(c8, "forbidden file type")

        c9 = fresh_case("local_path_leak")
        (c9 / "assets/leak.txt").write_text(
            "built from " + "/Users/" + "alice/project\n", encoding="utf-8"
        )
        assert_fails(c9, "/Users/")

        c9b = fresh_case("runner_path_leak")
        (c9b / "assets/leak.txt").write_text(
            "built from " + "/home/runner/" + "work/baux\n", encoding="utf-8"
        )
        assert_fails(c9b, "/home/runner/")

        # (vii) Inline DOM event-handler attribute (onclick=...)
        c10 = fresh_case("inline_onclick")
        html10 = (c10 / "index.html").read_text(encoding="utf-8")
        (c10 / "index.html").write_text(
            html10.replace("<body>", '<body onclick="alert(1)">'),
            encoding="utf-8",
        )
        assert_fails(c10, "onclick")

        # (viii) Cleartext http:// link rejection (while xmlns="http://..." stays valid)
        c11 = fresh_case("cleartext_http_link")
        html11 = (c11 / "index.html").read_text(encoding="utf-8")
        (c11 / "index.html").write_text(
            html11.replace(
                "https://github.com/Kvutza/baux", "http://github.com/Kvutza/baux"
            ),
            encoding="utf-8",
        )
        assert_fails(c11, "http://")

        # (ix) data: URI policy: reject data:image/svg+xml, allow raster data:image/png; on <img src>
        c12 = fresh_case("svg_data_uri")
        html12 = (c12 / "index.html").read_text(encoding="utf-8")
        (c12 / "index.html").write_text(
            html12.replace(
                "</body>",
                '<img src="data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=" /></body>',
            ),
            encoding="utf-8",
        )
        assert_fails(c12, "data:")

        c13 = fresh_case("png_data_uri_allowed")
        (c13 / "index.html").write_text(
            html12.replace(
                "</body>",
                '<img src="data:image/png;base64,iVBORw0KGgo=" alt="ok" /></body>',
            ),
            encoding="utf-8",
        )
        check_security(c13)
