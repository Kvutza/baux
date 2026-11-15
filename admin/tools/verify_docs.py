#!/usr/bin/env python3
"""Static AST/PyO3 documentation, anchor, and snippet contract verifier.

Verifies `forest/trees/*.tree` without requiring the native `baux.baux_rust`
shared library or building the HTML bundle.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES_DIR = ROOT / "forest" / "trees"
PYTHON_PKG = ROOT / "python" / "baux"
PYO3_DIR = ROOT / "rust" / "crates" / "baux-py" / "src"

ANCHOR_RE = re.compile(
    r"github\.com/Kvutza/baux/blob/main/"
    r"([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)#L(\d+)(?:-L(\d+))?"
)
ENTRY_RE = re.compile(
    r"\[data-symbol\]\{(baux\.[^}]+)\}\[data-kind\]\{([^}]+)\}(?:\[data-module\]\{([^}]+)\})?.*?\\pre\{(.*?)\n\s*\}",
    flags=re.DOTALL,
)
DECL_LINE_RE = re.compile(
    r"^(?:async\s+def\s+|def\s+|class\s+|pub(?:\([^)]*\))?\s+|fn\s+|struct\s+|enum\s+|type\s+|const\s+|static\s+|trait\s+|impl\b|@|[A-Za-z_][A-Za-z0-9_]*\s*(?::[^=]+)?=\s*)"
)


@dataclass(frozen=True)
class CallableSig:
    """Extracted callable parameter layout from Python AST or PyO3 Rust."""

    source: str
    lineno: int
    params: list[str]
    kwonly: list[str]
    has_default: dict[str, bool] | None
    kind: str  # "py_def" | "dataclass" | "pyo3"


def func_sig(
    source: str, lineno: int, args: ast.arguments, *, drop_self: bool, kind: str
) -> CallableSig:
    pos = args.posonlyargs + args.args
    if drop_self and pos and pos[0].arg in ("self", "cls"):
        pos = pos[1:]
    params = [a.arg for a in pos]
    if args.vararg:
        params.append(args.vararg.arg)
    kwonly = [a.arg for a in args.kwonlyargs]
    params.extend(kwonly)
    if args.kwarg:
        params.append(args.kwarg.arg)

    pos_names = [a.arg for a in pos]
    defaults_map: dict[str, bool] = dict.fromkeys(pos_names, False)
    if args.defaults:
        for arg_name in pos_names[-len(args.defaults) :]:
            defaults_map[arg_name] = True
    for kw_arg, kw_def in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        defaults_map[kw_arg.arg] = kw_def is not None
    return CallableSig(
        source=source,
        lineno=lineno,
        params=params,
        kwonly=kwonly,
        has_default=defaults_map,
        kind=kind,
    )


def is_dataclass(cls: ast.ClassDef) -> bool:
    for dec in cls.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "dataclass"
        ):
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
            return True
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "dataclass"
        ):
            return True
    return False


def dataclass_sig(source: str, cls: ast.ClassDef) -> CallableSig:
    params: list[str] = []
    defaults_map: dict[str, bool] = {}
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if (
                isinstance(stmt.annotation, ast.Subscript)
                and isinstance(stmt.annotation.value, ast.Name)
                and stmt.annotation.value.id == "ClassVar"
            ):
                continue
            name = stmt.target.id
            params.append(name)
            defaults_map[name] = stmt.value is not None
    return CallableSig(
        source=source,
        lineno=cls.lineno,
        params=params,
        kwonly=[],
        has_default=defaults_map,
        kind="dataclass",
    )


def module_name(rel: str) -> str:
    stem = rel.removesuffix(".py")
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.removeprefix("python/").replace("/", ".")


def scan_python() -> tuple[
    dict[str, list[CallableSig]],
    dict[tuple[str, str], list[CallableSig]],
]:
    """Build global and `(module, name)` AST signature maps from `python/baux/**/*.py`."""
    sigs: dict[str, list[CallableSig]] = {}
    mod_sigs: dict[tuple[str, str], list[CallableSig]] = {}
    aliases: list[tuple[str, str, str]] = []

    for path in sorted(PYTHON_PKG.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
        mod = module_name(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = func_sig(
                    rel, node.lineno, node.args, drop_self=False, kind="py_def"
                )
                sigs.setdefault(node.name, []).append(sig)
                mod_sigs.setdefault((mod, node.name), []).append(sig)
            elif isinstance(node, ast.ClassDef):
                init_fn = next(
                    (
                        item
                        for item in node.body
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
                    ),
                    None,
                )
                if init_fn is not None:
                    sig = func_sig(
                        rel, node.lineno, init_fn.args, drop_self=True, kind="py_def"
                    )
                    sigs.setdefault(node.name, []).append(sig)
                    mod_sigs.setdefault((mod, node.name), []).append(sig)
                elif is_dataclass(node):
                    sig = dataclass_sig(rel, node)
                    sigs.setdefault(node.name, []).append(sig)
                    mod_sigs.setdefault((mod, node.name), []).append(sig)
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        msig = func_sig(
                            rel,
                            item.lineno,
                            item.args,
                            drop_self=True,
                            kind="py_def",
                        )
                        sigs.setdefault(f"{node.name}.{item.name}", []).append(msig)
                        mod_sigs.setdefault(
                            (mod, f"{node.name}.{item.name}"), []
                        ).append(msig)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Name)
            ):
                aliases.append((mod, node.targets[0].id, node.value.id))

    for mod, alias_name, target in aliases:
        target_sigs = mod_sigs.get((mod, target)) or sigs.get(target)
        if target_sigs:
            mod_sigs.setdefault((mod, alias_name), list(target_sigs))
            sigs.setdefault(alias_name, list(target_sigs))
    return sigs, mod_sigs


def pyo3_tokens(
    sig_text: str,
) -> tuple[list[str], list[str], dict[str, bool]]:
    params: list[str] = []
    kwonly: list[str] = []
    has_default: dict[str, bool] = {}
    after_star = False
    for raw in sig_text.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok == "*":
            after_star = True
            continue
        if tok.startswith("**"):
            name = tok[2:].strip()
            params.append(name)
            has_default[name] = False
            continue
        if tok.startswith("*"):
            name = tok[1:].strip()
            params.append(name)
            has_default[name] = False
            after_star = True
            continue
        name, has_eq, _ = tok.partition("=")
        name = name.strip()
        params.append(name)
        if after_star:
            kwonly.append(name)
        has_default[name] = bool(has_eq)
    return params, kwonly, has_default


def scan_pyo3() -> dict[str, list[CallableSig]]:
    """Extract `#[pyfunction]` and `#[pyclass]` `#[new]` signatures from `baux-py`."""
    sigs: dict[str, list[CallableSig]] = {}
    fn_re = re.compile(
        r'#\[pyfunction(?:\(\s*name\s*=\s*"([^"]+)"\s*\))?\](?:\s*#\[pyo3\(\s*signature\s*=\s*\(([^)]*)\)\s*\)\])?\s*pub\s+fn\s+(\w+)\s*\((.*?)\)\s*(?:->|\{)',
        flags=re.DOTALL,
    )
    for path in sorted(PYO3_DIR.rglob("*.rs")):
        rel = str(path.relative_to(ROOT)).replace(os.sep, "/")
        text = path.read_text(encoding="utf-8")
        for m in fn_re.finditer(text):
            py_name = m.group(1) or m.group(3)
            sig_attr = m.group(2)
            rust_args = m.group(4)
            lineno = text[: m.start()].count("\n") + 1
            if sig_attr is not None:
                params, kwonly, has_default = pyo3_tokens(sig_attr)
                sigs.setdefault(py_name, []).append(
                    CallableSig(rel, lineno, params, kwonly, has_default, "pyo3")
                )
            else:
                params = []
                for arg in rust_args.split(","):
                    a = arg.strip()
                    if not a or ":" not in a:
                        continue
                    pname, pty = (part.strip() for part in a.split(":", 1))
                    if pty.startswith("Python<") or pname == "py":
                        continue
                    params.append(pname.lstrip("_"))
                sigs.setdefault(py_name, []).append(
                    CallableSig(rel, lineno, params, [], None, "pyo3")
                )
    return sigs


def parse_pre(sym: str, pre: str) -> tuple[list[str], list[str], dict[str, bool]]:
    short = sym.split(".", 1)[1]
    decl = pre.replace(f"baux.{short}", "def documented", 1)
    fn_node = ast.parse(f"{decl}:\n    pass").body[0]
    assert isinstance(fn_node, ast.FunctionDef)
    parsed = func_sig("doc", 1, fn_node.args, drop_self=False, kind="doc")
    assert parsed.has_default is not None
    return parsed.params, parsed.kwonly, parsed.has_default


def pre_blocks(text: str) -> list[tuple[int, str]]:
    r"""Extract brace-balanced `\pre{...}` blocks as `(start_line, body)`."""
    blocks: list[tuple[int, str]] = []
    idx = 0
    marker = "\\pre{"
    while True:
        pos = text.find(marker, idx)
        if pos < 0:
            break
        start = pos + len(marker)
        depth = 1
        cur = start
        while cur < len(text) and depth > 0:
            ch = text[cur]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cur += 1
        if depth == 0:
            lineno = text[:pos].count("\n") + 1
            blocks.append((lineno, text[start : cur - 1]))
            idx = cur
        else:
            lineno = text[:pos].count("\n") + 1
            blocks.append((lineno, text[start:]))
            break
    return blocks


def check_docs() -> tuple[int, int, int, int]:
    """Run symbol, signature, anchor, and snippet checks over `forest/trees`."""
    py_sigs, mod_sigs = scan_python()
    pyo3_sigs = scan_pyo3()

    pages = [TREES_DIR / "baux-0007.tree", *sorted(TREES_DIR.glob("api-*.tree"))]
    all_trees = sorted(TREES_DIR.glob("*.tree"))
    ref_content = "\n".join(p.read_text(encoding="utf-8") for p in pages)

    errors: list[str] = []

    # 1. Exact API reference symbol table invariants (mirrors tests/test_pages.py).
    symbols = re.findall(r"\[data-symbol\]\{(baux\.[^}]+)\}", ref_content)
    if len(symbols) != len(set(symbols)) or len(symbols) != 85:
        errors.append(
            f"expected 85 unique [data-symbol] entries across API trees, found {len(symbols)} ({len(set(symbols))} unique)"
        )
    for symbol in symbols:
        if f"[id]{{{symbol}}}" not in ref_content:
            errors.append(f"missing [id]{{{symbol}}} target for API entry")
        if f".html#{symbol})" not in ref_content:
            errors.append(f"missing .html#{symbol}) self/cross link for API entry")

    # 2. Static signature verification for documented callables.
    sigs_checked = 0
    for page in pages:
        content = page.read_text(encoding="utf-8")
        for m in ENTRY_RE.finditer(content):
            sym, kind, mod_name, pre = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4).strip(),
            )
            short = sym.split(".", 1)[1]
            if pre.startswith("class ") or f"baux.{short}(" not in pre:
                continue
            try:
                doc_params, doc_kwonly, doc_defaults = parse_pre(sym, pre)
            except SyntaxError as err:
                errors.append(f"{page.name} {sym}: unparseable \\pre signature ({err})")
                continue
            mod_cands = mod_sigs.get((mod_name, short), []) if mod_name else []
            candidates = mod_cands or (
                py_sigs.get(short, []) + pyo3_sigs.get(short, [])
            )
            if not candidates:
                continue
            matched = any(
                cand.params == doc_params
                and cand.kwonly == doc_kwonly
                and (cand.has_default is None or cand.has_default == doc_defaults)
                for cand in candidates
            )
            if not matched and kind in ("function", "class"):
                rendered = [
                    f"{c.source}:{c.lineno}({', '.join(c.params)})" for c in candidates
                ]
                errors.append(
                    f"{page.name} {sym}: documented params {doc_params} (kw={doc_kwonly}) do not match code {', '.join(rendered)}"
                )
            sigs_checked += 1

    # 3. GitHub `#L...` line anchor validity across every tree.
    anchors_checked = 0
    for tree in all_trees:
        lines = tree.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            for m in ANCHOR_RE.finditer(line):
                anchors_checked += 1
                rel, l1_str, l2_str = m.group(1), m.group(2), m.group(3)
                l1 = int(l1_str)
                l2 = int(l2_str) if l2_str else l1
                target = ROOT / rel
                if not target.is_file():
                    errors.append(f"{tree.name}:{idx}: missing anchor file {rel}")
                    continue
                src_lines = target.read_text(encoding="utf-8").splitlines()
                if l1 < 1 or l2 < l1 or l2 > len(src_lines):
                    errors.append(
                        f"{tree.name}:{idx}: anchor {rel}#L{l1_str} out of bounds (len={len(src_lines)})"
                    )
                    continue
                if not DECL_LINE_RE.match(src_lines[l1 - 1].strip()):
                    errors.append(
                        f"{tree.name}:{idx}: anchor {rel}#L{l1} points to non-declaration line {src_lines[l1 - 1].strip()!r}"
                    )

    # 4. Syntax verification for inline Python code snippets across every tree.
    snippets_checked = 0
    for tree in all_trees:
        text = tree.read_text(encoding="utf-8")
        for lineno, body in pre_blocks(text):
            snippet = body.strip()
            if "import baux" not in snippet:
                continue
            snippets_checked += 1
            try:
                ast.parse(snippet, filename=f"{tree.name}:{lineno}")
            except SyntaxError as err:
                errors.append(
                    f"{tree.name}:{lineno}: Python snippet syntax error: {err}"
                )

    if errors:
        raise ValueError(
            "Documentation / API contract violation(s):\n  " + "\n  ".join(errors)
        )
    return len(symbols), sigs_checked, anchors_checked, snippets_checked


def main() -> int:
    try:
        symbols, sigs, anchors, snippets = check_docs()
    except ValueError as err:
        sys.stderr.write(f"error: {err}\n")
        return 1
    print(
        f"[ok] Docs/AST contracts: symbols={symbols}, signatures={sigs}, anchors={anchors}, snippets={snippets}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
