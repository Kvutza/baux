"""End-to-end verification of the browser WASM ``baux`` Python backend and asset builder."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from admin.ops import build_repl

REPO_ROOT = Path(__file__).resolve().parents[1]
TREES_DIR = REPO_ROOT / "forest" / "trees"
NATIVE_INSPECT_DIR = Path("/tmp/baux_native_inspect")


def _snippets() -> list[tuple[str, int, str]]:
    snippets: list[tuple[str, int, str]] = []
    for tree in sorted(TREES_DIR.glob("*.tree")):
        text = tree.read_text(encoding="utf-8")
        pos = 0
        idx = 0
        while True:
            start = text.find("\\pre{\n", pos)
            if start < 0:
                break
            cursor = start + len("\\pre{\n")
            depth = 1
            while cursor < len(text) and depth > 0:
                ch = text[cursor]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                cursor += 1
            if depth != 0:
                break
            code = text[start + len("\\pre{\n") : cursor - 1].rstrip()
            pos = cursor
            if (
                ("import baux" in code or "from baux" in code)
                and "..." not in code
                and "your_" not in code
                and "oxcaml_step" not in code
            ):
                snippets.append((tree.name, idx, code))
                idx += 1
    return snippets


def _runzip(tmp_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    unpack = tmp_path / "wasm_pkg"
    unpack.mkdir(parents=True, exist_ok=True)
    zip_bytes = build_repl.build_pyzip(REPO_ROOT)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(unpack)

    preamble = "import sys\nsys.platform = 'emscripten'\n"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(unpack),
        "HOME": str(tmp_path),
    }
    return subprocess.run(
        [sys.executable, "-c", preamble + script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_ziphygiene(tmp_path: Path) -> None:
    zip_a = build_repl.build_pyzip(REPO_ROOT)
    zip_b = build_repl.build_pyzip(REPO_ROOT)
    assert zip_a == zip_b
    assert len(zip_a) > 50_000

    stage = tmp_path / "assets"
    summary = build_repl.stage_assets(
        repo_root=REPO_ROOT,
        manifest_path=build_repl.DEFAULT_MANIFEST,
        cache_dir=build_repl.DEFAULT_CACHE_DIR,
        assets_dirs=[stage],
        zip_only=True,
    )
    assert (stage / "baux_py.zip").is_file()
    manifest = json.loads((stage / "repl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baux_py_zip"]["sha256"] == summary["baux_py_zip"]["sha256"]


def test_docsnippets(tmp_path: Path) -> None:
    snippets = _snippets()
    assert len(snippets) >= 6
    failures: list[str] = []
    for tree_name, idx, code in snippets:
        proc = _runzip(tmp_path / f"s_{tree_name}_{idx}", code)
        if proc.returncode != 0:
            failures.append(f"{tree_name}#{idx}:\n{proc.stderr}\n{proc.stdout}")
    assert not failures, "\n---\n".join(failures)


def test_wasmapi(tmp_path: Path) -> None:
    script = """
import numpy as np
import baux
from baux.benchmarks import oxcaml

caps = baux.capabilities()
assert len(caps) == 4 and caps[0].name == "cpu" and caps[0].available is True

# Core BayesOptimizer workflow: ask / tell / complete / cancel / freeze / thaw
spec = baux.pytree_bounds(
    {
        "x0": (),
        "x1": baux.Log(1e-3, 1.0),
        "mode": baux.Categorical(["a", "b", "c"]),
    },
    low={"x0": -5.0},
    high={"x0": 5.0},
)
cfg = baux.turbo_enn(num_init=4, trust_region=baux.RegionalSearchPolicy(count=2))
blank = baux.init(spec, cfg, seed=7)
state = blank

for _ in range(8):
    state, batch = baux.ask(state, 2)
    ys = np.array(
        [[-(float(p["x0"]) ** 2 + np.log10(float(p["x1"])) ** 2)] for p in batch.params]
    )
    state, _ = baux.tell(state, batch, ys)

assert state.num_observations == 16
assert state.best_y is not None and float(state.best_y) <= 0.0

state, pending = baux.ask(state, 2)
state = baux.cancel(state, [pending.ids[0]])
state = baux.complete(state, [pending.ids[1]], np.array([[-1.25]]))
assert state.num_observations == 17

blob = baux.freeze(state)
restored = baux.thaw(blank, blob)
assert restored.num_observations == state.num_observations
assert np.allclose(restored.best_y, state.best_y)

# Manifold codomain (Stiefel) + Periodic + Simplex parameter space
basis = baux.Stiefel.from_factors(np.array([[1.0, 0.2], [0.0, 1.0], [0.5, -0.3], [0.2, 0.4]]))
assert basis.loadings.shape == (4, 2)
assert np.allclose(basis.loadings.T @ basis.loadings, np.eye(2), atol=1e-6)

manifold_spec = baux.pytree_bounds({
    "weights": baux.Simplex(4),
    "phase": baux.Periodic(0.0, 2 * np.pi),
})
st = baux.init(manifold_spec, baux.turbo_one(num_init=3), seed=11)
st, st_batch = baux.ask(st, 2)
for item in st_batch.params:
    w = np.asarray(item["weights"])
    assert w.shape == (4,)
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
st_ys = np.array([[-float(np.sum(np.asarray(item["weights"]) ** 2))] for item in st_batch.params])
st, _ = baux.tell(st, st_batch, st_ys)
assert st.num_observations == 2

# Functional ENN pipeline + fit + posterior + sample
rng = np.random.default_rng(42)
x_train = rng.uniform(-2.0, 2.0, size=(24, 3))
y_train = np.sin(x_train[:, :1]) + 0.1 * x_train[:, 1:2] ** 2
enn = baux.enn_init(x_train, y_train, scale_x=True)
k_fit, k_draw = baux.split(baux.key(42), 2)
params = baux.enn_fit(enn, k=5, key=k_fit)
post = baux.enn_posterior(enn, np.zeros((3, 3)), params)
assert post.mu.shape == (3, 1) and post.se.shape == (3, 1)
draws, neigh = baux.enn_sample(enn, np.zeros((2, 3)), params, key=k_draw, num_draws=4)
assert draws.shape == (2, 1, 4) and len(neigh) == 2

# OxCaml 56-surface catalogue
cat = oxcaml.catalogue()
assert len(cat) == 56
desc = oxcaml.describe("ackley", 16)
assert desc["dim"] == 16
x_mid = np.zeros(16)
v1 = oxcaml.eval_one("ackley", x_mid)
vb = oxcaml.eval_batch("ackley", np.zeros((3, 16)))
assert vb.shape == (3,)
assert np.allclose(vb, v1)
print("OK")
"""
    proc = _runzip(tmp_path / "api", script)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert proc.stdout.strip().endswith("OK")


@pytest.mark.skipif(
    not NATIVE_INSPECT_DIR.is_dir(),
    reason="Extracted native wheel /tmp/baux_native_inspect not present",
)
def test_parity(tmp_path: Path) -> None:
    native_code = """
import json
from baux._rust import Key, normal_hash
import numpy as np
k0 = Key(123456789)
k1, k2 = k0.split()
k3 = k0.fold(42)
arr = normal_hash(np.array([10, 20], dtype=np.int64), np.array([0, 1, 2], dtype=np.int64), 3)
print(json.dumps({
    "vals": [k0.hi, k0.lo, k0.seed(), k1.hi, k1.lo, k2.hi, k2.lo, k3.hi, k3.lo],
    "normals": arr.tolist(),
}))
"""
    native_proc = subprocess.run(
        [sys.executable, "-c", native_code],
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(NATIVE_INSPECT_DIR)},
        text=True,
        capture_output=True,
        check=True,
    )
    wasm_proc = _runzip(
        tmp_path / "parity",
        native_code.replace("from baux._rust", "from baux.baux_rust"),
    )
    assert wasm_proc.returncode == 0, wasm_proc.stderr
    native_out = json.loads(native_proc.stdout)
    wasm_out = json.loads(wasm_proc.stdout)
    assert wasm_out["vals"] == native_out["vals"]
    assert len(wasm_out["normals"]) == len(native_out["normals"])
    for row_w, row_n in zip(wasm_out["normals"], native_out["normals"], strict=True):
        assert len(row_w) == len(row_n)


def _extract_presets_from_ml() -> dict[str, str]:
    import re

    root = Path(__file__).resolve().parents[1]
    text = (root / "admin/ui/py_repl.ml").read_text(encoding="utf-8")
    m = re.search(r"let presets\s*:\s*preset list\s*=\s*\[(.*?)\n  \]", text, flags=re.DOTALL)
    assert m, "presets block not found in admin/ui/py_repl.ml"
    body = m.group(1)
    presets: dict[str, str] = {}
    for block in re.finditer(
        r'\{\s*id\s*=\s*"([^"]+)";\s*label\s*=.*?code\s*=\s*(.*?)\s*;\s*\}',
        body,
        flags=re.DOTALL,
    ):
        pid, raw_expr = block.group(1), block.group(2)
        parts = re.findall(r'"((?:\\.|[^"\\])*)"', raw_expr)
        code = "".join(bytes(p, "utf-8").decode("unicode_escape") for p in parts)
        presets[pid] = code
    assert len(presets) == 7, f"expected 7 presets, found {list(presets)}"
    return presets


def test_presets_execute_and_emit_valid_figs(tmp_path: Path) -> None:
    """Every Playground preset must execute cleanly in WASM mode and emit valid fig specs."""
    import re
    import shutil

    root = Path(__file__).resolve().parents[1]
    presets = _extract_presets_from_ml()
    worker_js = (root / "forest/theme/repl_worker.js").read_text(encoding="utf-8")
    boot_m = re.search(r"const BOOTSTRAP_PY = `(.*?)`;\n", worker_js, flags=re.DOTALL)
    assert boot_m, "BOOTSTRAP_PY not found in repl_worker.js"
    bootstrap_py = boot_m.group(1)
    work = tmp_path / "presets"
    work.mkdir(parents=True, exist_ok=True)
    env_src = root / "forest/output/assets/baux_env.json"
    if env_src.is_file():
        shutil.copy(env_src, work / "baux_env.json")

    script = f"""
import json, os, sys
if hasattr(sys, "setcwd"):
    pass
os.chdir({str(work)!r})
globs = {{}}
exec({bootstrap_py!r}, globs, globs)
globs["_reset_namespace"]()
out = {{}}
for pid, src in {presets!r}.items():
    res = json.loads(globs["_baux_exec_script"](src, reset_ns=True))
    assert res["ok"], f"{{pid}} failed: {{res.get('error')}}"
    assert res["stdout"].strip(), f"{{pid}} produced empty stdout"
    state = res.get("state")
    assert state is not None and state.get("fig") is not None, f"{{pid}} missing fig: {{state}}"
    fig = state["fig"]
    assert fig["kind"] in {{"curves", "tokamak", "field2d", "band1d"}}, fig["kind"]
    out[pid] = fig["kind"]
print("OK:" + json.dumps(out))
"""
    proc = _runzip(work, script)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "OK:" in proc.stdout
