"""End-to-end verification of the browser WASM ``baux`` Python backend and asset builder."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TREES_DIR = REPO_ROOT / "forest" / "trees"
NATIVE_INSPECT_DIR = Path(tempfile.gettempdir()) / "baux_native_inspect"
WASM_SO_NAME = "baux_rust.cpython-312-wasm32-emscripten.so"
WASM_SO_ARCNAME = f"baux/{WASM_SO_NAME}"
FORBIDDEN_WASM_SHIMS = (
    "rust_wasm.py",
    "oxcaml_wasm.py",
    "_oxcaml_wasm.py",
    "baux_rust.py",
)
REQUIRED_PYODIDE_FILES = (
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
    "numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl",
)

_NODE_PYODIDE_HARNESS = r"""
const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline");

const oxcamlPath = process.env.BAUX_OXCAML_JS;
const loadedOxcaml = require(oxcamlPath);
if (!globalThis.BauxOxcaml && loadedOxcaml) {
  globalThis.BauxOxcaml = loadedOxcaml.BauxOxcaml || loadedOxcaml;
}
globalThis._baux_oxcaml = globalThis.BauxOxcaml;

const pyodideDir = process.env.BAUX_PYODIDE_DIR;
const { loadPyodide } = require(path.join(pyodideDir, "pyodide.js"));

function registerWebGpuTestStubs(pyodide) {
  const mod = pyodide._module;
  if (!mod) {
    return;
  }
  const makeStub = (sig, rc) => {
    const fn = () => rc;
    fn.sig = sig;
    return fn;
  };
  const symbols = {
    bauxgpu_available: makeStub("i", 0),
    bauxgpu_alloc: makeStub("ii", 0),
    bauxgpu_free: makeStub("ii", 2),
    bauxgpu_readf32: makeStub("iiii", 2),
    bauxgpu_exactdist: makeStub("iiiiiiii", 2),
    bauxgpu_knntopk: makeStub("iiiiiiii", 2),
    bauxgpu_posterior: makeStub("iiiiiiifffiiiiiiiii", 2),
    bauxgpu_denseapply: makeStub("iiiiiiii", 2),
    bauxgpu_densedist2: makeStub("iiiiiii", 2),
    bauxgpu_weightselect: makeStub("iiiiifffiiiiiii", 2),
  };
  if (typeof mod.mergeLibSymbols === "function") {
    mod.mergeLibSymbols(symbols, "bauxgpu");
  }
  for (const [k, v] of Object.entries(symbols)) {
    mod[k] = v;
  }
}

const DRIVER_PY = `
import io
import json
import os
import sys
import traceback

if "/home/pyodide/baux_pkg" not in sys.path:
    sys.path.insert(0, "/home/pyodide/baux_pkg")


def _baux_test_exec(src):
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    old_cwd = os.getcwd()
    sys.stdout = out_buf
    sys.stderr = err_buf
    ok = True
    try:
        os.chdir("/home/pyodide")
        ns = {"__name__": "__main__"}
        exec(compile(src, "<pyodide-test>", "exec"), ns, ns)
    except BaseException:
        ok = False
        traceback.print_exc(file=err_buf)
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
    return json.dumps({
        "ok": ok,
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue(),
    })
`;

async function main() {
  const indexURL = pyodideDir.endsWith(path.sep)
    ? pyodideDir
    : pyodideDir + path.sep;
  const pyodide = await loadPyodide({ indexURL });
  registerWebGpuTestStubs(pyodide);
  await pyodide.loadPackage(["numpy"]);

  const zipRaw = fs.readFileSync(process.env.BAUX_PY_ZIP);
  const zipBuf = new Uint8Array(zipRaw.buffer, zipRaw.byteOffset, zipRaw.byteLength);
  pyodide.FS.mkdirTree("/home/pyodide/baux_pkg");
  pyodide.unpackArchive(zipBuf, "zip", { extractDir: "/home/pyodide/baux_pkg" });

  const soPath = "/home/pyodide/baux_pkg/baux/baux_rust.cpython-312-wasm32-emscripten.so";
  if (pyodide.FS.analyzePath(soPath).exists) {
    if (pyodide._api && typeof pyodide._api.loadDynlib === "function") {
      await pyodide._api.loadDynlib(soPath, false);
    } else if (typeof pyodide.loadDynamicLibrary === "function") {
      await pyodide.loadDynamicLibrary(soPath, {
        loadAsync: true,
        nodelete: true,
        global: false,
        fs: pyodide.FS,
      });
    } else if (pyodide._module && typeof pyodide._module.loadDynamicLibrary === "function") {
      await pyodide._module.loadDynamicLibrary(soPath, {
        loadAsync: true,
        nodelete: true,
        allowUndefined: true,
        global: false,
        fs: pyodide.FS,
      });
    }
  }

  const envPath = process.env.BAUX_ENV_JSON;
  if (envPath && fs.existsSync(envPath)) {
    const envRaw = fs.readFileSync(envPath);
    pyodide.FS.writeFile(
      "/home/pyodide/baux_env.json",
      new Uint8Array(envRaw.buffer, envRaw.byteOffset, envRaw.byteLength)
    );
  }

  await pyodide.runPythonAsync(DRIVER_PY);
  const execFn = pyodide.globals.get("_baux_test_exec");

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  process.stdout.write("__BAUX_PYODIDE_READY__\n");

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    const req = JSON.parse(trimmed);
    const raw = execFn(req.code);
    process.stdout.write("__BAUX_PYODIDE_RESULT__:" + raw + "\n");
  }
}

main().catch((err) => {
  process.stderr.write(String(err && err.stack ? err.stack : err) + "\n");
  process.exit(1);
});
""".lstrip()


def _wasmso() -> Path | None:
    env_val = os.environ.get("BAUX_WASM_SO", "").strip()
    if env_val:
        cand = Path(env_val) if Path(env_val).is_absolute() else REPO_ROOT / env_val
        return cand if cand.is_file() else None

    for cand in (
        REPO_ROOT / ".cache" / "baux" / "wasm" / WASM_SO_NAME,
        REPO_ROOT
        / "rust"
        / "target"
        / "wasm32-unknown-emscripten"
        / "release"
        / WASM_SO_NAME,
        REPO_ROOT
        / "rust"
        / "target"
        / "wasm32-unknown-emscripten"
        / "release"
        / "libbaux_rust.so",
    ):
        if cand.is_file():
            return cand

    buck_out = REPO_ROOT / "buck-out"
    if buck_out.is_dir():
        exact = sorted(
            (p for p in buck_out.rglob(WASM_SO_NAME) if p.is_file()),
            key=lambda p: p.as_posix(),
        )
        if exact:
            return exact[0]
        tagged = sorted(
            (
                p
                for p in buck_out.rglob("*.so")
                if p.is_file() and "baux-py-wasm" in p.as_posix()
            ),
            key=lambda p: p.as_posix(),
        )
        if tagged:
            return tagged[0]
    return None


def _replassets(
    *,
    assets_dirs: list[Path],
    wasm_so: Path | None,
    zip_only: bool = True,
) -> dict[str, Any]:
    cmd = [str(REPO_ROOT / "baux"), "pages", "repl-assets", "--json"]
    for assets_dir in assets_dirs:
        cmd.extend(["--assets-dir", str(assets_dir)])
    if wasm_so is not None:
        cmd.extend(["--wasm-so", str(wasm_so)])
    if zip_only:
        cmd.append("--zip-only")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or "baux pages repl-assets failed").strip()
        )
    return json.loads(proc.stdout)


def _pyodidedir() -> Path | None:
    for cand in (
        REPO_ROOT / "forest" / "output" / "assets" / "pyodide",
        REPO_ROOT / ".cache" / "baux" / "pyodide" / "v0.27.5",
    ):
        if cand.is_dir() and all(
            (cand / name).is_file() for name in REQUIRED_PYODIDE_FILES
        ):
            return cand
    return None


def _oxcamljsp() -> Path | None:
    for cand in (
        REPO_ROOT / "forest" / "output" / "assets" / "oxcaml_eval.js",
        REPO_ROOT / "admin" / "ui" / "_build" / "default" / "oxcaml_eval.js",
    ):
        if cand.is_file():
            return cand
    return None


def _pyzipbytes(wasm_so: Path | None) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = Path(tmpdir) / "assets"
        _replassets(assets_dirs=[stage], wasm_so=wasm_so, zip_only=True)
        return (stage / "baux_py.zip").read_bytes()


def _stageassets(
    *,
    assets_dirs: list[Path],
    wasm_so: Path | None,
    zip_only: bool = True,
) -> dict[str, Any]:
    return _replassets(
        assets_dirs=assets_dirs,
        wasm_so=wasm_so,
        zip_only=zip_only,
    )


def _ziphaswasm(zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False
    with zipfile.ZipFile(zip_path) as zf:
        return WASM_SO_ARCNAME in zf.namelist()


@dataclass
class _PyodideRunResult:
    returncode: int
    stdout: str
    stderr: str


class _PyodideSession:
    """Persistent Node v26 + Pyodide v0.27.5 runner backed by compiled baux assets."""

    def __init__(
        self,
        *,
        pyodide_dir: Path,
        oxcaml_js: Path,
        py_zip_path: Path,
        driver_js: Path,
        env_json: Path | None,
    ) -> None:
        self.has_env_json = env_json is not None
        driver_js.write_text(_NODE_PYODIDE_HARNESS, encoding="utf-8")
        env = os.environ.copy()
        env["BAUX_PYODIDE_DIR"] = str(pyodide_dir)
        env["BAUX_OXCAML_JS"] = str(oxcaml_js)
        env["BAUX_PY_ZIP"] = str(py_zip_path)
        if env_json is not None and env_json.is_file():
            env["BAUX_ENV_JSON"] = str(env_json)

        self._proc = subprocess.Popen(
            ["node", str(driver_js)],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        assert self._proc.stdin is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                rc = self._proc.wait()
                raise RuntimeError(
                    f"Pyodide Node worker exited during startup (rc={rc}):\n{err}"
                )
            if line.strip() == "__BAUX_PYODIDE_READY__":
                break

    def run_python(self, code: str) -> _PyodideRunResult:
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps({"code": code}) + "\n")
        self._proc.stdin.flush()

        prefix = "__BAUX_PYODIDE_RESULT__:"
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                rc = self._proc.wait()
                return _PyodideRunResult(
                    returncode=rc or 1,
                    stdout="",
                    stderr=f"Pyodide Node worker terminated unexpectedly:\n{err}",
                )
            if line.startswith(prefix):
                payload = json.loads(line[len(prefix) :])
                return _PyodideRunResult(
                    returncode=0 if payload["ok"] else 1,
                    stdout=str(payload["stdout"]),
                    stderr=str(payload["stderr"]),
                )

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=5)


@pytest.fixture(scope="module")
def pyodide_session(tmp_path_factory: pytest.TempPathFactory):
    if shutil.which("node") is None:
        pytest.skip("node executable unavailable for Pyodide runtime verification")
    pyodide_dir = _pyodidedir()
    if pyodide_dir is None:
        pytest.skip(
            "Offline Pyodide v0.27.5 cache not staged yet "
            "(.cache/baux/pyodide/v0.27.5 or forest/output/assets/pyodide)"
        )
    oxcaml_js = _oxcamljsp()
    if oxcaml_js is None:
        pytest.skip("Compiled oxcaml_eval.js not staged yet; build admin/ui first")

    work = tmp_path_factory.mktemp("pyodide_runner")
    zip_path = work / "baux_py.zip"
    wasm_so = _wasmso()
    if wasm_so is not None:
        zip_path.write_bytes(_pyzipbytes(wasm_so))
    else:
        staged_zip = REPO_ROOT / "forest" / "output" / "assets" / "baux_py.zip"
        if _ziphaswasm(staged_zip):
            shutil.copy2(staged_zip, zip_path)
        else:
            pytest.skip(
                f"Compiled {WASM_SO_ARCNAME} not yet built/staged; "
                "build //:baux-py-wasm or stage baux_py.zip first"
            )

    env_json = next(
        (
            cand
            for cand in (
                REPO_ROOT / "forest" / "output" / "assets" / "baux_env.json",
                REPO_ROOT / ".cache" / "pages-bundle" / "assets" / "baux_env.json",
            )
            if cand.is_file()
        ),
        None,
    )

    session = _PyodideSession(
        pyodide_dir=pyodide_dir,
        oxcaml_js=oxcaml_js,
        py_zip_path=zip_path,
        driver_js=work / "driver.js",
        env_json=env_json,
    )
    try:
        yield session
    finally:
        session.close()


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


def test_ziphygiene(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        _pyzipbytes(tmp_path / "missing_baux_rust.so")

    wasm_so = _wasmso()
    if wasm_so is None:
        pytest.skip(f"Compiled {WASM_SO_NAME} not found in buck-out or BAUX_WASM_SO")

    zip_a = _pyzipbytes(wasm_so)
    zip_b = _pyzipbytes(wasm_so)
    assert zip_a == zip_b
    assert len(zip_a) > 50_000

    with zipfile.ZipFile(io.BytesIO(zip_a)) as zf:
        names = zf.namelist()
        assert WASM_SO_ARCNAME in names
        assert not any(name.endswith(FORBIDDEN_WASM_SHIMS) for name in names), (
            f"Found forbidden legacy WASM Python shim(s): "
            f"{[n for n in names if n.endswith(FORBIDDEN_WASM_SHIMS)]}"
        )
        for name in names:
            payload = zf.read(name)
            assert b"/Users/" not in payload, f"Machine path /Users/ leaked into {name}"
            assert b"/home/runner/" not in payload, (
                f"Runner path /home/runner/ leaked into {name}"
            )

    stage = tmp_path / "assets"
    summary = _stageassets(assets_dirs=[stage], wasm_so=wasm_so, zip_only=True)
    assert (stage / "baux_py.zip").is_file()
    manifest = json.loads((stage / "repl_manifest.json").read_text(encoding="utf-8"))
    assert manifest["baux_py_zip"]["sha256"] == summary["baux_py_zip"]["sha256"]


def test_docsnippets(pyodide_session: _PyodideSession) -> None:
    snippets = _snippets()
    assert len(snippets) >= 6
    failures: list[str] = []
    for tree_name, idx, code in snippets:
        proc = pyodide_session.run_python(code)
        if proc.returncode != 0:
            failures.append(f"{tree_name}#{idx}:\n{proc.stderr}\n{proc.stdout}")
    assert not failures, "\n---\n".join(failures)


def test_wasmapi(pyodide_session: _PyodideSession) -> None:
    script = """
import sys
import numpy as np
import baux
from baux import baux_rust
from baux.benchmarks import synthetic

assert sys.platform == "emscripten"
assert not str(getattr(baux_rust, "__file__", "")).endswith(".py")

caps = baux.capabilities()
assert [row.name for row in caps] == ["cpu", "cuda", "metal", "opencl", "webgpu"]
assert caps[0].name == "cpu" and caps[0].available is True
webgpu_row = next(row for row in caps if row.name == "webgpu")
assert webgpu_row.compiled is True
assert "device_posterior" in webgpu_row.direct_operations

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
cfg = baux.lhd(4) | baux.enn() | baux.regions(count=2)
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
st = baux.init(manifold_spec, pipe=baux.enn() | baux.ucb(), seed=11)
st, st_batch = baux.ask(st, 2)
for item in st_batch.params:
    w = np.asarray(item["weights"])
    assert w.shape == (4,)
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
st_ys = np.array([[-float(np.sum(np.asarray(item["weights"]) ** 2))] for item in st_batch.params])
st, _ = baux.tell(st, st_batch, st_ys)
assert st.num_observations == 2

try:
    baux.init(manifold_spec, pipe=baux.gp(), seed=11)
except RuntimeError as gp_exc:
    assert "Local Native" in str(gp_exc)
else:
    raise AssertionError("Expected GaussianProcess/gp to be disabled in WASM")

# Functional ENN pipeline + fit + posterior + sample
rng = np.random.default_rng(42)
x_train = rng.uniform(-2.0, 2.0, size=(24, 3))
y_train = np.sin(x_train[:, :1]) + 0.1 * x_train[:, 1:2] ** 2
enn = baux.enn_init(x_train, y_train, scale_x=True)
k_fit, k_draw = baux.split(baux.key(42), 2)
params = baux.fit(enn, k=5, key=k_fit)
post = baux.posterior(enn, np.zeros((3, 3)), params)
assert post.mu.shape == (3, 1) and post.se.shape == (3, 1)
draws, neigh = baux.sample(enn, np.zeros((2, 3)), params, key=k_draw, num_draws=4)
assert draws.shape == (2, 1, 4) and len(neigh) == 2

# OxCaml 56-surface catalogue via compiled oxcaml_eval.js bridge
cat = synthetic.catalogue()
assert len(cat) == 56
desc = synthetic.describe("ackley", 16)
assert desc["dim"] == 16
x_mid = np.zeros(16)
v1 = synthetic.eval_one("ackley", x_mid)
vb = synthetic.eval_batch("ackley", np.zeros((3, 16)))
assert vb.shape == (3,)
assert np.allclose(vb, v1)
print("OK")
"""
    proc = pyodide_session.run_python(script)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert proc.stdout.strip().endswith("OK")


@pytest.mark.skipif(
    not NATIVE_INSPECT_DIR.is_dir(),
    reason="Extracted native wheel /tmp/baux_native_inspect not present",
)
def test_parity(pyodide_session: _PyodideSession) -> None:
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
    import sys

    native_proc = subprocess.run(
        [sys.executable, "-c", native_code],
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(NATIVE_INSPECT_DIR)},
        text=True,
        capture_output=True,
        check=True,
    )
    wasm_proc = pyodide_session.run_python(native_code)
    assert wasm_proc.returncode == 0, wasm_proc.stderr
    native_out = json.loads(native_proc.stdout)
    wasm_out = json.loads(wasm_proc.stdout)
    assert wasm_out["vals"] == native_out["vals"]
    assert len(wasm_out["normals"]) == len(native_out["normals"])
    for row_w, row_n in zip(wasm_out["normals"], native_out["normals"], strict=True):
        assert len(row_w) == len(row_n)


def _mlpresets() -> dict[str, str]:
    text = (REPO_ROOT / "admin/ui/py_repl.ml").read_text(encoding="utf-8")
    m = re.search(
        r"let presets\s*:\s*preset list\s*=\s*\[(.*?)\n  \]", text, flags=re.DOTALL
    )
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


def test_presets(pyodide_session: _PyodideSession) -> None:
    """Every Playground preset must execute cleanly in real Pyodide + WASM and emit valid fig specs."""
    if not pyodide_session.has_env_json:
        pytest.skip("baux_env.json not staged when session initialized")

    presets = _mlpresets()
    worker_js = (REPO_ROOT / "forest/theme/repl_worker.js").read_text(encoding="utf-8")
    boot_m = re.search(r"const BOOTSTRAP_PY\s*=\s*`(.*?)`;", worker_js, flags=re.DOTALL)
    assert boot_m, "BOOTSTRAP_PY not found in repl_worker.js"
    bootstrap_py = boot_m.group(1)

    script = f"""
import json, os
os.chdir("/home/pyodide")
globs = {{}}
exec({bootstrap_py!r}, globs, globs)
globs["_reset_namespace"]()
out = {{}}
for pid, src in {presets!r}.items():
    res = json.loads(globs["_baux_exec_script"](src, reset_ns=True))
    assert res["ok"], f"{{pid}} failed: {{res.get('error')}} stderr={{res.get('stderr')}}"
    assert res["stdout"].strip(), f"{{pid}} produced empty stdout"
    state = res.get("state")
    assert state is not None and state.get("fig") is not None, f"{{pid}} missing fig: {{state}}"
    fig = state["fig"]
    assert fig["kind"] in {{"curves", "tokamak", "field2d", "band1d"}}, fig["kind"]
    out[pid] = fig["kind"]
print("OK:" + json.dumps(out))
"""
    proc = pyodide_session.run_python(script)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert "OK:" in proc.stdout
