"""Tests for the modal OxCaml 56-function Python benchmark bridge."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import baux
import numpy as np
from baux.benchmarks import oxcaml

ROOT = Path(__file__).resolve().parent.parent


def test_catalogue() -> None:
    entries = oxcaml.catalogue()
    assert len(entries) == 56
    ids = {row["id"] for row in entries}
    assert len(ids) == 56
    scalable = [row["id"] for row in entries if row["scalable"]]
    assert len(scalable) == 18
    for row in entries:
        assert len(row["bounds"]) == row["dimension"]
        for lo, hi in row["bounds"]:
            assert lo < hi


def test_reference() -> None:
    ref_path = ROOT / "ocaml" / "tests" / "reference.tsv"
    grouped: dict[tuple[str, int], list[tuple[list[float], float]]] = defaultdict(list)
    total = 0
    for raw in ref_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fid, dim_s, coords_s, expected_s = line.split("\t")
        coords = [float(tok) for tok in coords_s.split(",")]
        assert len(coords) == int(dim_s)
        expected = float(expected_s)
        actual = oxcaml.eval_one(fid, coords)
        assert np.isclose(actual, expected, rtol=1e-9, atol=1e-9), (
            f"{fid}({coords}): got {actual}, expected {expected}"
        )
        grouped[(fid, len(coords))].append((coords, expected))
        total += 1

    assert total == 460
    for (fid, _dim), items in grouped.items():
        pts = np.asarray([pt for pt, _ in items], dtype=np.float64)
        exp = np.asarray([val for _, val in items], dtype=np.float64)
        got_simd = oxcaml.eval_batch(fid, pts, backend="simd", workers=1)
        got_scalar = oxcaml.eval_batch(fid, pts, backend="scalar", workers=1)
        np.testing.assert_allclose(got_simd, exp, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(got_scalar, exp, rtol=1e-9, atol=1e-9)


def test_scalable() -> None:
    rng = np.random.default_rng(7)
    scalable = [row["id"] for row in oxcaml.catalogue() if row["scalable"]]
    assert len(scalable) == 18
    for dim in (16, 64):
        for fid in scalable:
            meta = oxcaml.describe(fid, dim)
            bounds = np.asarray(meta["bounds"], dtype=np.float64)
            assert bounds.shape == (dim, 2)
            pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(64, dim))
            y_scalar = oxcaml.eval_batch(fid, pts, backend="scalar", workers=1)
            y_simd = oxcaml.eval_batch(fid, pts, backend="simd", workers=1)
            y_par = oxcaml.eval_batch(fid, pts, backend="simd", workers=4)
            np.testing.assert_allclose(y_simd, y_scalar, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(y_par, y_scalar, rtol=1e-10, atol=1e-10)


def test_refcounts() -> None:
    token = ["modal", "probe", 42]
    # sys.getrefcount adds 1 temporary reference for its own call argument;
    # inside py_refcount_probe (METH_O), borrowed_rc equals base - 1? Let's verify exact delta:
    probe = oxcaml.refcount_probe(token)
    assert probe["owned_rc"] == probe["borrowed_rc"] + 1
    assert probe["after_release_rc"] == probe["borrowed_rc"]

    pts = np.zeros((32, 8), dtype=np.float64)
    rc_before = sys.getrefcount(pts)
    raw = oxcaml.raw_batch("ackley", pts, backend="simd", workers=2)
    rc_after = sys.getrefcount(pts)
    assert rc_after == rc_before

    # Stolen @ unique bytearray has ob_refcnt == 1:
    # - Python 3.14+ passes local variables to C builtins via LOAD_FAST_BORROW -> getrefcount == 1
    # - Earlier CPython increments arg refcount on call -> getrefcount == 2
    assert isinstance(raw, bytearray)
    assert oxcaml.refcount_probe(raw)["borrowed_rc"] == sys.getrefcount(raw)
    assert sys.getrefcount(raw) in (1, 2)


def test_loop() -> None:
    bounds, objective = oxcaml.problem("ackley", dim=16, workers=2)
    assert bounds.shape == (16, 2)
    pipe = baux.enn(k=10) | baux.ucb(beta=2.0)
    state = baux.init(bounds, pipe, key=baux.key(0))

    state, first = baux.ask(state, num_arms=8)
    y_init = objective(first.points)
    init_best = float(np.max(y_init))
    state, _ = baux.tell(state, first, y_init)

    for _ in range(18):
        state, batch = baux.ask(state, num_arms=8)
        state, _ = baux.tell(state, batch, objective(batch.points))

    assert state.best_x.shape == (16,)
    assert state.best_y > init_best
