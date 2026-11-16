from __future__ import annotations

import json
from typing import Any

import baux
import numpy as np
import pytest
from baux._rust import ENN as _RustENN
from baux.enn.enn_fit import subsample_loglik
from baux.enn.enn_params import ENNParams
from baux.enn.enn_state import ENNState

if not hasattr(_RustENN, "x_obs"):
    _RustENN.x_obs = lambda self: self.train_rows(list(range(len(self))))[0]
if not hasattr(_RustENN, "y_obs"):
    _RustENN.y_obs = lambda self: self.train_rows(list(range(len(self))))[1]
if not hasattr(_RustENN, "yvar_obs"):
    _RustENN.yvar_obs = lambda self: self.train_rows(list(range(len(self))))[2]


def _fitmodel(
    state: ENNState,
    *,
    k: int = 10,
    num_candidates: int = 100,
    num_samples: int = 10,
    rng: Any = None,
    seed: int | None = None,
    key: Any = None,
    warm: ENNParams | None = None,
    infer_aleatoric: bool = True,
) -> ENNParams:
    fit_seed = seed if seed is not None else rng
    return baux.fit(
        state,
        k=k,
        num_candidates=num_candidates,
        num_samples=num_samples,
        infer_aleatoric=infer_aleatoric,
        key=key,
        seed=fit_seed,
        warm=warm,
    )


def _ybatch(inc_std, batch_std) -> None:
    for a, b in zip(inc_std, batch_std, strict=True):
        expected = b if b > 1e-10 else 1.0
        assert abs(a - expected) < 1e-10


def _makedata(
    *,
    rng,
    n: int,
    d: int,
    noise_std: float,
    yvar: float | None,
):
    x = rng.standard_normal((n, d))
    y = x.sum(axis=1, keepdims=True) + rng.standard_normal((n, 1)) * float(noise_std)
    if yvar is None:
        return x, y, None
    return x, y, float(yvar) * np.ones_like(y)


def test_001():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 2))
    y = (x @ np.array([1.5, -0.5]) + 0.1 * rng.standard_normal(40)).reshape(-1, 1)
    state = baux.enn_init(x, y, 0.01 * np.ones_like(y))
    result = _fitmodel(
        state,
        k=10,
        num_candidates=30,
        num_samples=20,
        seed=1,
    )
    assert (
        isinstance(result, ENNParams)
        and result.k_neighbors == 10
        and result.epistemic_scale > 0.0
    )
    tuned_ll = subsample_loglik(
        state, x, y[:, 0], paramss=[result], P=20, rng=np.random.default_rng(2)
    )[0]
    assert np.isfinite(tuned_ll), "tuned log-likelihood must be finite"


def _captureresult(
    state: ENNState, params: ENNParams, num_samples: int, eval_x, eval_y
):
    loglik = subsample_loglik(
        state,
        eval_x,
        eval_y,
        paramss=[params],
        P=len(eval_x),
        rng=np.random.default_rng(2000 + num_samples),
    )[0]
    return {
        "num_samples": num_samples,
        "params": {
            "k_neighbors": params.k_neighbors,
            "epistemic_scale": params.epistemic_scale,
            "aleatoric_scale": params.aleatoric_scale,
        },
        "subsample_loglik": loglik,
    }


def _runsweep(x_train, y_train, y_var_train, sample_sizes):
    captured = []
    for num_samples in sample_sizes:
        state = baux.enn_init(x_train, y_train, y_var_train)
        params = _fitmodel(
            state,
            k=10,
            num_candidates=100,
            num_samples=num_samples,
            seed=1000 + num_samples,
        )
        captured.append(_captureresult(state, params, num_samples, x_train, y_train))
    return captured


def _runsweep2(x_train, y_train, y_var_train, sample_sizes):
    state = baux.enn_init(
        np.empty((0, x_train.shape[1])),
        np.empty((0, y_train.shape[1])),
        np.empty((0, y_var_train.shape[1])),
    )
    params_warm: ENNParams | None = None
    captured = []
    for num_samples, (x, y, y_var) in enumerate(
        zip(x_train, y_train, y_var_train, strict=True), start=1
    ):
        row_x = x.reshape(1, -1)
        row_y = y.reshape(1, -1)
        row_yvar = y_var.reshape(1, -1)
        state = baux.append(state, row_x, row_y, row_yvar)
        params_warm = _fitmodel(
            state,
            k=10,
            num_candidates=1,
            num_samples=100,
            seed=4242,
            warm=params_warm,
        )
        if num_samples in sample_sizes:
            captured.append(
                _captureresult(
                    state,
                    params_warm,
                    num_samples,
                    x_train[:num_samples],
                    y_train[:num_samples],
                )
            )
    return captured


@pytest.mark.slow
def test_002():
    rng = np.random.default_rng(20260522)
    x_train = rng.standard_normal((1000, 3))
    y_train = (
        x_train @ np.array([1.5, -0.5, 0.25]) + 0.1 * rng.standard_normal(1000)
    ).reshape(-1, 1)
    y_var_train = 0.01 * np.ones_like(y_train)
    sample_sizes = [10, 30, 100, 300, 1000]

    batch_captured = _runsweep(x_train, y_train, y_var_train, sample_sizes)
    incremental_captured = _runsweep2(x_train, y_train, y_var_train, sample_sizes)

    output = json.dumps(
        {"batch": batch_captured, "incremental": incremental_captured}, indent=2
    )
    assert len(batch_captured) == len(sample_sizes)
    assert len(incremental_captured) == len(sample_sizes)
    assert "subsample_loglik" in output


def test_keydeterminism():
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((30, 3))
    y = (x @ [1.0, -1.0, 0.5]).reshape(-1, 1)
    state = baux.enn_init(x, y)
    k1 = baux.key(999)
    p1 = baux.fit(state, k=4, num_candidates=15, num_samples=10, key=k1)
    p2 = baux.fit(state, k=4, num_candidates=15, num_samples=10, key=k1)
    assert p1 == p2
    assert p1.k_neighbors == 4
    assert np.isfinite(p1.epistemic_scale)


def test_forkinvariance():
    rng = np.random.default_rng(5678)
    x0 = rng.standard_normal((15, 2))
    y0 = np.sum(x0**2, axis=1, keepdims=True)
    st0 = baux.enn_init(x0, y0)
    p0_before = baux.fit(st0, k=3, num_candidates=10, num_samples=5, seed=101)

    x1 = rng.standard_normal((5, 2))
    y1 = np.sum(x1**2, axis=1, keepdims=True)
    st1 = baux.append(st0, x1, y1)

    assert len(st0) == 15
    assert len(st1) == 20
    np.testing.assert_allclose(st0.x, x0)
    np.testing.assert_allclose(st0.y, y0)

    p0_after = baux.fit(st0, k=3, num_candidates=10, num_samples=5, seed=101)
    assert p0_before == p0_after


def test_warmstart():
    rng = np.random.default_rng(999)
    x = rng.standard_normal((25, 2))
    y = (x[:, :1] * 2.0).astype(float)
    state = baux.enn_init(x, y)

    warm = ENNParams(k_neighbors=4, epistemic_scale=1.5, aleatoric_scale=0.05)
    p_cold = baux.fit(state, k=4, num_candidates=10, num_samples=5, seed=12)
    p_warm = baux.fit(state, k=4, num_candidates=10, num_samples=5, seed=12, warm=warm)
    assert isinstance(p_cold, ENNParams)
    assert isinstance(p_warm, ENNParams)
    assert p_cold.k_neighbors == 4
    assert p_warm.k_neighbors == 4


def test_aleatorictoggle():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((20, 2))
    y = (x @ [1.0, 2.0]).reshape(-1, 1)
    state = baux.enn_init(x, y, 0.05 * np.ones_like(y))

    p_off = baux.fit(
        state, k=3, num_candidates=10, num_samples=5, infer_aleatoric=False, seed=7
    )
    assert p_off.aleatoric_scale == 0.0

    p_on = baux.fit(
        state, k=3, num_candidates=10, num_samples=5, infer_aleatoric=True, seed=7
    )
    assert p_on.aleatoric_scale >= 0.0
