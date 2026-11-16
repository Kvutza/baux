from __future__ import annotations

import baux
import numpy as np
import pytest
from baux._rust import ENN as _RustENN
from baux.enn.enn_fit import subsample_loglik
from baux.enn.enn_params import ENNParams
from testefit import _fitmodel, _makedata, _ybatch

if not hasattr(_RustENN, "x_obs"):
    _RustENN.x_obs = lambda self: self.train_rows(list(range(len(self))))[0]
if not hasattr(_RustENN, "y_obs"):
    _RustENN.y_obs = lambda self: self.train_rows(list(range(len(self))))[1]
if not hasattr(_RustENN, "yvar_obs"):
    _RustENN.yvar_obs = lambda self: self.train_rows(list(range(len(self))))[2]


def test_ennfitwithyvarnone():
    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _makedata(rng=rng, n=n, d=d, noise_std=0.1, yvar=None)
    state = baux.enn_init(x, y, train_yvar=yvar)
    result = _fitmodel(
        state,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
    )
    assert isinstance(result, ENNParams)
    assert result.k_neighbors == 5
    assert result.epistemic_scale > 0.0
    assert result.aleatoric_scale >= 0.0


def test_ennfitwithwarmstart():
    rng = np.random.default_rng(42)
    n = 30
    d = 2
    x, y, yvar = _makedata(rng=rng, n=n, d=d, noise_std=0.1, yvar=0.01)
    state = baux.enn_init(x, y, yvar)
    result1 = _fitmodel(
        state,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
    )
    result2 = _fitmodel(
        state,
        k=5,
        num_candidates=20,
        num_samples=10,
        rng=rng,
        warm=result1,
    )
    assert isinstance(result2, ENNParams)
    assert result2.k_neighbors == 5
    assert result2.epistemic_scale > 0.0
    assert result2.aleatoric_scale >= 0.0


def test_003():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((60, 3))
    y1 = x @ [1.0, -2.0, 0.5] + 0.1 * rng.standard_normal(60)
    y2 = np.sin(x @ [-0.5, 0.25, 1.25]) + 0.3 * rng.standard_normal(60)
    y = np.column_stack([y1, y2]).astype(float)
    state = baux.enn_init(x, y, np.ones_like(y) * [[0.01, 0.09]])
    params = _fitmodel(
        state,
        k=12,
        num_candidates=40,
        num_samples=25,
        rng=np.random.default_rng(456),
    )
    assert isinstance(params, ENNParams) and params.k_neighbors == 12
    lls = subsample_loglik(
        state, x, y, paramss=[params], P=25, rng=np.random.default_rng(789)
    )
    assert len(lls) == 1 and np.isfinite(lls[0])


def test_004():
    rng = np.random.default_rng(42)
    x, y, yvar = _makedata(rng=rng, n=40, d=3, noise_std=0.2, yvar=0.01)
    state = baux.enn_init(x, y, yvar)
    warm = ENNParams(
        k_neighbors=5,
        epistemic_scale=1.0,
        aleatoric_scale=123.0,
    )
    result = _fitmodel(
        state,
        k=5,
        num_candidates=25,
        num_samples=15,
        rng=rng,
        warm=warm,
        infer_aleatoric=False,
    )
    assert isinstance(result, ENNParams)
    assert result.k_neighbors == 5
    assert result.aleatoric_scale == 0.0


def test_005():
    with pytest.raises(ValueError, match="k_neighbors must be > 0"):
        ENNParams(k_neighbors=0, epistemic_scale=1.0, aleatoric_scale=0.1)
    with pytest.raises(ValueError, match="epistemic_scale must be >= 0"):
        ENNParams(k_neighbors=2, epistemic_scale=-1.0, aleatoric_scale=0.1)
    with pytest.raises(ValueError, match="aleatoric_scale must be >= 0"):
        ENNParams(k_neighbors=2, epistemic_scale=1.0, aleatoric_scale=-0.1)
    state = baux.enn_init([[0.0, 0.0], [1.0, 1.0]], [[0.0], [1.0]])
    with pytest.raises(ValueError):
        baux.fit(state, k=0)


def test_006():
    rng = np.random.default_rng(2020)
    x = rng.standard_normal((12, 2))
    y = (10.0 * rng.standard_normal((12, 1))).astype(float)

    st_batch = baux.enn_init(x, y)
    p_batch = _fitmodel(
        st_batch,
        k=3,
        num_candidates=8,
        num_samples=6,
        rng=np.random.default_rng(100),
    )

    st_inc = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)))
    for row_x, row_y in zip(x, y, strict=True):
        st_inc = baux.append(st_inc, row_x.reshape(1, -1), row_y.reshape(1, -1))
    p_inc = _fitmodel(
        st_inc,
        k=3,
        num_candidates=8,
        num_samples=6,
        rng=np.random.default_rng(100),
    )

    assert p_batch.k_neighbors == p_inc.k_neighbors
    assert abs(p_batch.epistemic_scale - p_inc.epistemic_scale) < 1e-9
    assert abs(p_batch.aleatoric_scale - p_inc.aleatoric_scale) < 1e-9


def test_007():
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0], [1.5]], dtype=float)
    state = baux.enn_init(x, y)
    warm = ENNParams(
        k_neighbors=2,
        epistemic_scale=1.5466372900231078,
        aleatoric_scale=0.2598217339416701,
    )

    p_cold = _fitmodel(
        state,
        k=2,
        num_candidates=1,
        num_samples=4,
        rng=np.random.default_rng(55),
    )
    p_warm = _fitmodel(
        state,
        k=2,
        num_candidates=1,
        num_samples=4,
        warm=warm,
        rng=np.random.default_rng(55),
    )
    assert np.isfinite(p_cold.epistemic_scale)
    assert np.isfinite(p_warm.epistemic_scale)
    assert p_warm == warm
    assert (
        abs(p_cold.epistemic_scale - p_warm.epistemic_scale) > 1e-12
        or abs(p_cold.aleatoric_scale - p_warm.aleatoric_scale) > 1e-12
    )


def test_008():
    rng = np.random.default_rng(88)
    x_all = rng.standard_normal((8, 2))
    y_all = rng.standard_normal((8, 1))
    state = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)))
    for row_x, row_y in zip(x_all, y_all, strict=True):
        state = baux.append(state, row_x.reshape(1, -1), row_y.reshape(1, -1))
    params = _fitmodel(
        state,
        k=2,
        num_candidates=5,
        num_samples=4,
        rng=np.random.default_rng(88),
    )
    assert np.isfinite(params.epistemic_scale)
    assert params.epistemic_scale > 0.0


def test_009():
    with pytest.raises(ValueError):
        baux.enn_init([[0.0]], [[float("nan")]])


def test_010():
    st = baux.enn_init([[0.0, 0.0]], [[0.0]])
    with pytest.raises(ValueError):
        baux.append(st, [[1.0, 2.0, 3.0]], [[1.0]])
    with pytest.raises(ValueError):
        baux.append(st, [[1.0, 2.0]], [[1.0]], y_var=[[0.1]])


def test_011():
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0], [1.5]], dtype=float)
    state = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)))
    for i in range(y.shape[0]):
        state = baux.append(state, x[i : i + 1], y[i : i + 1])
        _ybatch(state.y.std(axis=0), y[: i + 1].std(axis=0))


def test_012():
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=float,
    )
    y = np.array(
        [[0.0, 1.0], [1.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 1.5]],
        dtype=float,
    )
    state = baux.enn_init(np.empty((0, 2)), np.empty((0, 2)))
    for i in range(y.shape[0]):
        state = baux.append(state, x[i : i + 1], y[i : i + 1])
        _ybatch(state.y.std(axis=0), y[: i + 1].std(axis=0))


def test_013():
    with pytest.raises(ValueError):
        baux.enn_init([[0.0, 0.0]], [[0.0]], [[float("nan")]])
    st = baux.enn_init([[0.0, 0.0]], [[0.0]], [[0.1]])
    assert st.y_var is not None


def test_014():
    x = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    state = baux.enn_init(x, y)
    warm = ENNParams(k_neighbors=2, epistemic_scale=2.5, aleatoric_scale=0.3)

    params = _fitmodel(
        state,
        k=2,
        num_candidates=0,
        num_samples=2,
        warm=warm,
        rng=np.random.default_rng(7),
    )
    assert params.k_neighbors == 2
    assert abs(params.epistemic_scale - 2.5) < 1e-12


def test_015():
    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    y = np.array([[0.0], [0.0], [0.0], [100.0], [100.0], [100.0]], dtype=float)
    st_sync = baux.enn_init(x, y)

    st_partial = baux.enn_init(x[:3], y[:3])
    st_fork = baux.append(st_partial, x[3:], y[3:])

    assert len(st_partial) == 3
    assert len(st_fork) == 6
    np.testing.assert_allclose(st_fork.y, st_sync.y)

    p_sync = _fitmodel(
        st_sync,
        k=2,
        num_candidates=6,
        num_samples=5,
        rng=np.random.default_rng(99),
    )
    p_fork = _fitmodel(
        st_fork,
        k=2,
        num_candidates=6,
        num_samples=5,
        rng=np.random.default_rng(99),
    )
    assert p_sync == p_fork


def test_016():
    state = baux.enn_init(
        np.array([[0.0, 0.0]]),
        np.array([[0.0]]),
    )
    params = _fitmodel(
        state,
        k=7,
        num_candidates=5,
        num_samples=3,
        rng=np.random.default_rng(1),
    )
    assert params.k_neighbors == 7
    assert params.epistemic_scale == 1.0
    assert params.aleatoric_scale == 0.0
