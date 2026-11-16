from __future__ import annotations

import baux
import numpy as np
import pytest
from baux._rust import ENN as _RustENN
from baux.enn.enn_params import ENNParams
from baux.enn.enn_state import ENNState

if not hasattr(_RustENN, "x_obs"):

    def _xobs(self):
        n = len(self)
        if n == 0:
            return np.empty((0, self.num_dim), dtype=float)
        return self.train_rows(list(range(n)))[0]

    _RustENN.x_obs = _xobs

if not hasattr(_RustENN, "y_obs"):

    def _yobs(self):
        n = len(self)
        if n == 0:
            return np.empty((0, self.num_outputs), dtype=float)
        return self.train_rows(list(range(n)))[1]

    _RustENN.y_obs = _yobs

if not hasattr(_RustENN, "yvar_obs"):

    def _yvarobs(self):
        try:
            n = len(self)
            rows = list(range(n))
            _, _, yvar = self.train_rows(rows)
            return yvar
        except Exception:
            return None

    _RustENN.yvar_obs = _yvarobs


def test_001():
    rng = np.random.default_rng(7)
    x_all = rng.standard_normal((12, 2))
    y_all = rng.standard_normal((12, 1))
    yvar_all = 0.01 * np.ones_like(y_all)

    init_state = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)), np.empty((0, 1)))
    assert isinstance(init_state, ENNState)
    assert len(init_state) == 0

    state = init_state
    params: ENNParams | None = None
    base_key = baux.key(7)

    states_history: list[ENNState] = [init_state]

    for i, (row_x, row_y, row_yvar) in enumerate(
        zip(x_all, y_all, yvar_all, strict=True)
    ):
        row_x_2d = row_x.reshape(1, -1)
        row_y_2d = row_y.reshape(1, -1)
        row_yvar_2d = row_yvar.reshape(1, -1)

        prev_state = state
        state = baux.append(state, row_x_2d, row_y_2d, row_yvar_2d)

        # Immutability assertions on previous state
        assert len(prev_state) == i
        assert len(state) == i + 1
        assert prev_state is not state
        assert prev_state.x.shape[0] == i
        assert state.x.shape[0] == i + 1
        assert prev_state.y.shape[0] == i
        assert state.y.shape[0] == i + 1
        if i > 0:
            np.testing.assert_allclose(prev_state.x, x_all[:i])
            np.testing.assert_allclose(prev_state.y, y_all[:i])

        states_history.append(state)

        step_key = baux.fold(base_key, i)
        params = baux.fit(
            state,
            k=3,
            num_candidates=1,
            num_samples=8,
            warm=params,
            key=step_key,
        )

        assert isinstance(params, ENNParams)
        assert params.k_neighbors == 3
        assert np.isfinite(params.epistemic_scale)
        assert np.isfinite(params.aleatoric_scale)

    assert isinstance(params, ENNParams)
    assert params.k_neighbors == 3
    assert np.isfinite(params.epistemic_scale)
    assert params.epistemic_scale > 0.0

    # Verify all historical states preserved immutability throughout the loop
    for expected_len, hist_state in enumerate(states_history):
        assert len(hist_state) == expected_len
        assert hist_state.x.shape[0] == expected_len
        assert hist_state.y.shape[0] == expected_len
        if expected_len > 0:
            np.testing.assert_allclose(hist_state.x, x_all[:expected_len])
            np.testing.assert_allclose(hist_state.y, y_all[:expected_len])


def test_002():
    rng = np.random.default_rng(99)
    x_all = rng.standard_normal((8, 2))
    y_all = rng.standard_normal((8, 1))

    # Run A: incremental append + fit with warm start and deterministic keys
    base_key = baux.key(99)
    state_a = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)))
    params_a: ENNParams | None = None
    for i, (row_x, row_y) in enumerate(zip(x_all, y_all, strict=True)):
        state_a = baux.append(state_a, row_x.reshape(1, -1), row_y.reshape(1, -1))
        params_a = baux.fit(
            state_a,
            k=2,
            num_candidates=1,
            num_samples=6,
            warm=params_a,
            key=baux.fold(base_key, i),
        )

    # Run B: independent identical replay
    state_b = baux.enn_init(np.empty((0, 2)), np.empty((0, 1)))
    params_b: ENNParams | None = None
    for i, (row_x, row_y) in enumerate(zip(x_all, y_all, strict=True)):
        state_b = baux.append(state_b, row_x.reshape(1, -1), row_y.reshape(1, -1))
        params_b = baux.fit(
            state_b,
            k=2,
            num_candidates=1,
            num_samples=6,
            warm=params_b,
            key=baux.fold(base_key, i),
        )

    assert params_a is not None and params_b is not None
    assert params_a.k_neighbors == params_b.k_neighbors
    assert abs(params_a.epistemic_scale - params_b.epistemic_scale) < 1e-12
    assert abs(params_a.aleatoric_scale - params_b.aleatoric_scale) < 1e-12
    assert state_a.key == state_b.key

    # Consistency with batch initialized state
    state_batch = baux.enn_init(x_all, y_all)
    assert len(state_a) == len(state_batch) == 8
    np.testing.assert_allclose(state_a.x, state_batch.x)
    np.testing.assert_allclose(state_a.y, state_batch.y)

    eval_key = baux.key(42)
    p_from_inc = baux.fit(
        state_a,
        k=2,
        num_candidates=10,
        num_samples=5,
        key=eval_key,
    )
    p_from_batch = baux.fit(
        state_batch,
        k=2,
        num_candidates=10,
        num_samples=5,
        key=eval_key,
    )
    assert p_from_inc.k_neighbors == p_from_batch.k_neighbors
    assert abs(p_from_inc.epistemic_scale - p_from_batch.epistemic_scale) < 1e-12
    assert abs(p_from_inc.aleatoric_scale - p_from_batch.aleatoric_scale) < 1e-12


def test_003():
    """Verify branching immutability, caller array mutation safety, and append guards."""
    x_base = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=float)
    y_base = np.array([[0.5], [1.5], [2.5]], dtype=float)
    state_base = baux.enn_init(x_base, y_base)
    base_key = state_base.key

    # Branch 1
    x_b1 = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=float)
    y_b1 = np.array([[3.5], [4.5]], dtype=float)
    state_b1 = baux.append(state_base, x_b1, y_b1)

    # Branch 2
    x_b2 = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=float)
    y_b2 = np.array([[10.5], [20.5]], dtype=float)
    state_b2 = baux.append(state_base, x_b2, y_b2)

    # Base state must remain completely unchanged
    assert len(state_base) == 3
    assert len(state_b1) == 5
    assert len(state_b2) == 5
    np.testing.assert_allclose(state_base.x, x_base)
    np.testing.assert_allclose(state_base.y, y_base)
    assert state_base.key == base_key

    # Fit each branch with warm start from base fit
    p_base = baux.fit(state_base, k=2, num_candidates=5, num_samples=3, key=baux.key(1))
    p_b1 = baux.fit(
        state_b1, k=2, num_candidates=5, num_samples=3, warm=p_base, key=baux.key(2)
    )
    p_b2 = baux.fit(
        state_b2, k=2, num_candidates=5, num_samples=3, warm=p_base, key=baux.key(2)
    )

    assert isinstance(p_b1, ENNParams)
    assert isinstance(p_b2, ENNParams)
    # Fitting must not mutate states
    assert len(state_base) == 3
    assert len(state_b1) == 5
    assert len(state_b2) == 5

    # Caller array mutation safety
    mut_x = np.array([[100.0, 200.0]], dtype=float)
    mut_y = np.array([[300.0]], dtype=float)
    appended = baux.append(state_base, mut_x, mut_y)
    mut_x[0, 0] = -999.0
    mut_y[0, 0] = -999.0
    assert appended.x[-1, 0] == 100.0
    assert appended.y[-1, 0] == 300.0

    # Append input validation guards
    with pytest.raises(ValueError):
        # Mismatched dimension (3D instead of 2D)
        baux.append(state_base, [[1.0, 2.0, 3.0]], [[1.0]])

    with pytest.raises(ValueError):
        # Mismatched variance presence (base has no y_var)
        baux.append(state_base, [[1.0, 2.0]], [[1.0]], y_var=[[0.01]])
