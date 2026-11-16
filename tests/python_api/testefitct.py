"""Contract tests for pure functional ENN hyperparameter fitting."""

from __future__ import annotations

import inspect

import baux
import numpy as np
from baux import ENNParams


class Test_001:
    def test_fitsignature(self):
        sig = inspect.signature(baux.fit)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "k" in params
        assert "num_candidates" in params
        assert "num_samples" in params
        assert "infer_aleatoric" in params

    def test_fitreturnsenparams(self):
        train_x = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=float,
        )
        train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
        state = baux.enn_init(train_x, train_y, scale_x=False)

        result = baux.fit(
            state,
            k=2,
            num_candidates=3,
            num_samples=2,
            seed=42,
        )

        assert isinstance(result, ENNParams)
        assert result.k_neighbors == 2
        assert result.epistemic_scale > 0
        assert result.aleatoric_scale >= 0
