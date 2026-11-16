"""Hyperparameter fitting and log-likelihood scoring for ENN models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from baux._rust import row_loglik as rust_rows
from baux._rust import subsample_loglik as _rust_subsample_loglik

if TYPE_CHECKING:
    from numpy.random import Generator

    from .enn_params import ENNParams
    from .enn_state import ENNState


def subsample_loglik(
    state: ENNState,
    x: np.ndarray,
    y: np.ndarray,
    *,
    paramss: list[ENNParams],
    P: int = 10,
    rng: Generator,
    y_std: np.ndarray | None = None,
) -> list[float]:
    """Score complete training arrays in model row order; use row_loglik for subsets."""
    from .enn_state import ENNState as StateCls

    if not isinstance(state, StateCls):
        raise TypeError(f"Expected ENNState, got {type(state)}")

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    seed = int(rng.integers(0, 2**63 - 1))

    k_values = [int(p.k_neighbors) for p in paramss]
    epi_scales = [float(p.epistemic_scale) for p in paramss]
    ale_scales = [float(p.aleatoric_scale) for p in paramss]

    y_std_arr = None
    if y_std is not None:
        y_std_arr = np.asarray(y_std, dtype=float).ravel()

    return _rust_subsample_loglik(
        state._rust,
        x_array,
        y_array,
        k_values,
        epi_scales,
        ale_scales,
        int(P),
        seed,
        y_std_arr,
    )


def row_loglik(
    state: ENNState,
    rows: list[int],
    *,
    paramss: list[ENNParams],
    P: int = 10,
    rng: Generator,
    y_std: np.ndarray | None = None,
) -> list[float]:
    """Score held-out model row IDs without ambiguous coordinate matching."""
    from .enn_state import ENNState as StateCls

    if not isinstance(state, StateCls):
        raise TypeError(f"Expected ENNState, got {type(state)}")

    return rust_rows(
        state._rust,
        [int(r) for r in rows],
        [int(p.k_neighbors) for p in paramss],
        [float(p.epistemic_scale) for p in paramss],
        [float(p.aleatoric_scale) for p in paramss],
        int(P),
        int(rng.integers(0, 2**63 - 1)),
        None if y_std is None else np.asarray(y_std, dtype=float).ravel(),
    )


__all__ = [
    "row_loglik",
    "subsample_loglik",
]
