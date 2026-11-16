"""ENN model construction and hyperparameter fitting helpers for proposals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from baux.enn.enn_params import ENNParams
    from baux.enn.enn_state import ENNState
    from numpy.random import Generator

    from .config.surrogate import NeighborIndex, SurrogateFit


def mk_enn(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    k: int,
    yvar_obs: np.ndarray | None = None,
    *,
    fit: SurrogateFit | None = None,
    scale_x: bool = False,
    index_driver: NeighborIndex | None = None,
    rng: Generator | None = None,
    params_warm_start: ENNParams | None = None,
) -> tuple[ENNState | None, ENNParams | None]:
    """Construct an ENN surrogate from observations and optionally fit its parameters."""
    from baux.enn.enn_params import ENNParams
    from baux.enn.enn_state import enn_init
    from baux.enn.enn_state import fit as fit_enn

    from .config.surrogate import NeighborIndex

    if index_driver is None:
        index_driver = NeighborIndex.FLAT

    x_obs_array = np.asarray(x_obs, dtype=float)
    if x_obs_array.size == 0:
        return None, None
    y_obs_array = np.asarray(y_obs, dtype=float)
    if y_obs_array.size == 0:
        return None, None
    y = y_obs_array.reshape(-1, 1) if y_obs_array.ndim == 1 else y_obs_array
    yvar = None
    if yvar_obs is not None:
        yvar_array = np.asarray(yvar_obs, dtype=float)
        if yvar_array.size > 0:
            yvar = yvar_array.reshape(-1, 1) if yvar_array.ndim == 1 else yvar_array
    enn_state = enn_init(
        x_obs_array,
        y,
        yvar,
        scale_x=scale_x,
        index_driver=index_driver,
    )
    if len(enn_state) == 0:
        return None, None
    if fit is not None and fit.num_samples is not None and rng is not None:
        num_candidates = fit.num_candidates if fit.num_candidates is not None else 30
        fitted_params = fit_enn(
            enn_state,
            k=k,
            num_candidates=num_candidates,
            num_samples=fit.num_samples,
            infer_aleatoric=fit.infer_aleatoric_variance_scale,
            seed=rng,
            warm=params_warm_start,
        )
    else:
        fitted_params = ENNParams(
            k_neighbors=k,
            epistemic_scale=1.0,
            aleatoric_scale=0.0,
        )
    return enn_state, fitted_params
