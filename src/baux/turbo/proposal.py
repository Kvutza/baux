from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from baux.enn.enn_class import ENN
    from baux.enn.enn_params import ENNParams

    from .config.surrogate import SurrogateFit
    from .config.surrogate import NeighborIndex


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
) -> tuple[ENN | None, ENNParams | None]:
    from baux.enn.enn_class import ENN
    from baux.enn.enn_params import ENNParams

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
    enn_model = ENN(
        x_obs_array,
        y,
        yvar,
        scale_x=scale_x,
        index_driver=index_driver,
    )
    if len(enn_model) == 0:
        return None, None
    fitted_params: ENNParams | None = None
    if fit is not None and fit.num_samples is not None and rng is not None:
        from baux.enn.enn_fitter import ENNFitter

        fitter = ENNFitter(
            k=k,
            rng=rng,
            infer_aleatoric_variance_scale=fit.infer_aleatoric_variance_scale,
        )
        fitter.tell(x_obs_array, y, yvar)
        fitted_params = fitter.ask(
            enn_model,
            num_candidates=(
                fit.num_candidates if fit.num_candidates is not None else 30
            ),
            num_samples=fit.num_samples,
            params_warm_start=params_warm_start,
        )
    else:
        fitted_params = ENNParams(
            k_neighbors=k,
            epistemic_scale=1.0,
            aleatoric_scale=0.0,
        )
    return enn_model, fitted_params
