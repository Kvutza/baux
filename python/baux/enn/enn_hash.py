"""Deterministic normal hash functions for ENN posterior function draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from baux._rust import normal_hash as _rust_hash_fast


def _ndtri(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Wichura AS 241 PPND16, float64 precision)."""
    import numpy as np

    a = (
        3.3871328727963666080e00,
        1.3314166789178437745e02,
        1.9715909503065514427e03,
        1.3731693765509461125e04,
        4.5921953931549871457e04,
        6.7265770927008700853e04,
        3.3430575583588128105e04,
        2.5090809287301226727e03,
    )
    b = (
        1.0,
        4.2313330701600911252e01,
        6.8718700749205790830e02,
        5.3941960214247511077e03,
        2.1213794301586595867e04,
        3.9307895800092710610e04,
        2.8729085735721942674e04,
        5.2264952788528545610e03,
    )
    c = (
        1.42343711074968357734e00,
        4.63033784615654529590e00,
        5.76949722146069140550e00,
        3.64784832476320460504e00,
        1.27045825245236838258e00,
        2.41780725177450611770e-01,
        2.27238449892691845833e-02,
        7.74545014278341407640e-04,
    )
    d = (
        1.0,
        2.05319162663775882187e00,
        1.67638483018380384940e00,
        6.89767334985100004550e-01,
        1.48103976427480074590e-01,
        1.51986665636164571966e-02,
        5.47593808499534494600e-04,
        1.05075007164441684324e-09,
    )
    e = (
        6.65790464350110377720e00,
        5.46378491116411436990e00,
        1.78482653991729133580e00,
        2.96560571828504891230e-01,
        2.65321895265761230930e-02,
        1.24266094738807843860e-03,
        2.71155556874348757815e-05,
        2.01033439929228813265e-07,
    )
    f = (
        1.0,
        5.99832206555887937690e-01,
        1.36929880922735805310e-01,
        1.48753612908506148525e-02,
        7.86869131145613259100e-04,
        1.84631831751005468180e-05,
        1.42151175831644588870e-07,
        2.04426310338993978564e-15,
    )

    def _poly(coeffs: tuple[float, ...], x: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x, dtype=np.float64)
        for coeff in reversed(coeffs):
            acc = acc * x + coeff
        return acc

    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    out = np.empty_like(p)
    central = np.abs(q) <= 0.425
    if np.any(central):
        qc = q[central]
        r = 0.180625 - qc * qc
        out[central] = qc * _poly(a, r) / _poly(b, r)
    tail = ~central
    if np.any(tail):
        pt = p[tail]
        qt = q[tail]
        r = np.sqrt(-np.log(np.where(qt < 0.0, pt, 1.0 - pt)))
        sign = np.where(qt < 0.0, -1.0, 1.0)
        mid = r <= 5.0
        vals = np.empty_like(r)
        if np.any(mid):
            rm = r[mid] - 1.6
            vals[mid] = _poly(c, rm) / _poly(d, rm)
        far = ~mid
        if np.any(far):
            rf = r[far] - 5.0
            vals[far] = _poly(e, rf) / _poly(f, rf)
        out[tail] = sign * vals
    return out


def normal_philox(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    """Generate deterministic standard normal draws via Philox counter-based RNG."""
    import numpy as np

    num_seeds = len(function_seeds)
    unique_indices, inverse = np.unique(data_indices, return_inverse=True)
    num_unique = len(unique_indices)
    seed_grid, idx_grid, metric_grid = np.meshgrid(
        function_seeds.astype(np.uint64),
        unique_indices.astype(np.uint64),
        np.arange(num_metrics, dtype=np.uint64),
        indexing="ij",
    )
    seed_flat = seed_grid.ravel()
    idx_flat = idx_grid.ravel()
    metric_flat = metric_grid.ravel()
    combined_seeds = (seed_flat * np.uint64(1_000_003) + idx_flat) * np.uint64(
        1_000_003
    ) + metric_flat
    uniform_vals = np.empty(len(combined_seeds), dtype=float)
    for i, seed in enumerate(combined_seeds):
        rng = np.random.Generator(np.random.Philox(int(seed)))
        uniform_vals[i] = rng.random()
    uniform_vals = np.clip(uniform_vals, 1e-10, 1.0 - 1e-10)
    normal_vals = _ndtri(uniform_vals).reshape(num_seeds, num_unique, num_metrics)
    return normal_vals[:, inverse.ravel(), :].reshape(
        num_seeds, *data_indices.shape, num_metrics
    )


def normal_hash(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    """Generate deterministic standard normal draws using the Rust hash backend."""
    import numpy as np

    function_seeds = np.asarray(function_seeds, dtype=np.int64)
    data_indices = np.asarray(data_indices)
    if num_metrics <= 0:
        raise ValueError(num_metrics)
    return np.asarray(
        _rust_hash_fast(function_seeds, data_indices, num_metrics), dtype=float
    )
