"""Sample stress and draw likelihood/argmin metrics for ENN."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ops.stress_disk import (
    DEFAULT_DRAW_K,
    DEFAULT_DRAW_NUM_DRAWS,
    DEFAULT_DRAW_NUM_FIT_CANDIDATES,
    DEFAULT_DRAW_NUM_FIT_SAMPLES,
    DEFAULT_DRAW_SEED,
    DEFAULT_NUM_DIM,
    DEFAULT_SAMPLE_X_HIGH,
    DEFAULT_SAMPLE_X_LOW,
    DRAW_F_CENTER,
    DRAW_OBS_NOISE_STD,
    DURATION_S_FMT,
    NUM_SAMPLE,
    STRESS_PARAMS,
    STRESS_QUERY_SEED,
    bpann_enn,
    function_sample,
    make_points2,
)


@dataclass(frozen=True)
class SampleStressConfig:
    num_samples: int = NUM_SAMPLE
    seed: int = STRESS_QUERY_SEED
    x_low: float = DEFAULT_SAMPLE_X_LOW
    x_high: float = DEFAULT_SAMPLE_X_HIGH


@dataclass(frozen=True)
class SampleStressResult:
    num_dim: int
    num_obs: int
    num_samples: int
    seed: int
    num_function_seeds: int
    draws_shape: tuple[int, ...]
    all_finite: bool
    init_s: float
    sample_s: float


def run_stress(
    *,
    work_dir: str,
    config: SampleStressConfig | None = None,
) -> SampleStressResult:
    """Reopen disk ENN and draw posterior function samples at uniform random x."""
    cfg = config if config is not None else SampleStressConfig()
    if cfg.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    t0 = time.perf_counter()
    model, meta = bpann_enn(work_dir)
    num_dim = int(meta["num_dim"])
    x_query = make_points2(
        cfg.num_samples,
        num_dim=num_dim,
        seed=cfg.seed,
        low=cfg.x_low,
        high=cfg.x_high,
    )
    init_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    function_seeds = function_sample(cfg.seed)
    draws, _idx = model.posterior_draw(
        x_query,
        STRESS_PARAMS,
        function_seeds=function_seeds,
    )
    sample_s = time.perf_counter() - t1
    return SampleStressResult(
        num_dim=num_dim,
        num_obs=len(model),
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        num_function_seeds=len(function_seeds),
        draws_shape=tuple(int(s) for s in draws.shape),
        all_finite=bool(np.all(np.isfinite(draws))),
        init_s=init_s,
        sample_s=sample_s,
    )


def format_header2(*, result: SampleStressResult, work_dir: str) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"work_dir={work_dir} num_samples={result.num_samples} seed={result.seed}"
    )


def format_summary(result: SampleStressResult) -> str:
    return (
        f"draws_shape={result.draws_shape} function_seeds={result.num_function_seeds} "
        f"all_finite={str(result.all_finite).lower()} "
        f"init_s={result.init_s:{DURATION_S_FMT}} "
        f"sample_s={result.sample_s:{DURATION_S_FMT}}"
    )


def draw_f(x: np.ndarray) -> np.ndarray:
    """Return sum_j (x_ij - DRAW_F_CENTER)^2 with shape (n, 1)."""
    x_arr = np.asarray(x, dtype=float)
    return np.sum((x_arr - DRAW_F_CENTER) ** 2, axis=1, keepdims=True)


def _drawindices(x_test: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Return per-sample argmin indices over metric 0; ``draws`` is ``(B, M, S)``."""
    x_arr = np.asarray(x_test, dtype=float)
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(f"draws must be 3D (B, M, S), got shape {draws_arr.shape}")
    if draws_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"x_test rows ({x_arr.shape[0]}) must match draws batch ({draws_arr.shape[0]})"
        )
    if draws_arr.shape[1] < 1:
        raise ValueError("draws must have at least one metric")
    if draws_arr.shape[2] < 1:
        raise ValueError("draws must have at least one sample")
    return np.argmin(draws_arr[:, 0, :], axis=0)


def argmin_rms(x_test: np.ndarray, draws: np.ndarray) -> float:
    """RMS of ||x_hat - DRAW_F_CENTER||_2 over draws; x_hat = argmin of metric 0."""
    x_arr = np.asarray(x_test, dtype=float)
    i_star = _drawindices(x_test, draws)
    eps = x_arr[i_star] - DRAW_F_CENTER
    return float(np.sqrt(np.mean(np.sum(eps * eps, axis=-1))))


def argmin_rate(x_test: np.ndarray, draws: np.ndarray) -> float:
    """Fraction of draws whose metric-0 argmin matches ``argmin_i draw_f(x_test)_i``."""
    i_true = int(np.argmin(draw_f(x_test)[:, 0]))
    i_star = _drawindices(x_test, draws)
    return float(np.mean(i_star == i_true))


def make_observations2(
    num_obs: int,
    *,
    num_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw x ~ U[0,1]^num_dim and y = draw_f(x) + 0.1 N(0,1)."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    x = rng.uniform(0.0, 1.0, size=(num_obs, num_dim))
    noise = rng.standard_normal((num_obs, 1))
    return x, draw_f(x) + DRAW_OBS_NOISE_STD * noise


def gaussian_likelihood(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Per-point N(y; mu, se^2) densities."""
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    z = (y_arr - mu_arr) / se_arr
    return (1.0 / (se_arr * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def average_likelihood(y: np.ndarray, mu: np.ndarray, se: np.ndarray) -> float:
    """Mean of per-point Gaussian predictive densities."""
    return float(np.mean(gaussian_likelihood(y, mu, se)))


def average_draws(y: np.ndarray, draws: np.ndarray) -> float:
    """Mean Gaussian density using empirical mean/std over sample axis -1."""
    draws_arr = np.asarray(draws, dtype=float)
    if draws_arr.ndim != 3:
        raise ValueError(f"draws must be 3D (B, M, S), got shape {draws_arr.shape}")
    if draws_arr.shape[-1] < 2:
        raise ValueError("draws must have at least 2 samples along axis -1")
    mu = draws_arr.mean(axis=-1)
    se = np.maximum(draws_arr.std(axis=-1, ddof=1), 1e-12)
    return average_likelihood(y, mu, se)


@dataclass(frozen=True)
class DrawStressConfig:
    num_obs: int
    num_test: int
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = DEFAULT_DRAW_SEED
    k: int = DEFAULT_DRAW_K
    num_candidates: int = DEFAULT_DRAW_NUM_FIT_CANDIDATES
    num_samples: int = DEFAULT_DRAW_NUM_FIT_SAMPLES
    num_draws: int = DEFAULT_DRAW_NUM_DRAWS


@dataclass(frozen=True)
class DrawMethodResult:
    method: str
    avg_likelihood: float
    argmin_rms: float
    argmin_rate: float
    draws_shape: tuple[int, ...]
    all_finite: bool
    eval_s: float
