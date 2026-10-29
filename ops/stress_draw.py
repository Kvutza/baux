"""Draw stress runner, multi-seed aggregation, and output formatting."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from baux.enn.enn_class import ENN
from baux.enn.enn_fit import enn_fit
from ops.stress_disk import (
    DRAW_FLAGS,
    DRAW_FLAGS_NO_OBS,
    DURATION_S_FMT,
    function_sample,
)
from ops.stress_sample import (
    DrawMethodResult,
    DrawStressConfig,
    argmin_rate,
    argmin_rms,
    average_draws,
    average_likelihood,
    make_observations2,
)


@dataclass(frozen=True)
class DrawStressResult:
    num_obs: int
    num_test: int
    num_dim: int
    seed: int
    k: int
    num_candidates: int
    num_samples: int
    num_draws: int
    epistemic_scale: float
    aleatoric_scale: float
    fit_s: float
    posterior: DrawMethodResult
    posterior_draw: DrawMethodResult


def _checkdrawcfg(config: DrawStressConfig) -> None:
    for name in (
        "num_obs",
        "num_test",
        "num_dim",
        "k",
        "num_candidates",
        "num_samples",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    if config.num_draws < 2:
        raise ValueError("num_draws must be >= 2")


def _evalpost(
    model: ENN,
    fitted,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_draws: int,
    sample_rng: np.random.Generator,
) -> DrawMethodResult:
    t0 = time.perf_counter()
    post_lik = model.posterior(x_test, params=fitted, flags=DRAW_FLAGS)
    avg_lik = average_likelihood(y_test, post_lik.mu, post_lik.se)
    post_rms = model.posterior(x_test, params=fitted, flags=DRAW_FLAGS_NO_OBS)
    draws = post_rms.sample(num_draws, sample_rng)
    rms = argmin_rms(x_test, draws)
    rate = argmin_rate(x_test, draws)
    eval_s = time.perf_counter() - t0
    return DrawMethodResult(
        method="posterior",
        avg_likelihood=avg_lik,
        argmin_rms=rms,
        argmin_rate=rate,
        draws_shape=tuple(int(s) for s in draws.shape),
        all_finite=bool(np.all(np.isfinite(draws))),
        eval_s=eval_s,
    )


def _evalfn(
    model: ENN,
    fitted,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    num_draws: int,
) -> DrawMethodResult:
    t0 = time.perf_counter()
    fn_draws, _idx = model.posterior_draw(
        x_test,
        fitted,
        function_seeds=function_sample(seed + 4, num_draws),
        flags=DRAW_FLAGS_NO_OBS,
    )
    avg_lik = average_draws(y_test, fn_draws)
    rms = argmin_rms(x_test, fn_draws)
    rate = argmin_rate(x_test, fn_draws)
    eval_s = time.perf_counter() - t0
    return DrawMethodResult(
        method="posterior_draw",
        avg_likelihood=avg_lik,
        argmin_rms=rms,
        argmin_rate=rate,
        draws_shape=tuple(int(s) for s in fn_draws.shape),
        all_finite=bool(np.all(np.isfinite(fn_draws))),
        eval_s=eval_s,
    )


def run_stress2(config: DrawStressConfig) -> DrawStressResult:
    """Fit ENN on synthetic draw data; score avg likelihood and argmin-RMS."""
    _checkdrawcfg(config)
    data_rng = np.random.default_rng(config.seed)
    fit_rng = np.random.default_rng(config.seed + 1)
    sample_rng = np.random.default_rng(config.seed + 2)
    x, y = make_observations2(config.num_obs, num_dim=config.num_dim, rng=data_rng)
    x_test, y_test = make_observations2(
        config.num_test, num_dim=config.num_dim, rng=data_rng
    )
    model = ENN(x, y, scale_x=False)
    t0 = time.perf_counter()
    fitted = enn_fit(
        model,
        k=config.k,
        num_candidates=config.num_candidates,
        num_samples=config.num_samples,
        rng=fit_rng,
    )
    fit_s = time.perf_counter() - t0
    post_res = _evalpost(model, fitted, x_test, y_test, config.num_draws, sample_rng)
    fn_res = _evalfn(model, fitted, x_test, y_test, config.seed, config.num_draws)
    return DrawStressResult(
        num_obs=config.num_obs,
        num_test=config.num_test,
        num_dim=config.num_dim,
        seed=config.seed,
        k=config.k,
        num_candidates=config.num_candidates,
        num_samples=config.num_samples,
        num_draws=config.num_draws,
        epistemic_scale=float(fitted.epistemic_scale),
        aleatoric_scale=float(fitted.aleatoric_scale),
        fit_s=fit_s,
        posterior=post_res,
        posterior_draw=fn_res,
    )


@dataclass(frozen=True)
class MeanSE:
    mean: float
    se: float


@dataclass(frozen=True)
class DrawMethodAggregate:
    method: str
    avg_likelihood: MeanSE
    argmin_rms: MeanSE
    argmin_rate: MeanSE
    eval_s: MeanSE


@dataclass(frozen=True)
class DrawStressAggregate:
    num_obs: int
    num_test: int
    num_dim: int
    seed: int
    num_seeds: int
    k: int
    num_candidates: int
    num_samples: int
    num_draws: int
    epistemic_scale: MeanSE
    aleatoric_scale: MeanSE
    fit_s: MeanSE
    posterior: DrawMethodAggregate
    posterior_draw: DrawMethodAggregate


def mean_se(values: np.ndarray | list[float]) -> MeanSE:
    """Sample mean and standard error of the mean (nan SE if fewer than 2 values)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 1:
        raise ValueError("values must be non-empty")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return MeanSE(mean=mean, se=float("nan"))
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return MeanSE(mean=mean, se=se)


def format_se(stat: MeanSE, *, fmt: str = ".6g") -> str:
    if not np.isfinite(stat.se):
        return format(stat.mean, fmt)
    return f"{stat.mean:{fmt}} ± {stat.se:{fmt}}"


def _aggregatemethod(
    method: str, results: list[DrawMethodResult]
) -> DrawMethodAggregate:
    return DrawMethodAggregate(
        method=method,
        avg_likelihood=mean_se([r.avg_likelihood for r in results]),
        argmin_rms=mean_se([r.argmin_rms for r in results]),
        argmin_rate=mean_se([r.argmin_rate for r in results]),
        eval_s=mean_se([r.eval_s for r in results]),
    )


def run_seeds(config: DrawStressConfig, *, num_seeds: int) -> DrawStressAggregate:
    """Run ``run_stress2`` for ``seed .. seed+num_seeds-1`` and aggregate metrics."""
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    results = [
        run_stress2(
            DrawStressConfig(
                num_obs=config.num_obs,
                num_test=config.num_test,
                num_dim=config.num_dim,
                seed=config.seed + i,
                k=config.k,
                num_candidates=config.num_candidates,
                num_samples=config.num_samples,
                num_draws=config.num_draws,
            )
        )
        for i in range(num_seeds)
    ]
    return DrawStressAggregate(
        num_obs=config.num_obs,
        num_test=config.num_test,
        num_dim=config.num_dim,
        seed=config.seed,
        num_seeds=num_seeds,
        k=config.k,
        num_candidates=config.num_candidates,
        num_samples=config.num_samples,
        num_draws=config.num_draws,
        epistemic_scale=mean_se([r.epistemic_scale for r in results]),
        aleatoric_scale=mean_se([r.aleatoric_scale for r in results]),
        fit_s=mean_se([r.fit_s for r in results]),
        posterior=_aggregatemethod("posterior", [r.posterior for r in results]),
        posterior_draw=_aggregatemethod(
            "posterior_draw", [r.posterior_draw for r in results]
        ),
    )


def format_header3(result: DrawStressResult) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"num_test={result.num_test} seed={result.seed} k={result.k} "
        f"num_draws={result.num_draws} "
        f"epistemic_scale={result.epistemic_scale:.6g} "
        f"aleatoric_scale={result.aleatoric_scale:.6g} "
        f"fit_s={result.fit_s:{DURATION_S_FMT}}"
    )


def format_summary2(method: DrawMethodResult) -> str:
    return (
        f"{method.method} avg_likelihood={method.avg_likelihood:.6g} "
        f"argmin_rms={method.argmin_rms:.6g} "
        f"argmin_rate={method.argmin_rate:0.4f} "
        f"eval_s={method.eval_s:{DURATION_S_FMT}}"
    )


def format_aggregate(result: DrawStressAggregate) -> str:
    return (
        f"num_dim={result.num_dim} num_obs={result.num_obs} "
        f"num_test={result.num_test} seed={result.seed} "
        f"num_seeds={result.num_seeds} k={result.k} "
        f"num_draws={result.num_draws} "
        f"epistemic_scale={format_se(result.epistemic_scale)} "
        f"aleatoric_scale={format_se(result.aleatoric_scale)} "
        f"fit_s={format_se(result.fit_s, fmt=DURATION_S_FMT)}"
    )


def format_aggregate2(method: DrawMethodAggregate) -> str:
    return (
        f"{method.method} "
        f"avg_likelihood={format_se(method.avg_likelihood)} "
        f"argmin_rms={format_se(method.argmin_rms)} "
        f"argmin_rate={format_se(method.argmin_rate, fmt='0.4f')} "
        f"eval_s={format_se(method.eval_s, fmt=DURATION_S_FMT)}"
    )
