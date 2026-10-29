from __future__ import annotations

import cProfile
import pstats
import time
from dataclasses import dataclass
from typing import Any

import click
import numpy as np

from baux import create_optimizer
from baux.turbo.config.acquisition import AcquisitionKind
from baux.turbo.config.candidates import CandidateConfig
from baux.turbo.config.surrogate import SurrogateFit, ENNSurrogate
from baux.turbo.config.factory import turbo_enn
from baux.turbo.config.trust_region import TurboTrustRegion


@dataclass(frozen=True)
class ProfileConfig:
    num_dim: int
    num_obs: int
    num_arms: int
    num_candidates: int | None
    num_samples: int
    fit_candidates: int
    seed: int


@dataclass(frozen=True)
class ProfileActions:
    profile_ask: bool
    time_center: bool


def _makebounds(num_dim: int) -> np.ndarray:
    return np.array([[0.0, 1.0]] * num_dim, dtype=float)


def _makeoptimizer(cfg: ProfileConfig) -> Any:
    rng = np.random.default_rng(cfg.seed)
    bounds = _makebounds(cfg.num_dim)
    enn_cfg = ENNSurrogate(
        k=10,
        fit=SurrogateFit(
            num_samples=cfg.num_samples,
            num_candidates=cfg.fit_candidates,
        ),
    )
    candidates = (
        CandidateConfig(num_candidates=cfg.num_candidates)
        if cfg.num_candidates is not None
        else None
    )
    opt_config = turbo_enn(
        enn=enn_cfg,
        acquisition=AcquisitionKind.UCB,
        trust_region=TurboTrustRegion(),
        num_init=1,
        candidates=candidates,
    )
    return create_optimizer(bounds=bounds, config=opt_config, rng=rng)


def _seedobservations(opt: Any, rng: np.random.Generator, cfg: ProfileConfig) -> None:
    num_obs = int(cfg.num_obs)
    num_dim = int(cfg.num_dim)
    bounds = _makebounds(num_dim)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(num_obs, num_dim))
    y = -np.sum(x**2, axis=1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    opt.tell(x, y)
    # Skip init LHD since we've already seeded observations.
    if hasattr(opt, "_strategy") and hasattr(opt._strategy, "_num_init"):
        opt._strategy._init_idx = opt._strategy._num_init


def _timeask(opt: Any, num_arms: int, repeats: int = 3) -> float:
    # Warmup to ensure model is fitted and FAISS index is built
    _ = opt.ask(num_arms=num_arms)
    t_min = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = opt.ask(num_arms=num_arms)
        t_min = min(t_min, time.perf_counter() - t0)
    return t_min


def _timecenter(opt: Any, rng: np.random.Generator) -> float:
    if not hasattr(opt, "_surrogate"):
        _ = opt.ask(num_arms=1)
        return 0.0
    x_obs = opt._xobs.view()
    y_obs = opt._yobs.view()
    _ = opt._surrogate.fit(x_obs, y_obs, None, num_steps=0, rng=rng)
    t0 = time.perf_counter()
    _ = opt._find_x_center(x_obs, y_obs)
    return time.perf_counter() - t0


def _timefit(opt: Any, rng: np.random.Generator) -> float:
    if not hasattr(opt, "_surrogate"):
        t0 = time.perf_counter()
        _ = opt.ask(num_arms=1)
        return time.perf_counter() - t0
    x_obs = opt._xobs.view()
    y_obs = opt._yobs.view()
    t0 = time.perf_counter()
    _ = opt._surrogate.fit(x_obs, y_obs, None, num_steps=0, rng=rng)
    return time.perf_counter() - t0


def _timetell(opt: Any, rng: np.random.Generator, cfg: ProfileConfig) -> float:
    num_dim = int(cfg.num_dim)
    bounds = _makebounds(num_dim)
    x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(cfg.num_arms, num_dim))
    y = -np.sum(x**2, axis=1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Profile tell() specifically
    prof = cProfile.Profile()
    prof.enable()
    opt.tell(x, y)
    prof.disable()
    if cfg.num_obs >= 8000:
        print(f"\nProfile for tell() at num_obs={cfg.num_obs}:")
        stats = pstats.Stats(prof).sort_stats("tottime")
        stats.print_stats(20)

    t0 = time.perf_counter()
    opt.tell(x, y)
    return time.perf_counter() - t0


def _olsfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError((x.shape, y.shape))
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    y_hat = X @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    return intercept, slope, r2


def _estimatescaling(ns: list[int], times: list[float], label: str) -> None:
    n = np.asarray(ns, dtype=float)
    t = np.asarray(times, dtype=float)
    intercept_n, slope_n, r2_n = _olsfit(n, t)
    intercept_n2, slope_n2, r2_n2 = _olsfit(n**2, t)
    if r2_n2 > r2_n:
        model = "N^2"
        intercept, slope, r2 = intercept_n2, slope_n2, r2_n2
    else:
        model = "N"
        intercept, slope, r2 = intercept_n, slope_n, r2_n
    print(
        f"{label}_scaling",
        f"model={model}",
        f"intercept={intercept:.6e}",
        f"slope={slope:.6e}",
        f"r2={r2:.6f}",
    )


def run_sweep(cfg: ProfileConfig, *, num_obs_values: list[int]) -> None:
    center_times: list[float] = []
    ask_times: list[float] = []
    fit_times: list[float] = []
    tell_times: list[float] = []
    for n in num_obs_values:
        cfg_n = ProfileConfig(
            num_dim=cfg.num_dim,
            num_obs=int(n),
            num_arms=cfg.num_arms,
            num_candidates=cfg.num_candidates,
            num_samples=cfg.num_samples,
            fit_candidates=cfg.fit_candidates,
            seed=cfg.seed,
        )
        rng = np.random.default_rng(cfg_n.seed)
        opt = _makeoptimizer(cfg_n)
        _seedobservations(opt, rng, cfg_n)
        center_times.append(_timecenter(opt, rng))
        fit_times.append(_timefit(opt, rng))
        ask_times.append(_timeask(opt, cfg_n.num_arms))
        tell_times.append(_timetell(opt, rng, cfg_n))
        print(
            "sweep",
            "num_obs",
            cfg_n.num_obs,
            "center_time_sec",
            f"{center_times[-1]:.6f}",
            "fit_time_sec",
            f"{fit_times[-1]:.6f}",
            "ask_time_sec",
            f"{ask_times[-1]:.6f}",
            "tell_time_sec",
            f"{tell_times[-1]:.6f}",
        )
    _estimatescaling(num_obs_values, center_times, "center_time_sec")
    _estimatescaling(num_obs_values, fit_times, "fit_time_sec")
    _estimatescaling(num_obs_values, ask_times, "ask_time_sec")
    _estimatescaling(num_obs_values, tell_times, "tell_time_sec")


def run_profile(cfg: ProfileConfig, actions: ProfileActions) -> None:
    rng = np.random.default_rng(cfg.seed)
    opt = _makeoptimizer(cfg)
    _seedobservations(opt, rng, cfg)

    # Warmup to avoid import overhead in profile
    _ = opt.ask(num_arms=cfg.num_arms)

    if actions.time_center:
        dt_center = _timecenter(opt, rng)
        print(
            "center_time_sec",
            f"{dt_center:.6f}",
            "num_obs",
            cfg.num_obs,
        )

    if actions.profile_ask:
        prof = cProfile.Profile()
        prof.enable()
        _ = opt.ask(num_arms=cfg.num_arms)
        prof.disable()
        stats = pstats.Stats(prof).sort_stats("tottime")
        stats.print_stats(30)
    else:
        dt = _timeask(opt, cfg.num_arms)
        print(
            "ask_time_sec",
            f"{dt:.6f}",
            "num_obs",
            cfg.num_obs,
            "num_candidates",
            cfg.num_candidates,
            "num_samples",
            cfg.num_samples,
            "fit_candidates",
            cfg.fit_candidates,
        )


def _sweepvalues(
    _context: click.Context, _parameter: click.Parameter, value: str
) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise click.BadParameter("must contain comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("must contain positive integers")
    return values


def main(**kwargs: Any) -> None:
    sweep = bool(kwargs.pop("sweep"))
    sweep_values = kwargs.pop("sweep_values")
    profile_ask = bool(kwargs.pop("profile_ask"))
    time_center = bool(kwargs.pop("time_center"))
    cfg = ProfileConfig(**kwargs)
    if sweep:
        run_sweep(cfg, num_obs_values=sweep_values)
    else:
        run_profile(
            cfg,
            ProfileActions(
                profile_ask=profile_ask,
                time_center=time_center,
            ),
        )


for _decorator in (
    click.option(
        "--sweep-values",
        callback=_sweepvalues,
        default="100,300,1000,3000,10000",
        show_default=True,
        help="Comma-separated observation counts.",
    ),
    click.option(
        "--sweep", is_flag=True, help="Estimate scaling over observation counts."
    ),
    click.option("--profile-center", "time_center", is_flag=True),
    click.option("--profile", "profile_ask", is_flag=True),
    click.option("--seed", type=int, default=0, show_default=True),
    click.option(
        "--num-fit-candidates",
        "fit_candidates",
        type=click.IntRange(min=1),
        default=60,
        show_default=True,
    ),
    click.option(
        "--num-fit-samples",
        "num_samples",
        type=click.IntRange(min=1),
        default=50,
        show_default=True,
    ),
    click.option("--num-candidates", type=click.IntRange(min=1)),
    click.option(
        "--num-arms", type=click.IntRange(min=1), default=4, show_default=True
    ),
    click.option(
        "--num-obs", type=click.IntRange(min=1), default=2000, show_default=True
    ),
    click.option("--num-dim", type=click.IntRange(min=1), default=8, show_default=True),
    click.command(help="Profile TuRBO-ENN ask() with ENN fitting enabled."),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
