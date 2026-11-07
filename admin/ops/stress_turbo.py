"""TuRBO-ENN ask/tell scaling and proposal-scale stress runners."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baux.turbo.config.surrogate import NeighborIndex
from ops.stress_disk import (
    DURATION_S_FMT,
    PROPOSAL_SCALE_NS,
    PROPOSAL_SCALE_NUM_PROBES,
    PROPOSAL_SCALE_SEED_CHUNK,
    PROPOSAL_SCALE_WARMUP,
    TURBO_ENN_ACKLEY_NOISE,
    TURBO_ENN_K,
    TURBO_ENN_NUM_FIT_SAMPLES,
    TURBO_ENN_NUM_INIT,
    TURBO_ENN_SEED,
    enn_chunk,
)


@dataclass(frozen=True)
class TurboEnnRoundResult:
    n: int
    iter_s: float
    ask_s: float
    tell_s: float


def enn_stops(num_obs: int, num_ask: int) -> tuple[int, ...]:
    """Return NUM_ASK exponentially spaced cumulative observation stops in [1, NUM_OBS]."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if num_ask < 1:
        raise ValueError("num_ask must be >= 1")
    if num_ask > num_obs:
        raise ValueError("num_ask must be <= num_obs")
    if num_ask == 1:
        return (num_obs,)
    ideal = np.exp(np.linspace(np.log(1.0), np.log(float(num_obs)), num=num_ask))
    stops: list[int] = []
    for i, value in enumerate(ideal):
        lo = (stops[-1] + 1) if stops else 1
        hi = num_obs - (num_ask - i - 1)
        assert lo <= hi, f"no room for stop {i}: lo={lo} hi={hi}"
        cand = min(max(round(float(value)), lo), hi)
        stops.append(cand)
    assert len(stops) == num_ask and stops[-1] == num_obs
    assert all(stops[i] < stops[i + 1] for i in range(num_ask - 1)), stops
    return tuple(stops)


def enn_config(
    *,
    index_driver: NeighborIndex,
    work_dir: str | None = None,
    num_init: int = TURBO_ENN_NUM_INIT,
):
    """Build turbo_enn config matching compare's single-metric Ackley path."""
    from baux.turbo.config import (
        AcquisitionKind,
        ENNSurrogate,
        SurrogateFit,
        TurboTrustRegion,
        turbo_enn,
    )

    if num_init < 1:
        raise ValueError("num_init must be >= 1")
    enn_kwargs: dict[str, Any] = {
        "k": TURBO_ENN_K,
        "fit": SurrogateFit(num_samples=TURBO_ENN_NUM_FIT_SAMPLES),
        "index_driver": index_driver,
    }
    if index_driver == NeighborIndex.BPANN_DISK:
        if work_dir is None:
            raise ValueError("bpann_disk requires work_dir")
        enn_kwargs["enn_storage"] = "disk"
        enn_kwargs["work_dir"] = work_dir
    elif work_dir is not None:
        raise ValueError("work_dir requires bpann_disk")
    return turbo_enn(
        enn=ENNSurrogate(**enn_kwargs),
        trust_region=TurboTrustRegion(noise_aware=True),
        acquisition=AcquisitionKind.UCB,
        num_init=num_init,
    )


def enn_header(
    *,
    num_dim: int,
    num_obs: int,
    num_ask: int,
    index_type: str,
    work_dir: str | None = None,
    tell_all: bool = False,
) -> str:
    header = (
        f"num_dim={num_dim} num_obs={num_obs} num_ask={num_ask} index_type={index_type}"
    )
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    if tell_all:
        header = f"{header} tell_all=true"
    return header


def enn_row(n: int, iter_s: float, ask_s: float, tell_s: float, *, n_width: int) -> str:
    return (
        f"{n:>{n_width}} {iter_s:{DURATION_S_FMT}} "
        f"{ask_s:{DURATION_S_FMT}} {tell_s:{DURATION_S_FMT}}"
    )


def _initturborun(
    index_driver: NeighborIndex,
    num_dim: int,
    work_dir: str | None,
    seed: int,
    options: dict[str, Any],
) -> tuple[Any, Any, np.ndarray, int]:
    from baux import create_optimizer
    from baux.benchmarks import Ackley

    seed_chunk = options.get("seed_chunk")
    if seed_chunk is None:
        seed_chunk = enn_chunk(index_driver)
    if options.get("tell_all", False):
        seed_chunk = 1
    if seed_chunk < 1:
        raise ValueError("seed_chunk must be >= 1")
    objective = options.get("objective")
    ackley = (
        Ackley(noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(seed))
        if objective is None
        else objective
    )
    raw_bounds = options.get("bounds")
    bounds_arr = (
        np.array([ackley.bounds] * num_dim, dtype=float)
        if raw_bounds is None
        else np.asarray(raw_bounds, dtype=float)
    )
    opt = options.get("optimizer")
    if opt is None:
        config = enn_config(
            index_driver=index_driver,
            work_dir=work_dir,
            num_init=TURBO_ENN_NUM_INIT,
        )
        opt = create_optimizer(
            bounds=bounds_arr, config=config, rng=np.random.default_rng(seed)
        )
    return opt, ackley, bounds_arr, int(seed_chunk)


def enn_stress2(
    index_driver: NeighborIndex,
    num_dim: int,
    num_obs: int,
    num_ask: int,
    *,
    work_dir: str | None = None,
    seed: int = TURBO_ENN_SEED,
    seed_chunk: int | None = None,
    **options: Any,
) -> Iterator[TurboEnnRoundResult]:
    """Tell DGP obs in exponential groups; at each stop time ask(1) and discard arms."""
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    stops = enn_stops(num_obs, num_ask)
    options["seed_chunk"] = seed_chunk
    opt, ackley, bounds_arr, seed_chunk = _initturborun(
        index_driver, num_dim, work_dir, seed, options
    )
    dgp_rng = np.random.default_rng(seed + 17)
    prev = 0
    for stop in stops:
        gap = stop - prev
        assert gap >= 1, f"empty group at stop={stop} prev={prev}"
        t0 = time.perf_counter()
        enn_n(opt, ackley, bounds_arr, gap, rng=dgp_rng, chunk=seed_chunk)
        tell_s = time.perf_counter() - t0
        t_ask0 = time.perf_counter()
        _ = opt.ask(num_arms=1)
        ask_s = time.perf_counter() - t_ask0
        yield TurboEnnRoundResult(
            n=stop, iter_s=ask_s + tell_s, ask_s=ask_s, tell_s=tell_s
        )
        prev = stop


@dataclass(frozen=True)
class ProposalScaleResult:
    n: int
    ask_s: float
    tell_s: float
    proposal_s: float


def proposal_ns(max_n: int) -> tuple[int, ...]:
    """Return fixed proposal-scale grid filtered by ``max_n`` (all N >= num_init)."""
    if max_n < TURBO_ENN_NUM_INIT:
        raise ValueError(f"max_n must be >= {TURBO_ENN_NUM_INIT} (TuRBO-ENN num_init)")
    return tuple(n for n in PROPOSAL_SCALE_NS if n <= max_n)


def enn_n(
    opt,
    objective,
    bounds: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    chunk: int = PROPOSAL_SCALE_SEED_CHUNK,
) -> None:
    """Bulk-seed optimizer with N synthetic Ackley points via chunked ``tell``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.ndim == 2 and bounds.shape[1] == 2, (
        f"expected bounds shape (d, 2), got {bounds.shape}"
    )
    num_dim = int(bounds.shape[0])
    lo, hi = bounds[:, 0], bounds[:, 1]
    assert np.all(hi > lo), "bounds require hi > lo per dimension"
    for start in range(0, n, chunk):
        take = min(start + chunk, n) - start
        x = rng.uniform(lo, hi, size=(take, num_dim))
        y = np.asarray(objective(x), dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        assert y.shape == (take, 1), f"expected y shape ({take}, 1), got {y.shape}"
        opt.tell(x, y)


def _timedround(opt, objective) -> tuple[float, float]:
    """Run one ask → eval → tell round; return (ask_s, tell_s)."""
    t0 = time.perf_counter()
    x = opt.ask(num_arms=1)
    ask_s = time.perf_counter() - t0
    y = np.asarray(objective(x), dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    assert y.shape == (1, 1), f"expected y shape (1, 1), got {y.shape}"
    t_tell0 = time.perf_counter()
    opt.tell(x, y)
    tell_s = time.perf_counter() - t_tell0
    return ask_s, tell_s


def enn_proposal(
    opt,
    objective,
    *,
    warmup: int = PROPOSAL_SCALE_WARMUP,
    num_probes: int = PROPOSAL_SCALE_NUM_PROBES,
) -> tuple[float, float, float]:
    """Warmup then time ask/tell probes; return mean ask_s, tell_s, proposal_s."""
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    for _ in range(warmup):
        _timedround(opt, objective)
    ask_times: list[float] = []
    tell_times: list[float] = []
    for _ in range(num_probes):
        ask_s, tell_s = _timedround(opt, objective)
        ask_times.append(ask_s)
        tell_times.append(tell_s)
    ask_mean = float(np.mean(ask_times))
    tell_mean = float(np.mean(tell_times))
    return ask_mean, tell_mean, ask_mean + tell_mean


def format_header4(
    *,
    num_dim: int,
    max_n: int,
    num_probes: int,
    index_type: str,
    work_dir: str | None = None,
) -> str:
    header = (
        f"num_dim={num_dim} max_n={max_n} num_probes={num_probes} "
        f"index_type={index_type}"
    )
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    return header


def format_row2(
    n: int, ask_s: float, tell_s: float, proposal_s: float, *, n_width: int
) -> str:
    return (
        f"{n:>{n_width}} {ask_s:{DURATION_S_FMT}} "
        f"{tell_s:{DURATION_S_FMT}} {proposal_s:{DURATION_S_FMT}}"
    )


def _makepropopt(
    index_driver: NeighborIndex,
    bounds: np.ndarray,
    work_dir: str | None,
    n: int,
    seed: int,
    factory,
):
    from baux import create_optimizer

    n_work_dir: str | None = None
    if work_dir is not None:
        n_work_dir = str(Path(work_dir) / f"n{n}")
        Path(n_work_dir).mkdir(parents=True, exist_ok=True)
    if factory is not None:
        return factory(n=n, work_dir=n_work_dir)
    config = enn_config(
        index_driver=index_driver,
        work_dir=n_work_dir,
        num_init=TURBO_ENN_NUM_INIT,
    )
    return create_optimizer(
        bounds=bounds, config=config, rng=np.random.default_rng(seed + n)
    )


def run_stress3(
    index_driver: NeighborIndex,
    num_dim: int,
    max_n: int,
    *,
    work_dir: str | None = None,
    num_probes: int = PROPOSAL_SCALE_NUM_PROBES,
    warmup: int = PROPOSAL_SCALE_WARMUP,
    seed_chunk: int = PROPOSAL_SCALE_SEED_CHUNK,
    seed: int = TURBO_ENN_SEED,
    **options: Any,
) -> Iterator[ProposalScaleResult]:
    """Fresh opt per N: seed to N, warmup, probe; yield mean proposal timings."""
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if seed_chunk < 1:
        raise ValueError("seed_chunk must be >= 1")
    grid = proposal_ns(max_n)
    from baux.benchmarks import Ackley

    objective = options.get("objective")
    ackley = (
        Ackley(noise=TURBO_ENN_ACKLEY_NOISE, rng=np.random.default_rng(seed))
        if objective is None
        else objective
    )
    bounds = np.array([ackley.bounds] * num_dim, dtype=float)
    factory = options.get("optimizer_factory")
    for n in grid:
        opt = _makepropopt(index_driver, bounds, work_dir, n, seed, factory)
        enn_n(
            opt,
            ackley,
            bounds,
            n,
            rng=np.random.default_rng(seed + 10_000 + n),
            chunk=seed_chunk,
        )
        ask_mean, tell_mean, proposal_mean = enn_proposal(
            opt, ackley, warmup=warmup, num_probes=num_probes
        )
        yield ProposalScaleResult(
            n=n, ask_s=ask_mean, tell_s=tell_mean, proposal_s=proposal_mean
        )
