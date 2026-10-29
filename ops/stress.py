#!/usr/bin/env python
"""Operational stress tools and benchmarks for ENN and TuRBO-ENN."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import numpy as np

from baux.enn.enn_class import ENN
from baux.turbo.config.surrogate import NeighborIndex
from ops import (
    stress_disk as _disk,
    stress_draw as _draw,
    stress_sample as _sample,
    stress_turbo as _turbo,
)
from ops.stress_disk import (
    DISK_DEFER_SYNC_DRIVERS,
    DISK_INDEX_TYPE_CHOICES,
    ENN_ADD_STRESS_BATCH_SIZE,
    TURBO_ENN_NUM_INIT,
    EnnAddStressConfig,
    _times,
    checkpoint_ns,
    format_header,
    format_row,
    iter_batches,
    iter_observations,
    load_existing,
    make_points,
    parse_driver2,
    stress_width,
)

globals().update(
    {
        k: v
        for mod in (_disk, _sample, _draw, _turbo)
        for k, v in vars(mod).items()
        if not k.startswith("__")
    }
)


def _chunkiter(num_obs: int, cfg: EnnAddStressConfig, checkpoint_list: list[int]):
    if cfg.batch_size <= 1:
        return iter_observations(num_obs, num_dim=cfg.num_dim, seed=cfg.seed)
    return iter_batches(
        num_obs,
        num_dim=cfg.num_dim,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        stop_points=checkpoint_list,
    )


def _logprogress(n: int, cfg: EnnAddStressConfig, last_hb: float) -> float:
    if cfg.progress_every and (n % cfg.progress_every == 0):
        click.echo(f"progress n={n}", err=True)
    if cfg.heartbeat_seconds and (
        time.perf_counter() - last_hb >= cfg.heartbeat_seconds
    ):
        click.echo(f"heartbeat n={n}", err=True)
        return time.perf_counter()
    return last_hb


def _emitcp(
    model: ENN,
    index_driver: NeighborIndex,
    x_query: np.ndarray,
    checkpoint_list: list[int],
    current_n: int,
    state: list,
) -> Iterator[tuple[int, float, float]]:
    while state[0] < len(checkpoint_list) and checkpoint_list[state[0]] <= current_n:
        cp = checkpoint_list[state[0]]
        if index_driver not in DISK_DEFER_SYNC_DRIVERS:
            t_sync = time.perf_counter()
            model.ensure_sync()
            state[1] += time.perf_counter() - t_sync
        seg_s, state[1] = state[1], 0.0
        yield (cp, _times(model, x_query), seg_s)
        state[0] += 1


def _streamadds(
    model: ENN,
    index_driver: NeighborIndex,
    num_obs: int,
    cfg: EnnAddStressConfig,
    checkpoint_list: list[int],
    x_query: np.ndarray,
) -> Iterator[tuple[int, float, float]]:
    n, last_hb, state = 0, time.perf_counter(), [0, 0.0]
    for x_chunk, y_chunk in _chunkiter(num_obs, cfg, checkpoint_list):
        t_add = time.perf_counter()
        model.add(x_chunk, y_chunk)
        if index_driver in DISK_DEFER_SYNC_DRIVERS:
            model.schedule_flush()
        state[1] += time.perf_counter() - t_add
        n += int(x_chunk.shape[0])
        last_hb = _logprogress(n, cfg, last_hb)
        yield from _emitcp(model, index_driver, x_query, checkpoint_list, n, state)


def enn_stress(
    *,
    index_driver: NeighborIndex,
    num_obs: int,
    config: EnnAddStressConfig | None = None,
) -> Iterator[tuple[int, float, float]]:
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    cfg = config if config is not None else EnnAddStressConfig()
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    checkpoint_list = sorted(set(checkpoint_ns(num_obs)))
    x_query = make_points(cfg.query_n, num_dim=cfg.num_dim, seed=cfg.query_seed)
    empty_x = np.empty((0, cfg.num_dim), dtype=float)
    empty_y = np.empty((0, 1), dtype=float)
    model_kwargs: dict[str, Any] = {
        "train_x": empty_x,
        "train_y": empty_y,
        "scale_x": False,
        "index_driver": index_driver,
    }
    if cfg.work_dir is not None:
        model_kwargs["work_dir"] = cfg.work_dir
        model_kwargs["enn_storage"] = "disk"
    model = ENN(**model_kwargs)
    model.add(empty_x, empty_y)
    try:
        yield from _streamadds(
            model, index_driver, num_obs, cfg, checkpoint_list, x_query
        )
    finally:
        if index_driver in DISK_DEFER_SYNC_DRIVERS:
            model.persist_index()


def _checkworkdir(index_type: str, work_dir: str | None) -> NeighborIndex:
    if work_dir is not None and index_type not in DISK_INDEX_TYPE_CHOICES:
        raise click.ClickException(
            f"work_dir requires index_type in {sorted(DISK_INDEX_TYPE_CHOICES)}"
        )
    if index_type in DISK_INDEX_TYPE_CHOICES and work_dir is None:
        raise click.ClickException(f"{index_type} requires --work-dir")
    return parse_driver2(index_type)


@click.group()
def cli() -> None:
    """Operational stress tools."""


_WORKDIR = click.Option(
    ["--work-dir"],
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    default=None,
    help="Disk-backed ENN work directory (requires bpann_disk).",
)
_NUMDIM = click.Option(
    ["--num-dim"],
    type=int,
    default=_disk.DEFAULT_NUM_DIM,
    show_default=True,
    help="Embedding dimension for synthetic observations.",
)


@cli.command(
    "enn",
    params=[
        click.Argument(["index_type"], type=click.Choice(_disk.INDEX_TYPE_CHOICES)),
        click.Argument(["num_obs"], type=int),
        _NUMDIM,
        click.Option(
            ["--progress-every"],
            type=int,
            default=0,
            show_default=True,
            help="Emit `progress n=<N>` to stderr every N additions (0 disables).",
        ),
        click.Option(
            ["--heartbeat-seconds"],
            type=float,
            default=_disk.DEFAULT_HEARTBEAT_SECONDS,
            show_default=True,
            help="Emit `heartbeat n=<N>` to stderr at most this often (0 disables).",
        ),
        _WORKDIR,
        click.Option(
            ["--batch"],
            is_flag=True,
            default=False,
            help=f"Add observations in batches of {ENN_ADD_STRESS_BATCH_SIZE} instead of one row per add.",
        ),
        click.Option(
            ["--baux-config"],
            type=click.Path(file_okay=True, dir_okay=False, path_type=str),
            default=None,
            help="Path to baux config.toml (overrides ~/.baux/config.toml).",
        ),
    ],
)
def enn(index_type: str, num_obs: int, **options: Any) -> None:
    """Time 1000-point ENN queries at sparse checkpoints while streaming adds."""
    if options["baux_config"] is not None:
        from baux._rust import set_path

        set_path(options["baux_config"])
    num_dim = int(options["num_dim"])
    if num_obs < 1 or num_dim < 1:
        raise click.ClickException(
            "num_obs must be >= 1" if num_obs < 1 else "num_dim must be >= 1"
        )
    if options["progress_every"] < 0 or options["heartbeat_seconds"] < 0:
        msg = (
            "progress_every must be >= 0"
            if options["progress_every"] < 0
            else "heartbeat_seconds must be >= 0"
        )
        raise click.ClickException(msg)
    work_dir = options["work_dir"]
    driver = _checkworkdir(index_type, work_dir)
    existing = load_existing(work_dir) if work_dir is not None else 0
    click.echo(
        format_header(
            num_dim=num_dim,
            num_obs=num_obs,
            work_dir=work_dir,
            num_obs_existing=existing if existing else None,
        )
    )
    n_width = stress_width(num_obs)
    cfg = EnnAddStressConfig(
        num_dim=num_dim,
        batch_size=ENN_ADD_STRESS_BATCH_SIZE if options["batch"] else 1,
        progress_every=options["progress_every"],
        heartbeat_seconds=options["heartbeat_seconds"],
        work_dir=work_dir,
    )
    for n, query_s, segment_s in enn_stress(
        index_driver=driver, num_obs=num_obs, config=cfg
    ):
        click.echo(format_row(n, query_s, segment_s, n_width=n_width))


@cli.command(
    "sample",
    params=[
        click.Argument(
            ["work_dir"],
            type=click.Path(file_okay=False, dir_okay=True, path_type=str),
        ),
        click.Argument(["num_samples"], type=int),
        click.Option(
            ["--seed"],
            type=int,
            default=_disk.STRESS_QUERY_SEED,
            show_default=True,
            help="RNG seed for query points and function draw.",
        ),
    ],
)
def sample(work_dir: str, num_samples: int, seed: int) -> None:
    """Draw posterior function samples at uniform random x on a persisted bpann store."""
    if num_samples < 1:
        raise click.ClickException("num_samples must be >= 1")
    if not Path(work_dir).is_dir():
        raise click.ClickException(f"work_dir does not exist: {work_dir}")
    try:
        result = _sample.run_stress(
            work_dir=work_dir,
            config=_sample.SampleStressConfig(num_samples=num_samples, seed=seed),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not result.all_finite:
        raise click.ClickException("posterior_draw returned non-finite values")
    click.echo(_sample.format_header2(result=result, work_dir=work_dir))
    click.echo(_sample.format_summary(result))


@cli.command(
    "draw",
    params=[
        click.Argument(["num_obs"], type=int),
        click.Argument(["num_test"], type=int),
        _NUMDIM,
        click.Option(
            ["--seed"],
            type=int,
            default=_disk.DEFAULT_DRAW_SEED,
            show_default=True,
            help="RNG seed for train/test data (fit uses seed+1).",
        ),
        click.Option(
            ["--k"],
            type=int,
            default=_disk.DEFAULT_DRAW_K,
            show_default=True,
            help="Number of neighbors for fit and posterior.",
        ),
        click.Option(
            ["--num-fit-candidates", "num_candidates"],
            type=int,
            default=_disk.DEFAULT_DRAW_NUM_FIT_CANDIDATES,
            show_default=True,
            help="Hyperparameter candidates per fit ask.",
        ),
        click.Option(
            ["--num-fit-samples", "num_samples"],
            type=int,
            default=_disk.DEFAULT_DRAW_NUM_FIT_SAMPLES,
            show_default=True,
            help="Subsample size for fit log-likelihood.",
        ),
        click.Option(
            ["--num-draws"],
            type=int,
            default=_disk.DEFAULT_DRAW_NUM_DRAWS,
            show_default=True,
            help="Number of posterior draws for likelihood and argmin metrics.",
        ),
        click.Option(
            ["--num-seeds"],
            type=int,
            default=_disk.DEFAULT_DRAW_NUM_SEEDS,
            show_default=True,
            help="Repeat full draw stress over seed..seed+num_seeds-1; report mean ± SE.",
        ),
    ],
)
def draw(num_obs: int, num_test: int, **options: Any) -> None:
    """Fit ENN on synthetic data; report avg likelihood for two draw methods."""
    checks = (
        ("num_obs", num_obs, 1),
        ("num_test", num_test, 1),
        ("num_dim", options["num_dim"], 1),
        ("k", options["k"], 1),
        ("num_candidates", options["num_candidates"], 1),
        ("num_samples", options["num_samples"], 1),
        ("num_draws", options["num_draws"], 2),
        ("num_seeds", options["num_seeds"], 1),
    )
    for name, val, min_val in checks:
        if val < min_val:
            raise click.ClickException(f"{name} must be >= {min_val}")
    config = _sample.DrawStressConfig(
        num_obs=num_obs,
        num_test=num_test,
        num_dim=options["num_dim"],
        seed=options["seed"],
        k=options["k"],
        num_candidates=options["num_candidates"],
        num_samples=options["num_samples"],
        num_draws=options["num_draws"],
    )
    try:
        agg = _draw.run_seeds(config, num_seeds=options["num_seeds"])
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(_draw.format_aggregate(agg))
    click.echo(_draw.format_aggregate2(agg.posterior))
    click.echo(_draw.format_aggregate2(agg.posterior_draw))


@cli.command(
    "turbo-enn",
    params=[
        click.Argument(["index_type"], type=click.Choice(_disk.INDEX_TYPE_CHOICES)),
        click.Argument(["num_obs"], type=int),
        click.Argument(["num_ask"], type=int),
        _NUMDIM,
        _WORKDIR,
        click.Option(
            ["--tell-all"],
            is_flag=True,
            default=False,
            help="Deliver each DGP observation with its own tell() (intended for small-N fidelity; large N is much slower).",
        ),
    ],
)
def turbo_enn(
    index_type: str,
    num_obs: int,
    num_ask: int,
    num_dim: int,
    work_dir: str | None,
    tell_all: bool,
) -> None:
    """Time TuRBO-ENN ask at exponentially spaced DGP observation checkpoints."""
    for name, val in (("num_dim", num_dim), ("num_obs", num_obs), ("num_ask", num_ask)):
        if val < 1:
            raise click.ClickException(f"{name} must be >= 1")
    if num_ask > num_obs:
        raise click.ClickException("num_ask must be <= num_obs")
    driver = _checkworkdir(index_type, work_dir)
    click.echo(
        _turbo.enn_header(
            num_dim=num_dim,
            num_obs=num_obs,
            num_ask=num_ask,
            index_type=index_type,
            work_dir=work_dir,
            tell_all=tell_all,
        )
    )
    n_width = len(str(num_obs))
    runner = globals()["enn_stress2"]
    try:
        for row in runner(
            index_driver=driver,
            num_dim=num_dim,
            num_obs=num_obs,
            num_ask=num_ask,
            work_dir=work_dir,
            tell_all=tell_all,
        ):
            click.echo(
                _turbo.enn_row(
                    row.n, row.iter_s, row.ask_s, row.tell_s, n_width=n_width
                )
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command(
    "proposal-scale",
    params=[
        click.Argument(["index_type"], type=click.Choice(_disk.INDEX_TYPE_CHOICES)),
        _NUMDIM,
        click.Option(
            ["--max-n"],
            type=int,
            default=_disk.PROPOSAL_SCALE_NS[-1],
            show_default=True,
            help="Largest N on the proposal-scale grid (filters fixed checkpoints).",
        ),
        click.Option(
            ["--num-probes"],
            type=int,
            default=_disk.PROPOSAL_SCALE_NUM_PROBES,
            show_default=True,
            help="Timed ask/tell rounds per N after warmup.",
        ),
        _WORKDIR,
    ],
)
def proposal_scale(
    index_type: str,
    num_dim: int,
    max_n: int,
    num_probes: int,
    work_dir: str | None,
) -> None:
    """Time mean TuRBO-ENN proposal cost at log-spaced N (seed then probe)."""
    if num_dim < 1 or num_probes < 1:
        raise click.ClickException(
            "num_dim must be >= 1" if num_dim < 1 else "num_probes must be >= 1"
        )
    if max_n < TURBO_ENN_NUM_INIT:
        raise click.ClickException(
            f"max_n must be >= {TURBO_ENN_NUM_INIT} (TuRBO-ENN num_init)"
        )
    driver = _checkworkdir(index_type, work_dir)
    click.echo(
        _turbo.format_header4(
            num_dim=num_dim,
            max_n=max_n,
            num_probes=num_probes,
            index_type=index_type,
            work_dir=work_dir,
        )
    )
    n_width = len(str(max_n))
    runner = globals()["run_stress3"]
    try:
        for row in runner(
            driver, num_dim, max_n, work_dir=work_dir, num_probes=num_probes
        ):
            click.echo(
                _turbo.format_row2(
                    row.n, row.ask_s, row.tell_s, row.proposal_s, n_width=n_width
                )
            )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
