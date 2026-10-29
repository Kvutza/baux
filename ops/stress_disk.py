"""Disk RSS stress, synthetic data generators, and bpann store helpers."""

from __future__ import annotations

import json
import resource
import struct
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baux.enn.enn_class import ENN
from baux.enn.enn_params import ENNParams, PosteriorFlags
from baux.turbo.config.surrogate import NeighborIndex

INDEX_TYPE_CHOICES: tuple[str, ...] = ("flat", "bpann_disk")
DISK_INDEX_TYPE_CHOICES: frozenset[str] = frozenset({"bpann_disk"})
DISK_DEFER_SYNC_DRIVERS: frozenset[NeighborIndex] = frozenset(
    {NeighborIndex.BPANN_DISK}
)
DEFAULT_NUM_DIM = 10
STRESS_OBS_BATCH_SIZE = 100
ENN_ADD_STRESS_BATCH_SIZE = 1000
DEFAULT_HEARTBEAT_SECONDS = 10.0
STRESS_QUERY_N = 1000
STRESS_QUERY_SEED = 1
STRESS_QUERY_K = 10
NUM_SAMPLE = STRESS_QUERY_N
DEFAULT_SAMPLE_WORK_DIR = "_enn"
DEFAULT_SAMPLE_X_LOW = -1.0
DEFAULT_SAMPLE_X_HIGH = 1.0
DEFAULT_DRAW_SEED = 0
DEFAULT_DRAW_K = 10
DEFAULT_DRAW_NUM_FIT_CANDIDATES = 30
DEFAULT_DRAW_NUM_FIT_SAMPLES = 10
DEFAULT_DRAW_NUM_DRAWS = 100
DEFAULT_DRAW_NUM_SEEDS = 1
DRAW_OBS_NOISE_STD = 0.1
DRAW_F_CENTER = 0.3
DRAW_FLAGS = PosteriorFlags(observation_noise=True)
DRAW_FLAGS_NO_OBS = PosteriorFlags(observation_noise=False)
DURATION_S_FMT = ".4f"
STRESS_PARAMS = ENNParams(
    k_neighbors=STRESS_QUERY_K,
    epistemic_scale=1.0,
    aleatoric_scale=0.1,
)
DISK_STRESS_RSS_BASELINE_MIB = 512
DISK_STRESS_RSS_PER_SHARD_ROW_BYTES = 64
DEFAULT_SHARD_MAX_ROWS = 500_000
TURBO_ENN_NUM_INIT = 10
TURBO_ENN_ACKLEY_NOISE = 0.1
TURBO_ENN_SEED = 0
TURBO_ENN_K = 10
TURBO_ENN_NUM_FIT_SAMPLES = 100
TURBO_ENN_SEED_CHUNK = 100_000
TURBO_ENN_SEED_CHUNK_FLAT = 200_000
PROPOSAL_SCALE_NS: tuple[int, ...] = (
    10,
    30,
    100,
    300,
    1000,
    3000,
    10000,
    30000,
    100000,
)
PROPOSAL_SCALE_NUM_PROBES = 30
PROPOSAL_SCALE_WARMUP = 2
PROPOSAL_SCALE_SEED_CHUNK = 1000


def rss_bytes() -> int:
    """Peak resident set size in bytes (platform-normalized)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def rss_bytes2(*, num_dim: int, shard_max_rows: int = DEFAULT_SHARD_MAX_ROWS) -> int:
    """Acceptance #3 RSS delta ceiling from plan."""
    return (
        DISK_STRESS_RSS_BASELINE_MIB * 1024 * 1024
        + DISK_STRESS_RSS_PER_SHARD_ROW_BYTES * num_dim * shard_max_rows
    )


@dataclass(frozen=True)
class DiskRssStressConfig:
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = 0
    batch_size: int = STRESS_OBS_BATCH_SIZE
    query_n: int = STRESS_QUERY_N


@dataclass(frozen=True)
class DiskRssStressResult:
    num_obs: int
    baseline_rss_bytes: int
    final_rss_bytes: int
    rss_delta_bytes: int
    train_x_bytes: int
    index_bytes: int


def _addbatches(
    model: ENN, num_obs: int, cfg: DiskRssStressConfig, x_query: np.ndarray
) -> None:
    rows_added = 0
    for x_batch, y_batch in iter_batches(
        num_obs,
        num_dim=cfg.num_dim,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
    ):
        model.add(x_batch, y_batch)
        rows_added += x_batch.shape[0]
        model.schedule_flush()
    assert rows_added == num_obs
    model.ensure_sync()
    posterior = model.posterior(x_query, params=STRESS_PARAMS)
    assert np.all(np.isfinite(posterior.mu))
    assert np.all(np.isfinite(posterior.se))


def rss_stress(
    *,
    num_obs: int,
    work_dir: str,
    index_driver: NeighborIndex = NeighborIndex.BPANN_DISK,
    config: DiskRssStressConfig | None = None,
) -> DiskRssStressResult:
    """Stream batched adds to disk ENN; return RSS and on-disk metrics."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    cfg = config if config is not None else DiskRssStressConfig()
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg.query_n < 1:
        raise ValueError("query_n must be >= 1")
    baseline_rss = rss_bytes()
    x_query = make_points(cfg.query_n, num_dim=cfg.num_dim, seed=STRESS_QUERY_SEED)
    model = ENN(
        np.empty((0, cfg.num_dim), dtype=float),
        np.empty((0, 1), dtype=float),
        scale_x=False,
        index_driver=index_driver,
        work_dir=work_dir,
        enn_storage="disk",
    )
    baseline = min(baseline_rss, rss_bytes())
    _addbatches(model, num_obs, cfg, x_query)
    final_rss = rss_bytes()
    train_x_path = Path(work_dir) / "train_x.bin"
    train_x_bytes = train_x_path.stat().st_size if train_x_path.exists() else 0
    assert abs(train_x_bytes - num_obs * cfg.num_dim * 8) <= cfg.num_dim * 8
    assert len(model) == num_obs
    return DiskRssStressResult(
        num_obs=num_obs,
        baseline_rss_bytes=baseline,
        final_rss_bytes=final_rss,
        rss_delta_bytes=final_rss - baseline,
        train_x_bytes=train_x_bytes,
        index_bytes=model.index_bytes(),
    )


@dataclass(frozen=True)
class EnnAddStressConfig:
    num_dim: int = DEFAULT_NUM_DIM
    seed: int = 0
    batch_size: int = 1
    progress_every: int = 0
    heartbeat_seconds: float = 0.0
    query_n: int = STRESS_QUERY_N
    query_seed: int = STRESS_QUERY_SEED
    work_dir: str | None = None


def parse_driver2(name: str) -> NeighborIndex:
    mapping = {
        "flat": NeighborIndex.FLAT,
        "bpann_disk": NeighborIndex.BPANN_DISK,
    }
    if name not in mapping:
        raise ValueError(f"Unknown index type: {name}")
    return mapping[name]


def enn_chunk(index_driver: NeighborIndex) -> int:
    """Return turbo-enn bulk-seed chunk sized for the index backend."""
    if index_driver == NeighborIndex.FLAT:
        return TURBO_ENN_SEED_CHUNK_FLAT
    return TURBO_ENN_SEED_CHUNK


def _nextcheckpoint(n: int) -> int:
    if n < 3:
        return 3
    if n < 10:
        return 10
    if n % 30 == 10:
        return n * 3
    return n * 10 // 3


def checkpoint_ns(max_n: int) -> tuple[int, ...]:
    """Return checkpoint sizes N=1, 3, 10, 30, 100, ... up to max_n."""
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    out: list[int] = []
    n = 1
    while n <= max_n:
        out.append(n)
        n = _nextcheckpoint(n)
    return tuple(out)


def make_observations(
    num_obs: int, *, num_dim: int = 10, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_obs, num_dim)), rng.standard_normal((num_obs, 1))


def make_points(query_n: int, *, num_dim: int, seed: int) -> np.ndarray:
    """Return (query_n, num_dim) query batch held constant across checkpoints."""
    if query_n < 1:
        raise ValueError("query_n must be >= 1")
    return np.random.default_rng(seed).standard_normal((query_n, num_dim))


def make_points2(
    query_n: int,
    *,
    num_dim: int,
    seed: int,
    low: float = DEFAULT_SAMPLE_X_LOW,
    high: float = DEFAULT_SAMPLE_X_HIGH,
) -> np.ndarray:
    """Return (query_n, num_dim) uniformly random query batch in [low, high]."""
    if query_n < 1:
        raise ValueError("query_n must be >= 1")
    if num_dim < 1:
        raise ValueError("num_dim must be >= 1")
    if low >= high:
        raise ValueError("low must be < high")
    return np.random.default_rng(seed).uniform(low, high, size=(query_n, num_dim))


def iter_observations(
    num_obs: int,
    *,
    num_dim: int = DEFAULT_NUM_DIM,
    seed: int = 0,
    batch_size: int = STRESS_OBS_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (1, num_dim) x and (1, 1) y rows without holding all num_obs rows."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < num_obs:
        n = min(batch_size, num_obs - emitted)
        for _ in range(n):
            yield rng.standard_normal((1, num_dim)), rng.standard_normal((1, 1))
        emitted += n


def iter_batches(
    num_obs: int,
    *,
    num_dim: int = DEFAULT_NUM_DIM,
    seed: int = 0,
    batch_size: int = STRESS_OBS_BATCH_SIZE,
    stop_points: Sequence[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (n, num_dim) x and (n, 1) y batches without holding all num_obs rows."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    stops = sorted(s for s in (stop_points or ()) if 1 <= s <= num_obs)
    stop_i = 0
    rng = np.random.default_rng(seed)
    emitted = 0
    while emitted < num_obs:
        n = min(batch_size, num_obs - emitted)
        while stop_i < len(stops) and stops[stop_i] <= emitted:
            stop_i += 1
        if stop_i < len(stops):
            n = min(n, stops[stop_i] - emitted)
        yield rng.standard_normal((n, num_dim)), rng.standard_normal((n, 1))
        emitted += n


def load_metadata(work_dir: str) -> dict:
    """Return metadata.json for a disk-backed bpann store."""
    meta_path = Path(work_dir) / "metadata.json"
    if not meta_path.is_file():
        raise ValueError(f"metadata.json not found in {work_dir}")
    meta = json.loads(meta_path.read_text())
    if meta.get("index_backend") != "bpann_disk":
        raise ValueError(
            f"work_dir index_backend is not bpann_disk: {meta.get('index_backend')!r}"
        )
    for key in ("num_dim", "num_metrics", "num_obs"):
        if key not in meta:
            raise ValueError(f"metadata.json missing {key!r}")
    return meta


def function_sample(seed: int, num_seeds: int = 1) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    return list(range(seed, seed + num_seeds))


def bpann_enn(work_dir: str) -> tuple[ENN, dict]:
    """Reopen a persisted bpann_disk store from work_dir."""
    meta = load_metadata(work_dir)
    num_dim = int(meta["num_dim"])
    num_metrics = int(meta["num_metrics"])
    model = ENN(
        np.empty((0, num_dim), dtype=float),
        np.empty((0, num_metrics), dtype=float),
        scale_x=bool(meta.get("scale_x", False)),
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=work_dir,
        enn_storage="disk",
    )
    model.ensure_sync()
    return model, meta


def load_existing(work_dir: str) -> int:
    """Return persisted observation count in work_dir, or 0 if none."""
    root = Path(work_dir)
    sidecar = root / "num_obs.bin"
    if sidecar.is_file() and sidecar.stat().st_size == 8:
        return int(struct.unpack("<Q", sidecar.read_bytes())[0])
    meta_path = root / "metadata.json"
    if meta_path.is_file():
        try:
            num_obs = json.loads(meta_path.read_text())["num_obs"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return 0
        if isinstance(num_obs, int) and num_obs >= 0:
            return num_obs
    return 0


def format_header(
    *,
    num_dim: int,
    num_obs: int,
    work_dir: str | None = None,
    num_obs_existing: int | None = None,
) -> str:
    prefix = "restarting " if num_obs_existing else ""
    header = f"{prefix}num_dim={num_dim} num_obs={num_obs}"
    if num_obs_existing:
        header = f"{header} num_obs_existing={num_obs_existing}"
    if work_dir is not None:
        header = f"{header} work_dir={work_dir}"
    return header


def _times(model: ENN, x_query: np.ndarray) -> float:
    t0 = time.perf_counter()
    model.posterior(x_query, params=STRESS_PARAMS)
    return time.perf_counter() - t0


def stress_width(num_obs: int) -> int:
    """Character width for the N column; sized for the largest checkpoint (num_obs)."""
    if num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    return len(str(num_obs))


def format_row(n: int, query_s: float, segment_s: float, *, n_width: int) -> str:
    return f"{n:>{n_width}} {query_s:{DURATION_S_FMT}} {segment_s:{DURATION_S_FMT}}"
