from __future__ import annotations

import gc
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import baux
import numpy as np
import pytest
from baux._rust import ENN as _RustENN
from baux.enn.enn_params import ENNParams, PosteriorFlags
from baux.enn.enn_state import ENNState

if not hasattr(_RustENN, "x_obs"):

    def _xobs(self):
        n = len(self)
        if n == 0:
            return np.empty((0, self.num_dim), dtype=float)
        return self.train_rows(list(range(n)))[0]

    _RustENN.x_obs = _xobs

if not hasattr(_RustENN, "y_obs"):

    def _yobs(self):
        n = len(self)
        if n == 0:
            return np.empty((0, self.num_outputs), dtype=float)
        return self.train_rows(list(range(n)))[1]

    _RustENN.y_obs = _yobs

if not hasattr(_RustENN, "yvar_obs"):

    def _yvarobs(self):
        try:
            n = len(self)
            rows = list(range(n))
            _, _, yvar = self.train_rows(rows)
            return yvar
        except Exception:
            return None

    _RustENN.yvar_obs = _yvarobs

pytestmark = pytest.mark.slow

_POSTERIOR_SPEED_BASELINE = (
    Path(__file__).resolve().parent / "fixtures" / "postspdbase.toml"
)


def enn_data(num_samples: int, k: int, noise: float, m: int = 1):
    x = np.sort(np.random.rand(num_samples + 4))
    x[-3] = x[-4]
    x[-2] = x[-4]
    x[-1] = x[-4]
    x[1] = x[0] + 0.03
    eps = np.random.randn(num_samples + 4)
    y = np.sin(2 * m * np.pi * x) + noise * eps
    yvar = (noise**2) * np.ones_like(y)
    state = baux.enn_init(x[:, None], y[:, None], yvar[:, None])
    params = baux.fit(
        state,
        k=k,
        num_candidates=100,
        num_samples=min(10, num_samples),
        seed=0,
    )
    return x, y, state, params


def enn_logic(state: ENNState, params: ENNParams):
    x_hat = np.linspace(0.0, 1.0, 30)
    x_hat_2d = x_hat[:, None]
    posterior = baux.posterior(
        state, x_hat_2d, params=params, flags=PosteriorFlags(exclude_nearest=False)
    )
    return posterior.mu[:, 0], posterior.se[:, 0]


@pytest.mark.skip(reason="Timing unreliable across machines; run manually to verify")
def test_enndemoperformance():
    np.random.seed(1)
    t0 = time.time()
    _x, _y, state, params = enn_data(num_samples=1_000_000, k=5, noise=0.3, m=3)
    _mu, _se = enn_logic(state, params)
    elapsed = time.time() - t0
    print(f"\nTime taken: {elapsed:.3f} seconds")
    assert elapsed < 0.3, f"Expected < 0.3s, got {elapsed:.3f}s"


_PERF_WARMUP = 1
_PERF_REPS = 1


def _medianseconds(fn, *, reps: int = _PERF_REPS) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _medianratio(
    fn_num: Callable[[], None],
    fn_den: Callable[[], None],
    *,
    warmup: int = _PERF_WARMUP,
    reps: int = _PERF_REPS,
) -> float:
    """Median of per-rep num/den ratios with interleaved warmup and alternating order."""
    for i in range(warmup):
        if i % 2 == 0:
            fn_den()
            fn_num()
        else:
            fn_num()
            fn_den()
    ratios: list[float] = []
    for i in range(reps):
        if i % 2 == 0:
            t0 = time.perf_counter()
            fn_num()
            t_num = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn_den()
            t_den = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            fn_den()
            t_den = time.perf_counter() - t0
            t0 = time.perf_counter()
            fn_num()
            t_num = time.perf_counter() - t0
        ratios.append(t_num / max(t_den, 1e-12))
    return float(np.median(ratios))


def _neighborratio(n: int, *, d: int = 1, k: int = 10, seed: int = 42) -> float:
    """Wall-clock ratio: index_search (f64 exact) vs FAISS-only neighbor lookup."""
    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = rng.normal(0.0, 100.0, size=(n, d))
    state = baux.enn_init(train_x, train_y, train_yvar=None)

    def index_neighbors() -> None:
        state.posterior_neighbors(
            train_x,
            search_k=k,
        )

    def faiss_neighbors() -> None:
        state.nearest_neighbors(
            train_x,
            search_k=k,
            exclude_nearest=False,
        )

    for _ in range(3):
        index_neighbors()
        faiss_neighbors()

    t_index = _medianseconds(index_neighbors)
    t_faiss = _medianseconds(faiss_neighbors)
    return t_index / max(t_faiss, 1e-12)


@pytest.mark.parametrize(
    ("n", "max_ratio"),
    [
        (20, 2.5),  # user-reported scenario (n=20, k=10)
        (500, 3.0),  # n_query=n_train; batched f64 matrix should track FAISS
    ],
)
def test_001(n: int, max_ratio: float):
    """index_search (f64 exact) must not regress far beyond FAISS-only lookup."""
    ratio = _neighborratio(n)
    assert ratio <= max_ratio, (
        f"index_search must be <= {max_ratio}x FAISS-only at n={n}, "
        f"got {ratio:.2f}x (slowdown bad)"
    )


def test_002():
    """posterior(train_x) tie-break-on must track tie-break-off at n=1024."""
    rng = np.random.default_rng(0)
    n, d, m = 1024, 10, 5
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    state = baux.enn_init(x, y)
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.1)

    def run_off() -> None:
        baux.posterior(
            state,
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=False),
        )

    def run_on() -> None:
        baux.posterior(
            state,
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=True),
        )

    ratio = _medianratio(run_on, run_off, warmup=3, reps=5)
    assert ratio <= 1.22, (
        f"tie-break posterior must be <= 1.22x no-tie-break at n=1024, got {ratio:.2f}x"
    )


def test_003():
    """Lattice posterior(x_train) tie-break-on must track tie-break-off at n=1024."""
    n, d = 1024, 10
    x = np.linspace(0.0, 1.0, n)[:, None]
    x = np.tile(x, (1, d))
    y = np.sin(8.0 * np.pi * x[:, 0:1])
    state = baux.enn_init(x, y)
    params = ENNParams(k_neighbors=10, epistemic_scale=1.0, aleatoric_scale=0.1)

    def run_off() -> None:
        baux.posterior(
            state,
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=False),
        )

    def run_on() -> None:
        baux.posterior(
            state,
            x,
            params=params,
            flags=PosteriorFlags(tie_neighbors=True),
        )

    ratio = _medianratio(run_on, run_off)
    assert ratio <= 1.15, (
        f"lattice tie-break posterior must be <= 1.15x no-tie-break at n=1024, "
        f"got {ratio:.2f}x"
    )


def test_004():
    """Ratio should stay bounded as n_query=n_train grows."""
    n_small, n_large = 256, 1024
    ratio_small = _neighborratio(n_small, seed=7)
    ratio_large = _neighborratio(n_large, seed=7)
    assert ratio_large <= max(ratio_small * 1.5, 3.0), (
        f"expected bounded slowdown ratio: n={n_small} -> {ratio_small:.2f}x, "
        f"n={n_large} -> {ratio_large:.2f}x"
    )


class SelfSearchTiming(NamedTuple):
    posterior_seconds: float
    index_seconds: float
    faiss_seconds: float
    index_vs_faiss: float
    posterior_vs_index: float


def _selfratios(
    scenario: dict,
    *,
    tie_neighbors: bool,
) -> SelfSearchTiming:
    """Return (t_post, t_index, t_faiss, index/faiss, posterior/index) medians."""
    rng = np.random.default_rng(int(scenario["seed"]))
    n, d, m, k = scenario["n"], scenario["d"], scenario["m"], scenario["k"]
    x = rng.standard_normal((n, d))
    y = rng.normal(0.0, 100.0, size=(n, m))
    state = baux.enn_init(x, y)
    params = ENNParams(k_neighbors=k, epistemic_scale=1.0, aleatoric_scale=0.1)
    flags = PosteriorFlags(tie_neighbors=tie_neighbors)
    warmup = 3
    reps = 7

    def run_posterior() -> None:
        baux.posterior(state, x, params=params, flags=flags)

    def index_neighbors() -> None:
        state.posterior_neighbors(
            x,
            search_k=k,
            flags=flags,
        )

    def faiss_neighbors() -> None:
        state.nearest_neighbors(x, search_k=k, exclude_nearest=False)

    for _ in range(warmup):
        run_posterior()
        index_neighbors()
        faiss_neighbors()

    return SelfSearchTiming(
        _medianseconds(run_posterior, reps=reps),
        _medianseconds(index_neighbors, reps=reps),
        _medianseconds(faiss_neighbors, reps=reps),
        _medianratio(index_neighbors, faiss_neighbors, warmup=warmup, reps=reps),
        _medianratio(run_posterior, index_neighbors, warmup=warmup, reps=reps),
    )


@pytest.mark.parametrize("tie_neighbors", [False, True])
def test_005(
    tie_neighbors: bool,
):
    """Self-search: index_search vs FAISS and posterior stats overhead stay bounded."""
    gc.collect()
    baseline = tomllib.loads(_POSTERIOR_SPEED_BASELINE.read_text(encoding="utf-8"))
    for scenario in baseline["scenarios"]:
        t_post, t_index, t_faiss, index_vs_faiss, posterior_vs_index = _selfratios(
            scenario, tie_neighbors=tie_neighbors
        )
        if tie_neighbors and "max_index_vs_faiss_tie_break" in scenario:
            max_index_vs_faiss = float(scenario["max_index_vs_faiss_tie_break"])
        else:
            max_index_vs_faiss = float(scenario["max_index_vs_faiss"])
        if tie_neighbors and "max_posterior_vs_index_tie_break" in scenario:
            max_posterior_vs_index = float(scenario["max_posterior_vs_index_tie_break"])
        else:
            max_posterior_vs_index = float(scenario["max_posterior_vs_index"])
        assert index_vs_faiss <= max_index_vs_faiss, (
            f"{scenario['name']} tie_break={tie_neighbors}: "
            f"index/faiss={index_vs_faiss:.3f}x exceeds cap {max_index_vs_faiss:.2f}x "
            f"(t_index={t_index:.4f}s t_faiss={t_faiss:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )
        assert posterior_vs_index <= max_posterior_vs_index, (
            f"{scenario['name']} tie_break={tie_neighbors}: "
            f"posterior/index={posterior_vs_index:.3f}x exceeds cap "
            f"{max_posterior_vs_index:.2f}x "
            f"(t_post={t_post:.4f}s t_index={t_index:.4f}s; "
            f"baseline_rev={baseline['baseline_revision'][:7]})"
        )


def test_006():
    """Benchmark baux.enn_init and baux.append throughput and latency."""
    rng = np.random.default_rng(42)
    n, d = 500, 8
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))

    # Benchmark baux.enn_init
    def run_init() -> None:
        baux.enn_init(x, y)

    t_init = _medianseconds(run_init, reps=5)
    assert t_init < 0.1, f"baux.enn_init too slow at n={n}: {t_init:.4f}s"

    # Benchmark baux.append
    base_state = baux.enn_init(x[:100], y[:100])
    chunk_x = x[100:150]
    chunk_y = y[100:150]

    def run_append() -> None:
        baux.append(base_state, chunk_x, chunk_y)

    t_append = _medianseconds(run_append, reps=5)
    assert t_append < 0.05, f"baux.append too slow: {t_append:.4f}s"


def test_007():
    """Benchmark baux.fit latency and warm-start speed."""
    rng = np.random.default_rng(42)
    n, d = 200, 4
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((n, 1))
    state = baux.enn_init(x, y)

    fit_key = baux.key(123)
    p_init = baux.fit(state, k=5, num_candidates=20, num_samples=10, key=fit_key)

    def run_fitcold() -> None:
        baux.fit(state, k=5, num_candidates=20, num_samples=10, key=fit_key)

    def run_fitwarm() -> None:
        baux.fit(
            state, k=5, num_candidates=20, num_samples=10, warm=p_init, key=fit_key
        )

    t_cold = _medianseconds(run_fitcold, reps=3)
    t_warm = _medianseconds(run_fitwarm, reps=3)
    assert t_cold < 1.0, f"baux.fit cold too slow: {t_cold:.4f}s"
    assert t_warm < 1.0, f"baux.fit warm too slow: {t_warm:.4f}s"
