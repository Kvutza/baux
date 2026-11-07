from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np
from numpy.random import Generator

from .turbo.config import OptimizerConfig
from .turbo.config.encode import encode, enn_k, is_lhd, supports
from .turbo.config.surrogate import (
    ENNSurrogate,
    ExternalSurrogate,
    GaussianProcess,
    NoSurrogate,
)
from .turbo.config.trust_region import RegionalSearchPolicy
from .turbo.optimizer_types import Telemetry, _Obs, _Plan, _View


@dataclass(frozen=True, order=True)
class AskId:
    """Opaque, process-wide identity for one regional evaluation.

    The optimizer checks this token against the proposal's stored generation
    provenance when completing or cancelling it. Pass it back unchanged.
    """

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("AskId.value must be an integer")
        if self.value < 0 or self.value >= 2**64:
            raise ValueError("AskId.value must fit in an unsigned 64-bit integer")


@dataclass(frozen=True)
class AskBatch:
    """Coordinates, evaluation identities, and structured PyTree views from one ask."""

    ids: tuple[AskId, ...]
    points: np.ndarray
    spec: Any = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[0] != len(self.ids):
            raise ValueError("AskBatch points must have one row per evaluation ID")

    @property
    def params(self) -> tuple[Any, ...]:
        """Return structured parameter PyTrees (one per arm) when initialized with a PyTreeSpec."""
        if self.spec is not None:
            return self.spec.unflatten_batch(self.points)
        return tuple(self.points[i] for i in range(self.points.shape[0]))

    @property
    def stacked(self) -> Any:
        """Return a single PyTree whose leaves have leading batch dimension `(num_arms, *shape)`."""
        if self.spec is not None:
            return self.spec.unflatten_stacked(self.points)
        return self.points

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.points, dtype=dtype)


def _obs(x: np.ndarray, y: np.ndarray, yvar: np.ndarray | None, d: int) -> _Obs:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError(f"x must have shape (n, {d}), got {x.shape}")
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must have shape ({x.shape[0]}, m), got {y.shape}")
    if yvar is not None:
        yvar = np.asarray(yvar, dtype=float)
        if yvar.ndim == 1:
            yvar = yvar.reshape(-1, 1)
        if yvar.shape != y.shape:
            raise ValueError(f"y_var must have shape {y.shape}, got {yvar.shape}")
    return _Obs(x, y, yvar)


class TellPredictionError(RuntimeError):
    """Native ingestion succeeded, but retrieving the fitted mean failed.

    Retry ``optimizer.posterior_mu(x)``, not ``optimizer.tell(x, y)``.
    This does not classify failures raised by native ingestion itself.
    """

    observations_accepted = True


def _plan(cfg: OptimizerConfig) -> _Plan:
    if is_lhd(cfg):
        return _Plan("lhd")
    if isinstance(cfg.surrogate, ENNSurrogate):
        return _Plan("enn", enn_k(cfg))
    if isinstance(cfg.surrogate, NoSurrogate):
        return _Plan("zero")
    if isinstance(cfg.surrogate, GaussianProcess):
        from .turbo.gp import Surrogate

        return _Plan("gp", gp=Surrogate())
    if isinstance(cfg.surrogate, ExternalSurrogate):
        return _Plan(
            "gp", gp=cfg.surrogate.factory(), fit_steps=cfg.surrogate.fit_steps
        )
    raise ValueError(f"unsupported surrogate: {type(cfg.surrogate)!r}")


def _submit(
    opt: Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    y_var: np.ndarray | None,
    *,
    imported: bool,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    y_var_arr = None if y_var is None else np.asarray(y_var, dtype=float)
    obs = _obs(x_arr, y_arr, y_var_arr, opt._d)
    if obs.y.shape[0] == 0:
        return y_arr
    # Preserve the existing read-only input contract, after shape checks.
    x_arr.flags.writeable = False
    y_arr.flags.writeable = False
    if y_var_arr is not None:
        y_var_arr.flags.writeable = False
    submit = opt._inner.import_observations if imported else opt._inner.tell
    submit(obs.x, obs.y, y_var=obs.yvar)
    if not opt._inner.has_surrogate():
        return y_arr
    try:
        mu = opt.posterior_mu(obs.x)
    except Exception as error:
        raise TellPredictionError(
            "observations accepted, but prediction retrieval failed; "
            "retry posterior_mu(x), not the submission"
        ) from error
    if y_arr.ndim == 1 and mu.ndim == 2 and mu.shape[1] == 1:
        return mu.reshape(-1)
    return mu


class Optimizer:
    """Optimize a bounded search space through Rust-owned ask/tell state.

    ``ask`` returns parameter rows for objective evaluation. ``tell`` records
    objective values and optional observation variances. Objective evaluation
    belongs to the caller; constructing this object does not select a Python
    array framework or imply that posterior computation runs on an accelerator.
    """

    def __init__(
        self, d: int, rng: Generator, inner: Any, *, identified_asks: bool = False
    ) -> None:
        self._d = d
        self._rng = rng
        self._inner = inner
        self._identified_asks = identified_asks

    @property
    def _xobs(self) -> _View:
        x = self._inner.x_obs()
        return _View(np.empty((0, self._d)) if x is None else x)

    @property
    def _yobs(self) -> _View:
        y = self._inner.y_obs()
        return _View(np.empty((0, 1)) if y is None else y)

    @property
    def num_observations(self) -> int:
        """Number of completed observations retained by the optimizer."""
        return int(self._inner.region_count())

    def __len__(self) -> int:
        return int(self._inner.region_count())

    region_count = num_observations

    @property
    def num_regions(self) -> int:
        """Number of configured local region slots."""
        return int(self._inner.num_regions())

    @property
    def tr_length(self) -> float:
        return float(self._inner.tr_length())

    @property
    def init_progress(self) -> tuple[int, int] | None:
        return self._inner.init_progress()

    @property
    def ingestion_failure(self) -> str | None:
        """Reason this regional optimizer must not accept another submission."""
        return getattr(self._inner, "ingestion_failure", None)

    def telemetry(self) -> Telemetry:
        t = self._inner.telemetry()
        return Telemetry(
            dt_fit=t.dt_fit,
            dt_gen=t.dt_gen,
            dt_sel=t.dt_sel,
            dt_tell=t.dt_tell,
            num_candidates=int(t.num_candidates),
        )

    def ask(self, num_arms: int) -> np.ndarray:
        """Return `(num_arms, d)` candidates inside the configured bounds."""
        n = int(num_arms)
        if n <= 0:
            raise ValueError(f"num_arms must be > 0, got {n}")
        return np.asarray(self._inner.ask(n), dtype=float)

    def ask_batch(self, num_arms: int) -> AskBatch:
        """Ask for points with IDs checked against proposal generation.

        This lifecycle is available for multi-region optimizers, whose pending
        evaluations are completed or cancelled by identity.
        """
        if not self._identified_asks:
            raise RuntimeError("identified asks require a multi-region optimizer")
        before = set(self.pending_ids or ())
        points = self.ask(num_arms)
        pending = self.pending_ids
        if pending is None:
            raise RuntimeError("identified asks require a pending regional batch")
        new_ids = tuple(identity for identity in pending if identity not in before)
        if len(new_ids) != len(points):
            raise RuntimeError(
                "native ask returned a different number of new IDs than points"
            )
        return AskBatch(tuple(AskId(value) for value in new_ids), points)

    def complete(
        self,
        ids: AskBatch | tuple[AskId, ...] | list[AskId],
        y: np.ndarray,
        y_var: np.ndarray | None = None,
    ) -> None:
        """Complete a subset of a pending batch using typed evaluation IDs."""
        identities = ids.ids if isinstance(ids, AskBatch) else ids
        if any(not isinstance(identity, AskId) for identity in identities):
            raise TypeError("complete expects AskId values")
        self.complete_ids(tuple(identity.value for identity in identities), y, y_var)

    def cancel(self, ids: AskBatch | tuple[AskId, ...] | list[AskId]) -> None:
        """Cancel unevaluated members of a pending batch by typed ID."""
        identities = ids.ids if isinstance(ids, AskBatch) else ids
        if any(not isinstance(identity, AskId) for identity in identities):
            raise TypeError("cancel expects AskId values")
        self.cancel_ids(tuple(identity.value for identity in identities))

    def restart_region(self, slot: int, center: np.ndarray) -> tuple[AskId, ...]:
        """Restart a regional slot at a natural-coordinate center.

        Outstanding IDs owned by that region are returned as abandoned IDs.
        Reject their late results with ``complete`` and add measured late
        observations through ``import_observations`` instead.
        """
        if not self._identified_asks:
            raise RuntimeError("region restart requires a multi-region optimizer")
        center_arr = np.array(center, dtype=float, copy=True)
        if center_arr.shape != (self._d,):
            raise ValueError(
                f"center must have shape ({self._d},), got {center_arr.shape}"
            )
        center_arr.flags.writeable = False
        abandoned = self._inner.restart_region(index(slot), center_arr)
        return tuple(AskId(int(identity)) for identity in abandoned)

    @property
    def pending_ids(self) -> tuple[int, ...] | None:
        """Outstanding warm-up or regional IDs in original ask order."""
        ids = self._inner.pending_ids()
        return None if ids is None else tuple(int(identity) for identity in ids)

    def complete_ids(
        self,
        ids: list[int] | tuple[int, ...],
        y: np.ndarray,
        y_var: np.ndarray | None = None,
    ) -> None:
        """Record results for a subset of a pending regional batch.

        Results may arrive in any order. The batch remains pending until each
        ID has completed; only then does regional controller credit apply.
        """
        identities = tuple(index(identity) for identity in ids)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape != (len(identities), 1):
            raise ValueError(
                f"y must have shape ({len(identities)}, 1), got {y_arr.shape}"
            )
        yvar_arr = None if y_var is None else np.asarray(y_var, dtype=float)
        if yvar_arr is not None:
            if yvar_arr.ndim == 1:
                yvar_arr = yvar_arr.reshape(-1, 1)
            if yvar_arr.shape != y_arr.shape:
                raise ValueError(
                    f"y_var must have shape {y_arr.shape}, got {yvar_arr.shape}"
                )
            yvar_arr.flags.writeable = False
        y_arr.flags.writeable = False
        self._inner.complete_ids(identities, y_arr, y_var=yvar_arr)

    def cancel_ids(self, ids: list[int] | tuple[int, ...]) -> None:
        """Retire unevaluated IDs without recording a reward or progress credit."""
        self._inner.cancel_ids(tuple(index(identity) for identity in ids))

    def tell(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        """Record observations and return fitted means, or y without a model.

        A 1D scalar objective retains its shape in the return value. If native
        ingestion succeeds but prediction retrieval fails, raise
        ``TellPredictionError``; repeat only ``posterior_mu(x)`` in that case.
        Native ingestion errors do not imply transactional rollback.
        """
        return _submit(self, x, y, y_var, imported=False)

    def import_observations(
        self, x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None = None
    ) -> np.ndarray:
        """Add external evaluations without completing an outstanding proposal."""
        return _submit(self, x, y, y_var, imported=True)

    def posterior_mu(self, x: np.ndarray) -> np.ndarray:
        """Query fitted means in natural units without submitting observations."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 2 or x_arr.shape[1] != self._d:
            raise ValueError(f"x must have shape (n, {self._d}), got {x_arr.shape}")
        return np.asarray(self._inner.posterior_mu(x_arr), dtype=float)


def create_optimizer(
    *, bounds: np.ndarray, config: OptimizerConfig, rng: Generator
) -> Optimizer:
    """Build an optimizer from the Python config."""
    from . import _rust

    if not supports(config):
        raise ValueError(f"unsupported optimizer config: {config!r}")

    bounds = np.asarray(bounds, dtype=float)
    plan = _plan(config)
    n = 10 if config.init.num_init is None else int(config.init.num_init)
    seed = int(rng.integers(2**63 - 1))
    regions = (
        config.trust_region.count
        if isinstance(config.trust_region, RegionalSearchPolicy)
        else 1
    )
    provider_factory: Callable[[], Any] | None = None
    if isinstance(config.trust_region, RegionalSearchPolicy):
        if isinstance(config.surrogate, ExternalSurrogate):
            provider_factory = config.surrogate.factory
        elif isinstance(config.surrogate, GaussianProcess):
            from .turbo.gp import Surrogate

            provider_factory = Surrogate
    gp_factory = None
    if provider_factory is not None:

        def gp_factory(slot: int, generation: int) -> Any:
            return provider_factory()

    inner = _rust.create_optimizer(
        bounds,
        plan.kind,
        plan.k,
        n,
        regions,
        seed,
        cfg=encode(config),
        gp=plan.gp,
        gp_factory=gp_factory,
        fit_steps=plan.fit_steps,
    )
    return Optimizer(
        len(bounds),
        rng,
        inner,
        identified_asks=isinstance(config.trust_region, RegionalSearchPolicy),
    )


__all__ = [
    "AskBatch",
    "AskId",
    "Optimizer",
    "Telemetry",
    "TellPredictionError",
    "create_optimizer",
]
