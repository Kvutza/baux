"""Shared numerical boundary for Python trial managers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .bo import AskId
from .functional import ask, cancel, complete, import_observations, init, tell
from .state import OptimizerState
from .turbo.config import OptimizerConfig, RegionalSearchPolicy, turbo_enn


class Space:
    def __init__(self, bounds: Mapping[str, tuple[float, float]], logs=()) -> None:
        if not bounds or any(not isinstance(name, str) for name in bounds):
            raise ValueError("the search space needs named continuous parameters")
        self.names = tuple(sorted(bounds))
        self.limits = np.asarray([bounds[name] for name in self.names], dtype=float)
        if self.limits.shape != (len(bounds), 2):
            raise ValueError("each parameter needs a lower and upper bound")
        if not np.isfinite(self.limits).all() or np.any(
            self.limits[:, 0] >= self.limits[:, 1]
        ):
            raise ValueError("bounds must be finite and strictly increasing")
        if not set(logs).issubset(self.names):
            raise ValueError("log parameters must belong to the search space")
        self.logs = np.array([name in logs for name in self.names], dtype=np.bool_)
        if np.any(self.limits[self.logs] <= 0):
            raise ValueError("log bounds must be positive")
        self.bounds = self.limits.copy()
        self.bounds[self.logs] = np.log(self.bounds[self.logs])

    def encode(self, parameters: Mapping[str, float]) -> np.ndarray:
        if set(parameters) != set(self.names):
            raise ValueError("parameters must match the fixed search space exactly")
        values = np.asarray([parameters[name] for name in self.names], dtype=float)
        if not np.isfinite(values).all() or np.any(
            (values < self.limits[:, 0]) | (values > self.limits[:, 1])
        ):
            raise ValueError("parameters must be finite and inside their bounds")
        values[self.logs] = np.log(values[self.logs])
        return values

    def decode(self, values: np.ndarray) -> dict[str, float]:
        values = np.asarray(values, dtype=float).copy()
        values[self.logs] = np.exp(values[self.logs])
        values = np.clip(values, self.limits[:, 0], self.limits[:, 1])
        return dict(zip(self.names, values.tolist(), strict=True))


class Loop:
    """Retain a live optimizer; ingest each completed observation exactly once.

    Reconstructing this object from observations is a warm restart, not a native
    optimizer checkpoint. Trial managers must have a single generating process.
    """

    def __init__(
        self, space: Space, config: OptimizerConfig | None, seed: int | None = None
    ) -> None:
        if config is not None and config.num_metrics is not None:
            raise ValueError(
                "trial adapters require a single-objective optimizer config"
            )
        self.space = space
        resolved_config = turbo_enn() if config is None else config
        self._regional = isinstance(resolved_config.trust_region, RegionalSearchPolicy)
        self.state: OptimizerState = init(
            space.bounds,
            resolved_config,
            seed=seed,
        )
        self._seen: dict[str, tuple[np.ndarray, float, float | None]] = {}
        self._issued: dict[tuple[float, ...], AskId] = {}

    @property
    def optimizer(self) -> Any:
        return self.state

    def check_health(self) -> None:
        failure = self.state.ingestion_failure
        if failure is not None:
            raise RuntimeError(
                "baux ingestion failed after mutation may have begun; this adapter "
                "cannot safely retry or accept more observations. Create a new "
                f"adapter from completed trial history. Cause: {failure}"
            )

    def update(self, records: Mapping[str, tuple[Mapping, float, float | None]]):
        self.check_health()
        if not self._seen.keys() <= records.keys():
            raise ValueError("completed observations were removed from the history")
        checked = {key: self._record(*record) for key, record in records.items()}
        if len({record[2] is None for record in checked.values()}) > 1:
            raise ValueError(
                "mixed known and unknown observation variances are unsupported"
            )
        for key, record in checked.items():
            if key in self._seen and not self._equal(self._seen[key], record):
                raise ValueError("completed observations changed; create a new adapter")
        for key, (point, score, variance) in checked.items():
            if key in self._seen:
                continue
            self._ingest(point, score, variance)
            self._seen[key] = (point, score, variance)

    def _ingest(self, point, score, variance):
        point_key = tuple(point.tolist())
        identity = self._issued.get(point_key)
        noise = None if variance is None else np.array([[variance]])
        y = np.array([[score]])
        if identity is None:
            if self._regional:
                self.state, _ = import_observations(
                    self.state, point[None, :], y, y_var=noise
                )
            else:
                self.state, _ = tell(self.state, point[None, :], y, y_var=noise)
        else:
            self.state = complete(self.state, [identity], y, y_var=noise)
            self._issued.pop(point_key)
        self.check_health()

    def _record(self, parameters, score, variance):
        point = self.space.encode(parameters)
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("completed objective values must be finite")
        if variance is not None:
            variance = float(variance)
            if not np.isfinite(variance) or variance < 0:
                raise ValueError("observation variance must be finite and nonnegative")
        return point, score, variance

    @staticmethod
    def _equal(left, right):
        return np.array_equal(left[0], right[0]) and left[1:] == right[1:]

    def suggest(self, pending: Sequence[Mapping[str, float]]) -> dict[str, float]:
        self.check_health()
        occupied = [self.space.encode(point) for point in pending]
        for _ in range(32):
            self.state, batch = ask(self.state, 1)
            point = batch.points[0]
            identity = batch.ids[0] if self._regional else None
            proposal = self.space.decode(point)
            point_re = self.space.encode(proposal)
            if not any(np.array_equal(point_re, other) for other in occupied):
                if identity is not None:
                    self._issued[tuple(point_re.tolist())] = identity
                return proposal
            if identity is not None:
                self.state = cancel(self.state, [identity])
        raise RuntimeError(
            "baux could not propose a point distinct from pending trials"
        )

    def sync(self):
        """Synchronize observations if supported by the optimizer/region."""
        return None

    def fork(self) -> Loop:
        """Fork the trial loop with an independent copy of the optimizer state."""
        clone = object.__new__(self.__class__)
        clone.space = self.space
        clone._regional = self._regional
        clone.state = self.state.fork()
        clone._seen = self._seen.copy()
        clone._issued = self._issued.copy()
        return clone
