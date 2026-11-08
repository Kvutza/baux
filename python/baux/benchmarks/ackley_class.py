"""Single-objective Ackley benchmark wrapper with Gaussian observation noise."""

import numpy as np
from numpy.random import Generator

from .ackley_core import ackley_core


class Ackley:
    """Noisy maximization wrapper around the Ackley benchmark function."""

    def __init__(self, noise: float, rng: Generator):
        self.noise = noise
        self.rng = rng
        self.bounds = [-32.768, 32.768]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate negated Ackley objective values at `x` with additive noise."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        y = -ackley_core(x) + self.noise * self.rng.normal(size=(x.shape[0],))
        return np.asarray(y, dtype=float)
