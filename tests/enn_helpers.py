from __future__ import annotations

from typing import Any, NamedTuple


class EnnFixture(NamedTuple):
    model: Any
    train_x: Any
    train_y: Any
    train_yvar: Any
    rng: Any

    @property
    def state(self) -> Any:
        return self.model


def sphere_objective(x):
    import numpy as np

    return -np.sum(x**2, axis=1)


def enn_model(n=20, d=3, seed=0, yvar_scale=0.1):
    import baux
    import numpy as np

    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = yvar_scale * np.ones_like(train_y)
    state = baux.enn_init(train_x, train_y, train_yvar)
    return EnnFixture(state, train_x, train_y, train_yvar, rng)


def enn_rows(model):
    """Return (x, y, yvar?) for all rows via index-based gather."""
    if hasattr(model, "rust_backend") and hasattr(model.rust_backend, "train_rows"):
        n = len(model)
        return model.rust_backend.train_rows(list(range(n)))
    if hasattr(model, "train_rows"):
        n = len(model)
        return model.train_rows(list(range(n)))
    return model.x, model.y, model.y_var
