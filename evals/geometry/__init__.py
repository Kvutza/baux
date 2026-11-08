"""Riemannian geometry, manifold embeddings, Stiefel Stein contraction, and WLS affine calibration evaluations."""

from __future__ import annotations

from .eval_manifolds import (
    eval_affine,
    eval_embeddings,
    eval_hyperbolic,
    eval_pmap,
    eval_stein,
    run_evals,
)
from .eval_manifolds import (
    eval_affine as eval_wls_affine,
)
from .eval_manifolds import (
    eval_embeddings as eval_manifold_embeddings,
)
from .eval_manifolds import (
    eval_hyperbolic as eval_hyperbolic_knn,
)
from .eval_manifolds import (
    eval_pmap as eval_parallel_pmap,
)
from .eval_manifolds import (
    eval_stein as eval_stiefel_stein,
)
from .eval_manifolds import (
    run_evals as run_all_evals,
)

__all__ = [
    "eval_affine",
    "eval_embeddings",
    "eval_hyperbolic",
    "eval_hyperbolic_knn",
    "eval_manifold_embeddings",
    "eval_parallel_pmap",
    "eval_pmap",
    "eval_stein",
    "eval_stiefel_stein",
    "eval_wls_affine",
    "run_all_evals",
    "run_evals",
]
