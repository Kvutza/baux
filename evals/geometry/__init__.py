"""Riemannian geometry, manifold embeddings, Stiefel Stein shrinkage, and WLS calibration evaluations."""

from __future__ import annotations

from .eval_manifolds import (
    eval_calibrator,
    eval_calibrator as eval_wls_calibrator,
    eval_embeddings,
    eval_embeddings as eval_manifold_embeddings,
    eval_hyperbolic,
    eval_hyperbolic as eval_hyperbolic_knn,
    eval_pmap,
    eval_pmap as eval_parallel_pmap,
    eval_stein,
    eval_stein as eval_stein_shrinkage,
    run_evals,
    run_evals as run_all_evals,
)

__all__ = [
    "eval_calibrator",
    "eval_embeddings",
    "eval_hyperbolic",
    "eval_hyperbolic_knn",
    "eval_manifold_embeddings",
    "eval_parallel_pmap",
    "eval_pmap",
    "eval_stein",
    "eval_stein_shrinkage",
    "eval_wls_calibrator",
    "run_all_evals",
    "run_evals",
]
