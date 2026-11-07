from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from . import baux_rust as _ext
except ImportError as exc:  # pragma: no cover - exercised when extension unavailable
    raise ImportError(
        "Rust extension submodule `baux.baux_rust` is not available"
    ) from exc

cuda = getattr(_ext, "cuda", None)


hypervolume2d_max = _ext.hypervolume.hypervolume2d_max
normal_hash = _ext.hash.normal_hash
standardize_y = _ext.util.standardize_y
pareto2d_max = _ext.util.pareto2d_max
sobol_indices = _ext.util.sobol_indices
sobol_sequence = _ext.util.sobol_sequence
pareto_arms = _ext.util.pareto_arms
quantize_int4 = _ext.util.quantize_int4
quantize_e2m1 = _ext.util.quantize_e2m1
set_path = _ext.util.set_path
ensure_config_file = _ext.util.ensure_config_file
ENN = _ext.model.ENN
Enn = getattr(_ext.model, "Enn", _ext.model.ENN)
ENNParams = _ext.model.ENNParams
Posterior = getattr(
    _ext.model, "Posterior", getattr(_ext.model, "DevicePosterior", None)
)
DevicePosterior = Posterior
Tensor = getattr(_ext.model, "Tensor", getattr(_ext.model, "DeviceArray", None))
DeviceArray = Tensor
Device = getattr(_ext, "Device", None) or getattr(_ext.optimizer, "Device", None)
ENNFitter = _ext.fit.ENNFitter
subsample_loglik = _ext.fit.subsample_loglik
row_loglik = _ext.fit.row_loglik
BayesOptimizer = _ext.optimizer.BayesOptimizer
Telemetry = _ext.optimizer.Telemetry
MultiTrustRegion = _ext.optimizer.MultiTrustRegion
BpannHistory = getattr(_ext.optimizer, "BpannHistory", None)
create_optimizer = _ext.optimizer.create_optimizer
enn_optimizer = _ext.optimizer.enn_optimizer
enn_tr = _ext.optimizer.enn_tr
create_zero = _ext.optimizer.create_zero
create_lhd = _ext.optimizer.create_lhd
dense_apply = getattr(_ext.optimizer, "dense_apply", None)
dense_dist2 = getattr(_ext.optimizer, "dense_dist2", None)
dense_linear = getattr(_ext.optimizer, "dense_linear", None)
DenseLinear = getattr(_ext.optimizer, "DenseLinear", None)
Var = (
    getattr(_ext.optimizer, "Var", None)
    or getattr(_ext.experimental, "Var", None)
    or getattr(_ext.optimizer, "ParameterBuffer", None)
    or getattr(_ext.experimental, "ParameterBuffer", None)
    or getattr(_ext.optimizer, "ParamBuffer", None)
    or getattr(_ext.experimental, "MetalWeights", None)
)
Span = (
    getattr(_ext.optimizer, "Span", None)
    or getattr(_ext.experimental, "Span", None)
    or getattr(_ext.optimizer, "ParamBlock", None)
    or getattr(_ext.experimental, "ParamBlock", None)
    or getattr(_ext.experimental, "MetalParamBlock", None)
)
Region = (
    getattr(_ext.optimizer, "Region", None)
    or getattr(_ext.experimental, "Region", None)
    or getattr(_ext.optimizer, "SearchState", None)
    or getattr(_ext.experimental, "SearchState", None)
    or getattr(_ext.experimental, "MetalSearchState", None)
)
Trial = (
    getattr(_ext.optimizer, "Trial", None)
    or getattr(_ext.experimental, "Trial", None)
    or getattr(_ext.optimizer, "Proposals", None)
    or getattr(_ext.experimental, "Proposals", None)
    or getattr(_ext.experimental, "MetalProposals", None)
)
Portfolio = (
    getattr(_ext.optimizer, "Portfolio", None)
    or getattr(_ext.experimental, "Portfolio", None)
    or getattr(_ext.experimental, "RegionOptimizer", None)
    or getattr(_ext.experimental, "MetalRegionOptimizer", None)
)
RegionTrial = (
    getattr(_ext.optimizer, "RegionTrial", None)
    or getattr(_ext.experimental, "RegionTrial", None)
    or getattr(_ext.experimental, "RegionalProposal", None)
)
RegionalProposal = RegionTrial
ParameterBuffer = Var
ParamBuffer = Var
MetalWeights = Var
ParamBlock = Span
MetalParamBlock = Span
SearchState = Region
MetalSearchState = Region
Proposals = Trial
MetalProposals = Trial
RegionOptimizer = Portfolio
MetalRegionOptimizer = Portfolio
weight_int4_select_ucb = getattr(_ext.optimizer, "weight_int4_select_ucb", None)
weight_select_ucb = getattr(_ext.optimizer, "weight_select_ucb", None)
ModelPackage = getattr(_ext.experimental, "ModelPackage", None)
NativeKdaModel = getattr(_ext.experimental, "NativeKdaModel", None)
FlameEvaluator = getattr(_ext.experimental, "FlameEvaluator", None)
MetalFlameEvaluator = getattr(_ext.experimental, "MetalFlameEvaluator", None)
MetalQwenEvaluator = getattr(_ext.experimental, "MetalQwenEvaluator", None)
ResidentBoSession = getattr(_ext.experimental, "ResidentBoSession", None)


__all__ = [
    "ENN",
    "BayesOptimizer",
    "BpannHistory",
    "DenseLinear",
    "Device",
    "DeviceArray",
    "DevicePosterior",
    "ENNFitter",
    "ENNParams",
    "Enn",
    "FlameEvaluator",
    "MetalFlameEvaluator",
    "MetalParamBlock",
    "MetalProposals",
    "MetalQwenEvaluator",
    "MetalRegionOptimizer",
    "MetalSearchState",
    "MetalWeights",
    "ModelPackage",
    "MultiTrustRegion",
    "NativeKdaModel",
    "ParamBlock",
    "ParamBuffer",
    "ParameterBuffer",
    "Portfolio",
    "Posterior",
    "Proposals",
    "Region",
    "RegionOptimizer",
    "RegionTrial",
    "RegionalProposal",
    "ResidentBoSession",
    "SearchState",
    "Span",
    "Telemetry",
    "Tensor",
    "Trial",
    "Var",
    "create_lhd",
    "create_optimizer",
    "create_zero",
    "dense_apply",
    "dense_dist2",
    "dense_linear",
    "enn_optimizer",
    "enn_tr",
    "ensure_config_file",
    "hypervolume2d_max",
    "normal_hash",
    "pareto2d_max",
    "pareto_arms",
    "quantize_e2m1",
    "quantize_int4",
    "row_loglik",
    "set_path",
    "sobol_indices",
    "sobol_sequence",
    "standardize_y",
    "subsample_loglik",
    "weight_int4_select_ucb",
    "weight_select_ucb",
]
