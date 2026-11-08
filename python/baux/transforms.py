"""Optax-style composable functional pipeline combinators for `OptimizerConfig`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from .turbo.config.acquisition import (
    DrawAcquisitionConfig,
    NDSOptimizerConfig,
    ParetoAcquisitionConfig,
    RAASPOptimizerConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .turbo.config.candidates import CandidateConfig, CandidateSampler, RAASPDriver
from .turbo.config.initialization import HybridInit, InitConfig, LHDOnlyInit
from .turbo.config.morbo_region import TrustRegionBounds
from .turbo.config.optimizer import OptimizerConfig
from .turbo.config.surrogate import (
    ENNSurrogate,
    GaussianProcess,
    NeighborIndex,
    NoSurrogate,
    SurrogateFit,
)
from .turbo.config.trust_region import (
    NoTrustRegion,
    RegionalSearchPolicy,
    TurboTrustRegion,
)


@dataclass(frozen=True)
class Transform:
    """Composable pure transformation `OptimizerConfig -> OptimizerConfig`."""

    name: str
    fn: Callable[[OptimizerConfig], OptimizerConfig]

    def __call__(self, config: OptimizerConfig | None = None) -> OptimizerConfig:
        base = OptimizerConfig() if config is None else config
        return self.fn(base)

    def __or__(
        self, other: Transform | Callable[[OptimizerConfig], OptimizerConfig]
    ) -> Transform:
        other_name = getattr(other, "name", getattr(other, "__name__", "step"))
        return Transform(
            name=f"{self.name} | {other_name}",
            fn=lambda cfg: other(self(cfg)),
        )


def chain(
    *steps: Transform | Callable[[OptimizerConfig], OptimizerConfig] | OptimizerConfig,
) -> OptimizerConfig:
    """Compose left-to-right pipeline transforms into a validated `OptimizerConfig`."""
    cfg = OptimizerConfig()
    for step in steps:
        if isinstance(step, OptimizerConfig):
            cfg = step
        else:
            cfg = step(cfg)
    return cfg


def lhd(num_init: int = 10, *, only: bool = False) -> Transform:
    """Configure Latin Hypercube Design warm-up (or pure LHD exploration when `only=True`)."""
    n = int(num_init)

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        if only:
            return replace(
                cfg,
                trust_region=NoTrustRegion(),
                init=InitConfig(init_strategy=LHDOnlyInit(), num_init=n),
                surrogate=NoSurrogate(),
                acquisition=RandomAcquisitionConfig(),
                acq_optimizer=RAASPOptimizerConfig(),
            )
        return replace(cfg, init=InitConfig(init_strategy=HybridInit(), num_init=n))

    return Transform(f"lhd({n}, only={only})", apply)


def raasp(
    num_pert: int | None = 20,
    *,
    num_candidates: int | None = None,
    per_arm: int | None = None,
    driver: RAASPDriver = RAASPDriver.ORIG,
) -> Transform:
    """Configure RAASP sparse coordinate perturbation candidate sampling."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        cand = CandidateConfig(
            sampler=CandidateSampler.RAASP,
            num_candidates=num_candidates
            if num_candidates is not None
            else cfg.candidates.num_candidates,
            num_candidates_per_arm=per_arm
            if per_arm is not None
            else cfg.candidates.num_candidates_per_arm,
            num_pert=num_pert,
            raasp_driver=driver,
        )
        return replace(cfg, candidates=cand)

    return Transform(f"raasp({num_pert})", apply)


def sobol(
    num_candidates: int | None = None,
    *,
    per_arm: int | None = None,
    num_pert: int | None = None,
) -> Transform:
    """Configure scrambled Sobol quasi-Monte Carlo candidate sampling."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        cand = CandidateConfig(
            sampler=CandidateSampler.SOBOL,
            num_candidates=num_candidates
            if num_candidates is not None
            else cfg.candidates.num_candidates,
            num_candidates_per_arm=per_arm
            if per_arm is not None
            else cfg.candidates.num_candidates_per_arm,
            num_pert=num_pert if num_pert is not None else cfg.candidates.num_pert,
            raasp_driver=cfg.candidates.raasp_driver,
        )
        return replace(cfg, candidates=cand)

    return Transform(f"sobol({num_candidates})", apply)


def uniform_pert(
    num_candidates: int | None = None,
    *,
    per_arm: int | None = None,
    num_pert: int | None = None,
) -> Transform:
    """Configure uniform random trust-region perturbation candidate sampling."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        cand = CandidateConfig(
            sampler=CandidateSampler.UNIFORM,
            num_candidates=num_candidates
            if num_candidates is not None
            else cfg.candidates.num_candidates,
            num_candidates_per_arm=per_arm
            if per_arm is not None
            else cfg.candidates.num_candidates_per_arm,
            num_pert=num_pert if num_pert is not None else cfg.candidates.num_pert,
            raasp_driver=cfg.candidates.raasp_driver,
        )
        return replace(cfg, candidates=cand)

    return Transform(f"uniform_pert({num_candidates})", apply)


def _driver(index: Any) -> NeighborIndex:
    if isinstance(index, NeighborIndex):
        return index
    val = str(getattr(index, "value", index)).lower()
    mapping = {
        "exact": NeighborIndex.FLAT,
        "flat": NeighborIndex.FLAT,
        "hnsw": NeighborIndex.USEARCH,
        "usearch": NeighborIndex.USEARCH,
        "tree": NeighborIndex.BPANN_DISK,
        "bpann": NeighborIndex.BPANN_DISK,
        "bpann_disk": NeighborIndex.BPANN_DISK,
        "metal": NeighborIndex.METAL,
        "cuda": NeighborIndex.CUDA,
        "auto": NeighborIndex.AUTO,
    }
    if val not in mapping:
        raise ValueError(f"Unsupported index driver: {index!r}")
    return mapping[val]


def enn(
    *,
    k: int | None = 10,
    num_samples: int | None = 100,
    scale_x: bool = False,
    index: Any = NeighborIndex.FLAT,
    storage: str | None = None,
    metric_learning: str = "none",
    geometry: str = "euclidean",
    local_k: int = 4,
    stiefel: Any = None,
    affine: Any = None,
    tied_dims: tuple[tuple[int, ...], ...] | None = None,
) -> Transform:
    """Configure the Epistemic Nearest Neighbors (ENN) surrogate model."""
    drv = _driver(index)

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        surrogate = ENNSurrogate(
            k=k,
            fit=SurrogateFit(num_samples=num_samples),
            scale_x=scale_x,
            index_driver=drv,
            enn_storage=storage,
            metric_learning=metric_learning,
            geometry=geometry,
            local_k=local_k,
            stiefel=stiefel,
            affine=affine,
            tied_dims=tied_dims,
        )
        acq = cfg.acquisition
        acq_opt = cfg.acq_optimizer
        if isinstance(acq, RandomAcquisitionConfig):
            if num_samples is None:
                acq = ParetoAcquisitionConfig()
                acq_opt = NDSOptimizerConfig()
            else:
                acq = UCBAcquisitionConfig()
                acq_opt = RAASPOptimizerConfig()
        return replace(cfg, surrogate=surrogate, acquisition=acq, acq_optimizer=acq_opt)

    return Transform(f"enn(k={k})", apply)


def gp() -> Transform:
    """Configure the Gaussian Process (TuRBO-1) surrogate model."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        acq = cfg.acquisition
        acq_opt = cfg.acq_optimizer
        if isinstance(acq, RandomAcquisitionConfig):
            acq = DrawAcquisitionConfig()
            acq_opt = RAASPOptimizerConfig()
        return replace(
            cfg, surrogate=GaussianProcess(), acquisition=acq, acq_optimizer=acq_opt
        )

    return Transform("gp()", apply)


def zero() -> Transform:
    """Configure surrogate-free TuRBO-zero optimization."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        tr = cfg.trust_region
        if isinstance(tr, NoTrustRegion):
            tr = TurboTrustRegion()
        return replace(
            cfg,
            trust_region=tr,
            surrogate=NoSurrogate(),
            acquisition=RandomAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )

    return Transform("zero()", apply)


def _ensurefit(surrogate: Any) -> Any:
    if isinstance(surrogate, NoSurrogate):
        return ENNSurrogate(k=10, fit=SurrogateFit(num_samples=100))
    if isinstance(surrogate, ENNSurrogate) and surrogate.fit.num_samples is None:
        return replace(surrogate, fit=replace(surrogate.fit, num_samples=100))
    return surrogate


def ucb(beta: float = 2.0) -> Transform:
    """Configure Upper Confidence Bound (UCB) acquisition."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        return replace(
            cfg,
            surrogate=_ensurefit(cfg.surrogate),
            acquisition=UCBAcquisitionConfig(beta=beta),
            acq_optimizer=RAASPOptimizerConfig(),
        )

    return Transform(f"ucb(beta={beta})", apply)


def thompson() -> Transform:
    """Configure Thompson Sampling posterior draw acquisition."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        return replace(
            cfg,
            surrogate=_ensurefit(cfg.surrogate),
            acquisition=DrawAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )

    return Transform("thompson()", apply)


def pareto() -> Transform:
    """Configure Pareto non-dominated sorting (mu, sigma) acquisition."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        surrogate = cfg.surrogate
        if isinstance(surrogate, NoSurrogate):
            surrogate = ENNSurrogate(k=10)
        return replace(
            cfg,
            surrogate=surrogate,
            acquisition=ParetoAcquisitionConfig(),
            acq_optimizer=NDSOptimizerConfig(),
        )

    return Transform("pareto()", apply)


def random_acq() -> Transform:
    """Configure uniform random candidate selection within the trust region."""

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        return replace(
            cfg,
            acquisition=RandomAcquisitionConfig(),
            acq_optimizer=RAASPOptimizerConfig(),
        )

    return Transform("random_acq()", apply)


def tr_bounds(
    *,
    length_init: float = 0.8,
    length_min: float = 0.5**7,
    length_max: float = 1.6,
) -> Transform:
    """Configure trust-region side-length bounds `(length_init, length_min, length_max)`."""
    bounds = TrustRegionBounds(
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
    )

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        tr = cfg.trust_region
        if isinstance(tr, RegionalSearchPolicy):
            return replace(cfg, trust_region=replace(tr, length=bounds))
        return replace(cfg, trust_region=TurboTrustRegion(length=bounds))

    return Transform(
        f"tr_bounds(init={length_init}, min={length_min}, max={length_max})",
        apply,
    )


def regions(
    count: int = 4,
    *,
    sharing: Any = "ellipsoid_ball",
    allocation: Any = "incumbent_ucb1",
    success_tolerance: int = 3,
    failure_tolerance: int = 5,
    length: TrustRegionBounds | None = None,
) -> Transform:
    """Configure multi-region search (`RegionalSearchPolicy`) with sharing & allocation policies."""
    share_str = str(getattr(sharing, "value", sharing))
    alloc_str = str(getattr(allocation, "value", allocation))

    def apply(cfg: OptimizerConfig) -> OptimizerConfig:
        bounds = (
            length
            if length is not None
            else getattr(cfg.trust_region, "length", TrustRegionBounds())
        )
        policy = RegionalSearchPolicy(
            count=count,
            length=bounds,
            success_tolerance=success_tolerance,
            failure_tolerance=failure_tolerance,
            sharing=cast(Any, share_str),
            allocation=cast(Any, alloc_str),
        )
        return replace(cfg, trust_region=policy)

    return Transform(f"regions({count}, sharing={share_str!r})", apply)


__all__ = [
    "Transform",
    "chain",
    "enn",
    "gp",
    "lhd",
    "pareto",
    "raasp",
    "random_acq",
    "regions",
    "sobol",
    "thompson",
    "tr_bounds",
    "uniform_pert",
    "ucb",
    "zero",
]
