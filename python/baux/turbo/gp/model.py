"""GPyTorch ExactGP model definitions for TuRBO surrogates."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpytorch.distributions import MultivariateNormal


def _base():
    with warnings.catch_warnings():
        # linear-operator 0.6.1 still compiles two helpers with TorchScript.
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        from gpytorch.models import ExactGP

    return ExactGP


class Base(_base()):
    """Base ExactGP model providing Matern covariance and posterior evaluation."""

    mean_module: Any
    covar_module: Any

    def forward(self, x) -> MultivariateNormal:
        """Compute the prior multivariate normal distribution at `x`."""
        from gpytorch.distributions import MultivariateNormal

        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def posterior(self, x) -> MultivariateNormal:
        """Evaluate the predictive posterior distribution at `x`."""
        return self(x)


class Exact(Base):
    """Exact Gaussian process with learned homoscedastic noise."""

    def __init__(self, x, y, like, ls_bound, scale_bound, d: int) -> None:
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

        super().__init__(x, y, like)
        batch = torch.Size(y.shape[:-1]) if getattr(y, "ndim", 0) > 1 else torch.Size()
        self.mean_module = ConstantMean(batch_shape=batch)
        kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            batch_shape=batch,
            lengthscale_constraint=ls_bound,
        )
        self.covar_module = ScaleKernel(
            kernel,
            batch_shape=batch,
            outputscale_constraint=scale_bound,
        )


class Noisy(Base):
    """Gaussian process with fixed observation noise variances."""

    def __init__(
        self,
        x,
        y,
        var,
        ls_bound,
        scale_bound,
        d: int,
        *,
        learn_noise: bool = True,
    ) -> None:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
        from gpytorch.means import ConstantMean

        like = FixedNoiseGaussianLikelihood(
            noise=var,
            learn_additional_noise=learn_noise,
        )
        super().__init__(x, y, like)
        self.mean_module = ConstantMean()
        kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=d,
            lengthscale_constraint=ls_bound,
        )
        self.covar_module = ScaleKernel(
            kernel,
            outputscale_constraint=scale_bound,
        )
