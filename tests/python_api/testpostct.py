"""Contract tests for pure functional ENN posterior and sample methods."""

from __future__ import annotations

import baux
import numpy as np
import pytest
from baux import ENNParams
from baux.enn.posterior_flags import PosteriorFlags


@pytest.fixture
def simple_model():
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=float)
    return baux.enn_init(train_x, train_y, scale_x=False)


@pytest.fixture
def query():
    return np.array([[0.5, 0.5]], dtype=float)


class TestPosteriorContract:
    """API contract tests for baux.posterior."""

    def test_001(self):
        assert hasattr(baux, "posterior")
        assert callable(baux.posterior)

    def test_002(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        out = baux.posterior(simple_model, query, params=params, flags=flags)
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)
        assert np.all(np.isfinite(out.mu))
        assert np.all(np.isfinite(out.se))


class TestBatchPosterior:
    """API contract tests for posterior evaluated across multiple parameter sets."""

    def test_003(self, simple_model, query):
        params1 = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        params2 = ENNParams(
            k_neighbors=3,
            epistemic_scale=2.0,
            aleatoric_scale=0.2,
        )
        outs = [
            baux.posterior(simple_model, query, params=p) for p in [params1, params2]
        ]
        mu = np.stack([o.mu for o in outs])
        se = np.stack([o.se for o in outs])
        se_epi = np.stack([o.se_epi for o in outs])
        se_ale = np.stack([o.se_ale for o in outs])

        assert mu.shape == (2, 1, 1)
        assert se.shape == (2, 1, 1)
        assert se_epi.shape == (2, 1, 1)
        assert se_ale.shape == (2, 1, 1)


class TestCondPosterior:
    """API contract tests for conditional posterior via functional state append."""

    def test_004(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        x_whatif = np.zeros((0, 2), dtype=float)
        y_whatif = np.zeros((0, 1), dtype=float)

        post = baux.posterior(simple_model, query, params=params, flags=flags)
        cond_model = (
            baux.append(simple_model, x_whatif, y_whatif)
            if len(x_whatif) > 0
            else simple_model
        )
        cond_post = baux.posterior(cond_model, query, params=params, flags=flags)

        np.testing.assert_allclose(post.mu, cond_post.mu, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(post.se, cond_post.se, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            post.se_epi, cond_post.se_epi, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            post.se_ale, cond_post.se_ale, rtol=1e-12, atol=1e-12
        )

    def test_005(self, simple_model, query):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()

        cond_model = baux.append(simple_model, x_whatif, y_whatif)
        out = baux.posterior(cond_model, query, params=params, flags=flags)
        assert hasattr(out, "mu")
        assert hasattr(out, "se")
        assert hasattr(out, "se_epi")
        assert hasattr(out, "se_ale")
        assert out.mu.shape == (1, 1)
        assert out.se.shape == (1, 1)
        assert out.se_epi.shape == (1, 1)
        assert out.se_ale.shape == (1, 1)


class TestPosteriorDraw:
    """API contract tests for baux.sample."""

    def test_006(self):
        assert hasattr(baux, "sample")
        assert callable(baux.sample)

    def test_007(self, simple_model, query):
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42, 43]

        draws, idx = baux.sample(
            simple_model, query, params=params, seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (1, 1, 2)
        assert isinstance(idx, (list, np.ndarray))
        assert len(idx) == 1

    def test_008(self, simple_model, query):
        x_whatif = np.array([[0.5, 0.5]], dtype=float)
        y_whatif = np.array([[1.5]], dtype=float)
        params = ENNParams(
            k_neighbors=2,
            epistemic_scale=1.0,
            aleatoric_scale=0.1,
        )
        flags = PosteriorFlags()
        seeds = [42]

        cond_model = baux.append(simple_model, x_whatif, y_whatif)
        draws, idx = baux.sample(
            cond_model, query, params=params, seeds=seeds, flags=flags
        )

        assert isinstance(draws, np.ndarray)
        assert draws.shape == (1, 1, 1)
        assert isinstance(idx, (list, np.ndarray))
