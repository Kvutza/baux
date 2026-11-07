"""Tests for the functional Python frontend API (PyTrees, combinators, pure state)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import baux


@dataclass(frozen=True)
class LayerBlock:
    weight: np.ndarray
    bias: np.ndarray


def test_pytreespec():
    template = {
        "encoder": LayerBlock(
            weight=np.zeros((4, 3)),
            bias=np.zeros((4,)),
        ),
        "head": (2, 4),
    }
    spec = baux.pytree_bounds(template, low=-2.0, high=2.0)
    assert spec.dim == 4 * 3 + 4 + 2 * 4
    assert spec.bounds.shape == (24, 2)
    assert not spec.bounds.flags.writeable

    flat = np.arange(24, dtype=float)
    tree = baux.unflatten(flat, spec)
    assert isinstance(tree["encoder"], LayerBlock)
    assert tree["encoder"].weight.shape == (4, 3)
    assert tree["encoder"].bias.shape == (4,)
    assert tree["head"].shape == (2, 4)
    np.testing.assert_allclose(baux.ravel(tree, spec), flat)

    batch_flat = np.vstack([flat, flat + 10.0])
    trees = baux.unflatten_batch(batch_flat, spec)
    assert len(trees) == 2
    np.testing.assert_allclose(baux.ravel(trees, spec), batch_flat)

    stacked = baux.unflatten_stacked(batch_flat, spec)
    assert stacked["encoder"].weight.shape == (2, 4, 3)
    assert stacked["head"].shape == (2, 2, 4)
    np.testing.assert_allclose(baux.ravel(stacked, spec), batch_flat)


def test_mlpspec():
    spec = baux.mlp_spec([(8, 4), (2, 8)], low=-0.5, high=0.5)
    assert spec.dim == (8 * 4 + 8) + (2 * 8 + 2)
    flat = np.full(spec.dim, 0.1, dtype=float)
    params = spec.unflatten(flat)
    obs = np.ones(4, dtype=float)
    out = baux.mlp_forward(params, obs, act=np.tanh)
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_combinators():
    cfg1 = baux.chain(
        baux.lhd(6),
        baux.raasp(12, num_candidates=64),
        baux.enn(k=5, num_samples=32),
        baux.ucb(beta=1.5),
        baux.tr_bounds(length_init=0.6, length_min=0.05, length_max=1.2),
        baux.regions(3, sharing="ellipsoid_ball", allocation="least_trials"),
    )
    pipe = (
        baux.lhd(6)
        | baux.raasp(12, num_candidates=64)
        | baux.enn(k=5, num_samples=32)
        | baux.ucb(beta=1.5)
        | baux.tr_bounds(length_init=0.6, length_min=0.05, length_max=1.2)
        | baux.regions(3, sharing="ellipsoid_ball", allocation="least_trials")
    )
    cfg2 = pipe()
    assert cfg1 == cfg2
    assert cfg1.init.num_init == 6
    assert cfg1.candidates.num_pert == 12
    assert cfg1.trust_region.count == 3
    assert cfg1.trust_region.sharing == "ellipsoid_ball"

    cfg_zero = baux.chain(baux.sobol(32), baux.zero(), baux.random_acq())
    assert cfg_zero.candidates.num_candidates == 32

    cfg_pareto = baux.chain(
        baux.uniform_pert(32), baux.enn(k=8, num_samples=None), baux.pareto()
    )
    assert cfg_pareto.surrogate.k == 8

    cfg_th = baux.chain(baux.lhd(4), baux.gp(), baux.thompson())
    assert cfg_th.init.num_init == 4


def test_purebranching():
    spec = baux.mlp_spec([(4, 2), (1, 4)], low=-1.0, high=1.0)
    cfg = baux.chain(
        baux.lhd(4), baux.raasp(4), baux.enn(k=3, num_samples=16), baux.ucb()
    )
    s0 = baux.init(spec, cfg, seed=101)
    assert s0.num_observations == 0

    s1, b1 = baux.ask(s0, 4)
    s1_dup, b1_dup = baux.ask(s0, 4)
    np.testing.assert_array_equal(b1.points, b1_dup.points)
    assert len(b1.params) == 4
    assert b1.stacked[0]["weight"].shape == (4, 4, 2)

    y_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    y_b = np.array([40.0, 30.0, 20.0, 10.0], dtype=float)
    x_caller = b1.points.copy()
    assert x_caller.flags.writeable
    assert y_a.flags.writeable

    s2_a, mu_a = baux.tell(s1, x_caller, y_a)
    s2_b, mu_b = baux.tell(s1, b1.stacked, y_b)

    assert x_caller.flags.writeable
    assert y_a.flags.writeable
    assert s0.num_observations == 0
    assert s1.num_observations == 0
    assert s2_a.num_observations == 4
    assert s2_b.num_observations == 4
    assert not np.allclose(mu_a, mu_b)
    assert s2_a.incumbent_params is not None

    pred_a = baux.posterior_mu(s2_a, b1.params)
    pred_b = baux.posterior_mu(s2_b, b1.stacked)
    assert not np.allclose(pred_a, pred_b)


def test_multiregionpure():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    cfg = baux.chain(
        baux.lhd(2),
        baux.zero(),
        baux.regions(2, sharing="ellipsoid_ball"),
    )
    s0 = baux.init(bounds, cfg, seed=77)
    s1, warm = baux.ask(s0, 2)
    s2 = baux.complete(s1, warm, np.array([0.5, 1.5]))
    assert s2.pending_ids is None
    assert s2.num_observations == 2

    s3, reg_batch = baux.ask(s2, 2)
    s3_branch = baux.cancel(s3, [reg_batch.ids[0]])
    s3_done = baux.complete(s3_branch, [reg_batch.ids[1]], np.array([2.5]))
    assert s3_done.pending_ids is None
    assert s3.pending_ids is not None

    s4, abandoned = baux.restart_region(s3, 0, np.zeros(3))
    assert isinstance(abandoned, tuple)
    s5, _ = baux.import_observations(s4, np.zeros((1, 3)), np.array([3.0]))
    assert s5.num_observations == 3


def test_stepscan():
    spec = baux.pytree_bounds({"x": (2,), "y": (2,)}, low=-2.0, high=2.0)
    cfg = baux.chain(baux.lhd(2), baux.zero())
    s0 = baux.init(spec, cfg, seed=55)

    def obj(tree):
        x = tree["x"]
        y = tree["y"]
        return -np.sum(x**2 + y**2, axis=-1)

    s_final, hist = baux.scan(s0, obj, steps=5, num_arms=2, use_pytree=True)
    assert hist.shape == (5, 2)
    assert s_final.num_observations == 10
    assert s0.num_observations == 0


def test_functionalenn():
    rng = np.random.default_rng(19)
    x0 = rng.uniform(-1.0, 1.0, size=(8, 3))
    y0 = np.sum(x0**2, axis=1)
    st0 = baux.enn.init(x0, y0, scale_x=True)
    assert x0.flags.writeable
    assert y0.flags.writeable
    assert len(st0) == 8

    x1 = rng.uniform(-1.0, 1.0, size=(4, 3))
    y1 = np.sum(x1**2, axis=1)
    st1 = baux.enn.append(st0, x1, y1)
    assert len(st0) == 8
    assert len(st1) == 12

    params = baux.enn.fit(st1, k=4, num_samples=16, seed=29)
    post = baux.enn.posterior(st1, x1, params)
    assert post.mu.shape == (4, 1)
    draws, idx = baux.enn.sample(st1, x1, params, seeds=[1, 2, 3])
    assert draws.shape == (4, 1, 3)
    assert idx.shape[0] == 4


def test_noquantization():
    import baux.experimental as exp

    for name in ("quantize_int4", "quantize_e2m1", "weight_int4_select_ucb"):
        assert not hasattr(baux, name)
        assert not hasattr(exp, name)
        with pytest.raises(AttributeError):
            getattr(baux, name)
