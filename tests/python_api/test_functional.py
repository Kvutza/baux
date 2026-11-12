"""Tests for the functional Python frontend API (PyTrees, combinators, pure state)."""

from __future__ import annotations

from dataclasses import dataclass

import baux
import numpy as np
import pytest


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
    s0 = baux.init(spec, cfg)
    assert s0.digest() == baux.init(spec, cfg).digest()
    assert s0.num_observations == 0 and not hasattr(s0, "region_count")

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
    s0 = baux.init(bounds, cfg)
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
    s0 = baux.init(spec, cfg)

    def obj(tree):
        x = tree["x"]
        y = tree["y"]
        return -np.sum(x**2 + y**2, axis=-1)

    s_final, hist = baux.scan(s0, obj, steps=5, num_arms=2)
    assert hist.shape == (5, 2)
    assert s_final.num_observations == 10
    assert s0.num_observations == 0

    s_keyed, k_next, traces = baux.scan(
        s0, s0.key, 4, lambda x: -np.sum(x**2, axis=-1), num_arms=2
    )
    assert len(traces) == 4 and k_next != s0.key
    assert traces[-1]["step"] == 3 and traces[-1]["digest"] == s_keyed.digest()
    assert traces[-1]["best_y"] == s_keyed.best_y


def test_pmap():
    cfg = baux.chain(baux.sobol(4), baux.enn(k=3, num_samples=8), baux.ucb())
    root = baux.init([[-2.0, 2.0], [-2.0, 2.0]], cfg)

    def run_branch(idx: int, st: baux.OptimizerState, bk: baux.Key):
        shift = 0.25 * float(idx)
        fin, nk, tr = baux.scan(
            st, bk, 3, lambda _, x: -np.sum((x - shift) ** 2, axis=1), num_arms=2
        )
        return fin.digest(), nk, tr[-1]["best_y"]

    out_a = baux.pmap(root, root.key, 4, run_branch)
    out_b = baux.pmap(root, root.key, 4, run_branch)
    assert out_a == out_b and len({d for d, _, _ in out_a}) == 4


def test_functionalenn():
    sp = baux.Space({"x": (3,)}, low=-1.0, high=1.0)
    rng = np.random.default_rng(sp.key.seed())
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

    xq = rng.uniform(-1.0, 1.0, size=(4, 3))
    params = baux.enn.fit(st1, k=4, num_samples=16)
    post = baux.enn.posterior(st1, xq, params)
    assert post.mu.shape == (4, 1)
    st_hyp = baux.with_geometry(st1, "hyperbolic", local_k=3)
    post_hyp = baux.enn.posterior(st_hyp, xq, params)
    assert post_hyp.mu.shape == (4, 1) and not np.allclose(post.mu, post_hyp.mu)
    draws, idx = baux.enn.sample(st1, xq, params, num_draws=3)
    assert draws.shape == (4, 1, 3)
    assert idx.shape[0] == 4


def test_noquantization():
    import baux.experimental as exp

    for name in ("quantize_int4", "quantize_e2m1", "weight_int4_select_ucb"):
        assert not hasattr(baux, name)
        assert not hasattr(exp, name)
        with pytest.raises(AttributeError):
            getattr(baux, name)


def test_key():
    sp_a = baux.Space({"x": (2,)}, low=-1.0, high=1.0)
    sp_b = baux.Space({"x": (3,)}, low=-1.0, high=1.0)
    k0 = baux.key(sp_a)
    assert k0 == sp_a.key and k0 != sp_b.key
    assert baux.key() != baux.key()
    assert repr(k0).startswith("Key(0x")
    a1, b1 = baux.split(k0)
    assert (a1, b1) == baux.split(k0) and a1 != b1
    quad = baux.split(k0, 4)
    assert len(quad) == 4 and quad == baux.split(k0, 4)
    assert baux.fold(k0, 7) == baux.fold(k0, 7) != baux.fold(k0, 8)
    assert baux.Key.from_pair(k0.hi, k0.lo) == k0


def test_categorical():
    cat = baux.categorical(["adam", "muon", "sgd"], separation=3.0)
    spec = baux.pytree_bounds(
        {"lr": 0.01, "opt": cat, "w": np.zeros(2)}, low=-1.0, high=1.0
    )
    assert spec.dim == 6 and spec.tied_dims == ((1, 2, 3),)
    vec = baux.ravel({"lr": 0.5, "opt": "muon", "w": np.array([0.2, -0.3])}, spec)
    assert np.allclose(vec[1:4], [0.0, 1.0, 0.0])
    restored = baux.unflatten(vec, spec)
    assert restored["opt"] == "muon" and np.allclose(restored["w"], [0.2, -0.3])


def test_stiefel():
    raw_l = np.array([[1.0], [0.9], [-0.8]])
    rng = np.random.default_rng(baux.key(raw_l).seed())
    x = rng.uniform(-1.0, 1.0, size=(12, 3))
    y = np.column_stack([x[:, 0], x[:, 0] + 0.1 * x[:, 1], -x[:, 0]])
    st = baux.Stiefel.from_cov(y, rank=1)
    aff = baux.Affine(np.zeros(3), np.array([1.2, 1.0, 0.8]), np.array([0.5, 0.5, 0.5]))
    s_plain = baux.enn_init(x, y)
    s_mod = baux.calibrate(baux.stein(s_plain, st), aff)
    s_auto = baux.calibrate(baux.stein(s_plain, rank=1), alpha=0.1)
    p = baux.ENNParams(k_neighbors=4, epistemic_scale=1.0, aleatoric_scale=0.1)
    xq = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, -0.1]])
    out_plain = baux.posterior(s_plain, xq, p)
    out_mod = baux.posterior(s_mod, xq, p)
    out_auto = baux.posterior(s_auto, xq, p)
    d_plain, _ = baux.sample(s_plain, xq, p, seeds=[7, 8])
    d_mod, _ = baux.sample(s_mod, xq, p, seeds=[7, 8])
    assert not np.allclose(out_plain.mu, out_mod.mu)
    assert not np.allclose(out_plain.mu, out_auto.mu)
    assert not np.allclose(d_plain, d_mod)


def test_fork():
    cfg = baux.chain(baux.sobol(4), baux.enn(k=3), baux.ucb())
    s0 = baux.init([[-1.0, 1.0], [-1.0, 1.0]], cfg)
    s1, b1 = baux.ask(s0, 4)
    s2, _ = baux.tell(s1, b1, -np.sum(b1.points**2, axis=1))
    object.__setattr__(s2, "_log", ("corrupted",))
    branch_a, ba = baux.ask(s2, 2)
    branch_b, bb = baux.ask(s2, 2)
    assert np.allclose(ba.points, bb.points) and baux.digest(branch_a) == baux.digest(
        branch_b
    )
    sa_done, _ = baux.tell(branch_a, ba, -np.sum(ba.points**2, axis=1))
    sb_done, _ = baux.tell(branch_b, bb, -np.sum(bb.points**2, axis=1) - 10.0)
    assert baux.digest(sa_done) != baux.digest(sb_done) and s2.num_observations == 4


def test_freeze():
    cfg = baux.chain(
        baux.lhd(8),
        baux.sobol(64),
        baux.enn(k=5, scale_x=True),
        baux.ucb(),
    )
    bounds = np.tile([-3.0, 3.0], (40, 1))
    s0 = baux.init(bounds, cfg, key=baux.key(42))

    s = s0
    for step in range(10):
        s, batch = baux.ask(s, 8)
        norm = np.sum(batch.points**2, axis=1)
        y = (100.0 - 1e-3 * norm) if step == 0 else (-50.0 * float(step) - 1e-3 * norm)
        yvar = 0.01 + 0.002 * np.arange(8, dtype=float)
        s, _ = baux.tell(s, batch, y, y_var=yvar)

    assert s.num_observations == 80
    assert s.epoch == 20
    assert abs(s.tr_length - 0.4) < 1e-12

    blob = baux.freeze(s)
    s_restored = baux.thaw(s0, blob)
    assert s_restored.num_observations == 80
    assert s_restored.epoch == s.epoch == 20
    assert s_restored.seed == s.seed
    assert s_restored.tr_length == s.tr_length
    assert baux.digest(s_restored) == baux.digest(s)
    assert baux.freeze(s_restored) == blob
    assert np.array_equal(s_restored.x_obs, s.x_obs)
    assert np.array_equal(s_restored.y_obs, s.y_obs)

    xq = np.linspace(-1.0, 1.0, 80).reshape(2, 40)
    assert np.array_equal(baux.posterior_mu(s_restored, xq), baux.posterior_mu(s, xq))

    s_after_live = s
    s_after_rest = s_restored
    for _ in range(2):
        s_next_live, b_live = baux.ask(s_after_live, 8)
        s_next_rest, b_rest = baux.ask(s_after_rest, 8)
        assert np.array_equal(b_live.points, b_rest.points)
        assert b_live.ids == b_rest.ids
        assert baux.digest(s_next_live) == baux.digest(s_next_rest)
        y_fail = -500.0 - np.sum(b_live.points**2, axis=1)
        yv_fail = 0.02 + 0.001 * np.arange(8, dtype=float)
        s_after_live, _ = baux.tell(s_next_live, b_live, y_fail, y_var=yv_fail)
        s_after_rest, _ = baux.tell(s_next_rest, b_rest, y_fail, y_var=yv_fail)
        assert s_after_rest.tr_length == s_after_live.tr_length
        assert baux.digest(s_after_rest) == baux.digest(s_after_live)
    assert abs(s_after_rest.tr_length - 0.2) < 1e-12

    # Verify y_var requirement is preserved on restored state
    with pytest.raises(ValueError, match="y_var"):
        baux.tell(s_restored, b_rest, y_fail)

    # Forked seed + scan/pmap rollout commutativity
    s_clean, _ = baux.scan(
        s0,
        baux.key(19),
        steps=6,
        eval_fn=lambda x: -np.sum((x - 0.25) ** 2, axis=1),
        num_arms=4,
    )[:2]
    s_fork = s_clean.fork(baux.key(777))
    assert baux.digest(s_fork) != baux.digest(s_clean)
    s_fork_rest = baux.thaw(s0, baux.freeze(s_fork))
    assert baux.digest(s_fork_rest) == baux.digest(s_fork)

    def fn(x):
        return -np.sum((x - 0.25) ** 2, axis=1)

    roll_live = baux.pmap(
        s_fork,
        baux.key(99),
        branches=4,
        branch_fn=lambda i, st, k: baux.digest(
            baux.scan(st, k, steps=3, eval_fn=fn, num_arms=2)[0]
        ),
    )
    roll_rest = baux.pmap(
        s_fork_rest,
        baux.key(99),
        branches=4,
        branch_fn=lambda i, st, k: baux.digest(
            baux.scan(st, k, steps=3, eval_fn=fn, num_arms=2)[0]
        ),
    )
    assert roll_live == roll_rest and len(set(roll_live)) == 4


def test_freezeportfolio():
    bounds = np.tile([-3.0, 3.0], (4, 1))
    pipe = (
        baux.lhd(6)
        | baux.enn(k=4)
        | baux.thompson()
        | baux.regions(count=4, sharing="ellipsoid_ball")
    )
    s0 = baux.init(bounds, pipe, key=baux.key(101))

    def fn(x):
        return -np.sum((x - 0.35) ** 2, axis=1)

    s_warm, b_init = baux.ask(s0, num_arms=6)
    s_warm = baux.complete(s_warm, b_init.ids, fn(b_init.points))

    s_live, b1 = baux.ask(s_warm, num_arms=2)
    s_live, b2 = baux.ask(s_live, num_arms=2)
    s_live = baux.complete(s_live, [b2.ids[0]], fn(b2.points[:1]))
    s_live = baux.cancel(s_live, [b2.ids[1]])

    # Checkpoint while b1 is STILL in flight
    assert s_live.pending_ids is not None and len(s_live.pending_ids) == 2
    blob = baux.freeze(s_live)
    s_rest = baux.thaw(s0, blob)
    assert s_rest.pending_ids == s_live.pending_ids
    assert s_rest.epoch == s_live.epoch
    assert baux.digest(s_rest) == baux.digest(s_live)
    assert baux.freeze(s_rest) == blob

    # Resolve in-flight b1 on both live and restored states
    s_live_done = baux.complete(s_live, b1.ids, fn(b1.points))
    s_rest_done = baux.complete(s_rest, b1.ids, fn(b1.points))
    assert s_rest_done.pending_ids is None
    assert baux.digest(s_rest_done) == baux.digest(s_live_done)

    s_live_next, bn_live = baux.ask(s_live_done, num_arms=2)
    s_rest_next, bn_rest = baux.ask(s_rest_done, num_arms=2)
    assert np.array_equal(bn_live.points, bn_rest.points)
    assert baux.digest(s_live_next) == baux.digest(s_rest_next)

    # Gate 3 adversarial inputs across FFI
    import pytest

    with pytest.raises(TypeError):
        baux.thaw(s0, 12345)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        baux.thaw(s0, b"")
    with pytest.raises(ValueError):
        baux.thaw(s0, blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    s_other_bounds = baux.init(np.tile([-1.0, 1.0], (4, 1)), pipe, key=baux.key(101))
    with pytest.raises(ValueError):
        baux.thaw(s_other_bounds, blob)
    s_single = baux.init(bounds, baux.lhd(6) | baux.enn(k=4), key=baux.key(101))
    with pytest.raises(ValueError):
        baux.thaw(s_single, blob)


def test_manifolds():
    from evals.geometry.eval_manifolds import run_all_evals

    res = run_all_evals()
    hyp, st, cal, ch, vm = (
        res["hyperbolic_knn"],
        res["stiefel_stein"],
        res["wls_affine"],
        res["manifold_embeddings"],
        res["parallel_pmap"],
    )
    assert (
        hyp["rmse_conformal"] < hyp["rmse_euclidean"]
        and hyp["rmse_hyperbolic"] < hyp["rmse_euclidean"]
        and hyp["nll_conformal"] < hyp["nll_euclidean"]
        and hyp["nll_hyperbolic"] < hyp["nll_euclidean"]
    )
    assert (
        st["mse_stein"] < st["mse_raw"]
        and st["stiefel_err"] < 1e-12
        and cal["slope_err_wls"] < cal["slope_err_ols"]
        and cal["empirical_coverage"] >= cal["target_coverage"] - 0.025
    )
    assert (
        ch["s1_wrap_dist"] < 0.05 * ch["s1_opp_dist"]
        and ch["hellinger_err"] < 1e-14
        and ch["simplex_sum_err"] < 1e-14
        and ch["log_mid_err"] < 1e-14
        and ch["mahalanobis_match_rate"] == 1.0
    )
    assert (
        vm["digest_match_rate"] == 1.0
        and vm["unique_branch_ratio"] == 1.0
        and vm["steps_per_sec"] > 0.0
        and np.isfinite(vm["mean_best_y"])
    )
    sp = baux.Space(
        {"th": baux.Periodic(), "p": baux.Simplex(3), "lr": baux.Log(1e-4, 1.0)}
    )
    assert sp.tied_dims == ((0, 1), (2, 3, 4))


def test_bogeometry():
    sp = baux.Space(
        {
            "opt": baux.Categorical(["adam", "muon", "sgd"]),
            "th": baux.Periodic(),
            "lr": baux.Log(1e-4, 1.0),
        }
    )
    aff = baux.Affine([0.1], [1.2], [0.5])
    cfg_euc = baux.chain(
        baux.lhd(4),
        baux.raasp(4),
        baux.enn(k=3, num_samples=8, metric_learning="auto"),
        baux.ucb(),
    )
    cfg_hyp = baux.chain(
        baux.lhd(4),
        baux.raasp(4),
        baux.enn(
            k=3,
            num_samples=8,
            metric_learning="auto",
            geometry="hyperbolic",
            local_k=2,
        ),
        baux.calibrate(aff),
        baux.ucb(),
    )
    s_euc = baux.init(sp, cfg_euc)
    s_hyp = baux.init(sp, cfg_hyp)
    s_zero = baux.init(sp, baux.chain(baux.raasp(4), baux.zero()))
    s_conf = baux.with_geometry(s_euc, "conformal", local_k=2)
    assert s_euc.config.surrogate.tied_dims == sp.tied_dims
    assert s_zero.config.candidates.tied_dims == sp.tied_dims
    assert s_conf.config.surrogate.geometry == "conformal"

    s1_e, b_e = baux.ask(s_euc, 4)
    y_warm = -np.sum(b_e.points**2, axis=1)
    s2_e, mu_e = baux.tell(s1_e, b_e, y_warm)
    s1_h, _ = baux.ask(s_hyp, 4)
    s2_h, mu_h = baux.tell(s1_h, b_e.points, y_warm)
    assert not np.allclose(mu_e, mu_h)
    assert s2_e.lengthscales is not None and s2_e.metric_weights is not None
    assert len(s2_e._log) == 0

    s3_e, b3_e = baux.ask(s2_e, 2)
    assert b3_e.weights is not None
    assert b3_e.params[0]["opt"] in ("adam", "muon", "sgd")
    assert s2_e.incumbent_params["opt"] in ("adam", "muon", "sgd")

    cfg_online = baux.chain(
        baux.lhd(6),
        baux.enn(k=3, num_samples=6, metric_learning="auto"),
        baux.stein(rank=1),
        baux.calibrate(alpha=0.1),
        baux.thompson(),
    )
    s_on, b_on = baux.ask(baux.init(sp, cfg_online), 10)
    y_on = -10.0 * b_on.points[:, -1] ** 2 - 0.01 * np.sum(
        b_on.points[:, :-1] ** 2, axis=1
    )
    s_on2, mu_on = baux.tell(s_on, b_on, y_on)
    assert (
        np.all(np.isfinite(mu_on))
        and s_on2.metric_weights[-1] > s_on2.metric_weights[0]
    )


test_manifold_evals = test_manifolds
