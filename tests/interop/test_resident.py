"""Device queries/results, snapshot ownership, and shared-seed ENN semantics."""

import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baux import ENN, capabilities
from baux.enn.enn_params import ENNParams, PosteriorFlags
from baux.turbo.config.surrogate import NeighborIndex


@pytest.fixture(params=["metal", "cuda"])
def device(request):
    name = request.param
    row = next(row for row in capabilities(probe=True) if row.name == name)
    if not row.available:
        pytest.skip(row.reason or f"{name} unavailable")
    return ("mps" if name == "metal" else "cuda"), NeighborIndex[name.upper()]


def models(device, variance=True):
    x = np.arange(48, dtype=float).reshape(16, 3) / 32
    y = np.column_stack((np.sin(x[:, 0]), np.cos(x[:, 1])))
    var = np.full_like(y, 0.01) if variance else None
    return ENN(x, y, var), ENN(x, y, var, index_driver=device[1])


def queries(device):
    return torch.tensor(
        [[0.125, 0.25, 0.375], [0.5, 0.625, 0.75], [0.125, 0.25, 0.375]],
        device=device[0],
    )


@pytest.mark.parametrize(
    ("noise", "variance"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_moments(device, noise, variance):
    cpu, native = models(device, variance)
    query = queries(device)
    params = ENNParams(3, 0.7, 0.13)
    flags = PosteriorFlags(observation_noise=noise)
    expected = cpu.posterior(
        query.cpu().numpy().astype(float), params=params, flags=flags
    )
    actual = native.device_posterior(query, params, flags=flags)
    mean, se = torch.from_dlpack(actual.mean), torch.from_dlpack(actual.se)
    assert mean.device == query.device and se.device == query.device
    np.testing.assert_allclose(mean.cpu().numpy(), expected.mu, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(se.cpu().numpy(), expected.se, rtol=3e-5, atol=3e-6)


def test_lifetime(device):
    cpu, native = models(device)
    query = queries(device)
    params = ENNParams(3, 0.7, 0.13)
    expected, _ = cpu.posterior_draw(
        query.cpu().numpy().astype(float), params, function_seeds=[41, 43]
    )
    snapshot = native.device_posterior(query, params)
    draws = torch.from_dlpack(snapshot.draw([41, 43])).reshape(2, 3, 2)
    mean = torch.from_dlpack(snapshot.mean)
    before = mean.clone()
    native.device_posterior(query + 0.25, params)
    native.add(np.ones((1, 3)), np.zeros((1, 2)), np.full((1, 2), 0.01))
    del snapshot, native, query
    gc.collect()
    assert torch.equal(mean, before)
    assert torch.equal(draws[:, 0], draws[:, 2])
    np.testing.assert_allclose(
        draws.cpu().numpy(), np.moveaxis(expected, -1, 0), rtol=5e-5, atol=5e-6
    )


@pytest.mark.parametrize("exclude", [False, True])
def test_offset(device, exclude):
    cpu, native = models(device)
    query = queries(device)
    # Contiguous views can still start inside the producer's allocation.
    query = torch.cat([query[:1], query])[1:]
    params = ENNParams(3, 0.7, 0.13)
    flags = PosteriorFlags(exclude_nearest=exclude)
    expected = cpu.posterior(
        query.cpu().numpy().astype(float), params=params, flags=flags
    )
    if device[0] == "cuda":
        with torch.cuda.stream(torch.cuda.Stream()):
            query = query * 1.0
            actual = native.device_posterior(query, params, flags=flags)
    else:
        actual = native.device_posterior(query, params, flags=flags)
    np.testing.assert_allclose(
        torch.from_dlpack(actual.mean).cpu().numpy(), expected.mu, rtol=3e-5, atol=3e-6
    )


def test_botorch(device, monkeypatch):
    pytest.importorskip("botorch")
    from baux.botorch import Model, Sampler

    cpu, native = models(device)
    query = queries(device).reshape(1, 3, 3)
    params = ENNParams(3, 0.7, 0.13)
    expected = Model(cpu, params).posterior(query.cpu())
    sampler = Sampler(torch.Size([2, 3]), seed=41)
    reference = sampler(expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("device posterior transferred an array through CPU")

    transfer = torch.Tensor.to

    def checked(tensor, *args, **kwargs):
        result = transfer(tensor, *args, **kwargs)
        if tensor.device.type != "cpu" and result.device.type == "cpu":
            forbidden()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr(torch.Tensor, "to", checked)
        actual = Model(native, params).posterior(query, output_indices=[1, 0])
        draws = sampler(actual)
        assert actual.mean.device == query.device and draws.device == query.device
    torch.testing.assert_close(
        actual.mean.cpu(), expected.mean[..., [1, 0]], rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        draws.cpu(), reference[..., [1, 0]], rtol=5e-5, atol=5e-6
    )


def test_rejections(device):
    _, native = models(device)
    params = ENNParams(3, 0.7, 0.13)
    query = queries(device)
    with pytest.raises((ValueError, BufferError)):
        native.device_posterior(query[:, :2], params)
    with pytest.raises((ValueError, BufferError)):
        native.device_posterior(query.to(torch.float16), params)
    array = native.device_posterior(query, params).mean
    with pytest.raises(BufferError):
        array.__dlpack__(copy=True)
    # Independent capsules retain the same read-only native allocation.
    left, right = torch.from_dlpack(array), torch.from_dlpack(array)
    assert left.data_ptr() == right.data_ptr()


def _jaxdevice(device):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", device[0] != "mps")
    platform = "MPS" if device[0] == "mps" else "cuda"
    platforms = {"mps", "metal"} if device[0] == "mps" else {"gpu", "cuda"}
    devices = [d for d in jax.devices() if d.platform.lower() in platforms]
    if not devices:
        if device[0] != "mps":
            pytest.fail("CUDA verification requires a JAX GPU runtime")
        pytest.skip(f"JAX {platform} runtime unavailable")
    return jax, devices[0]


def _jaxoutputs(device, native, query, query_host, params, jax_dev):
    import jax
    import jax.numpy as jnp

    if device[0] == "mps":
        with pytest.raises(ValueError, match="JAX Metal arrays are not supported"):
            native.device_posterior(query, params)
        actual = native.posterior(query_host, params=params)
        mean = jax.device_put(jnp.asarray(actual.mu, dtype=jnp.float32), jax_dev)
        draws_arr, _ = native.posterior_draw(
            query_host, params, function_seeds=[41, 43]
        )
        draws = jax.device_put(
            jnp.asarray(np.moveaxis(draws_arr, -1, 0), dtype=jnp.float32), jax_dev
        )
        return mean, draws
    snapshot = native.device_posterior(query, params)
    mean = jax.dlpack.from_dlpack(snapshot.mean, copy=False)
    draws = jax.dlpack.from_dlpack(snapshot.draw([41, 43]), copy=False).reshape(2, 2, 2)
    return mean, draws


def test_jax(device):
    jax, jax_dev = _jaxdevice(device)
    import jax.numpy as jnp

    cpu, native = models(device)
    query = jax.device_put(
        jnp.array([[0.125, 0.25, 0.375], [0.5, 0.625, 0.75]], dtype=jnp.float32),
        jax_dev,
    )
    params = ENNParams(3, 0.7, 0.13)
    query_host = np.asarray(jax.device_get(query), dtype=float)
    expected = cpu.posterior(query_host, params=params)
    reference, _ = cpu.posterior_draw(query_host, params, function_seeds=[41, 43])
    mean, draws = _jaxoutputs(device, native, query, query_host, params, jax_dev)
    np.testing.assert_allclose(jax.device_get(mean), expected.mu, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(
        jax.device_get(draws), np.moveaxis(reference, -1, 0), rtol=5e-5, atol=5e-6
    )
    assert next(iter(mean.devices())).id == 0
    del native
    gc.collect()
    assert jax.jit(lambda x: x + 1)(draws).shape == draws.shape


def test_torax():
    """Verify TORAX bridge and WPO Table 6 across TORAX 2.0.0, NumPy, PyTorch, JAX, and Rust/Metal.

    References: Pfau et al., arXiv:2505.00663; Tracey et al., arXiv:2307.11546;
    Citrin et al., arXiv:2406.06718.
    """
    import json
    from pathlib import Path

    from baux import verify_upstream_lock

    from evals.torax.tcv_coupling import ToraxTcvBridge, eval_wpo_table6

    torax = pytest.importorskip("torax")
    lock = json.loads(
        verify_upstream_lock(
            "torax", override_dir=str(Path(torax.__file__).resolve().parent)
        )
    )
    assert "merkle_root_blake3" in lock["merkle"]

    bridge = ToraxTcvBridge()
    zeros = np.zeros(160531, dtype=np.float64)
    score = bridge.eval_policy(zeros, steps=1)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0

    zero_np = eval_wpo_table6(zeros, backend="numpy")
    zero_pt = eval_wpo_table6(zeros, backend="torch")
    zero_jx = eval_wpo_table6(zeros, backend="jax")
    assert 0.0 <= zero_np <= 100.0
    np.testing.assert_allclose(zero_np, 22.101235, atol=1e-5)
    np.testing.assert_allclose(zero_np, zero_pt, atol=1e-4)
    np.testing.assert_allclose(zero_np, zero_jx, atol=1e-4)

    rng = np.random.default_rng(42)
    seeded = rng.uniform(-0.08, 0.08, size=160531)
    seed_np = eval_wpo_table6(seeded, backend="numpy")
    seed_pt = eval_wpo_table6(seeded, backend="torch")
    seed_jx = eval_wpo_table6(seeded, backend="jax")
    assert abs(seed_np - zero_np) > 1e-3
    np.testing.assert_allclose(seed_np, seed_pt, atol=1e-4)
    np.testing.assert_allclose(seed_np, seed_jx, atol=1e-4)


def test_tokagym():
    """Verify upstream TokaGym provenance lock and closed-loop TCV rollout.

    References: Theiler et al., doi:10.1088/1741-4326/ae6d12; Wang et al.,
    doi:10.1038/s41467-025-63917-3; Zenodo doi:10.5281/zenodo.16621120.
    """
    import json
    from pathlib import Path

    upstream = (
        Path(__file__).resolve().parents[2] / ".cache/baux/upstream/tokagym/TokaGym"
    )
    if not upstream.is_dir():
        pytest.skip(f"upstream directory missing: {upstream}")

    from baux import verify_upstream_lock

    from evals.tokagym.tokagym_rollout import (
        ACTION_DIM,
        OBS_DIM,
        TOKAGYM_STEPS,
        TOTAL_PARAMS,
        TokaGymClosedLoopEnv,
    )

    lock = json.loads(verify_upstream_lock("tokagym"))
    assert lock["archive"]["doi"] == "10.5281/zenodo.16621120"
    assert "merkle_root_blake3" in lock["merkle"]

    env = TokaGymClosedLoopEnv.shared(scenario="170kA")
    zeros = np.zeros(TOTAL_PARAMS, dtype=np.float32)
    eps = np.zeros((TOKAGYM_STEPS, ACTION_DIM), dtype=np.float32)
    ret, obs, u, rewards = env.eval_traj(zeros, eps)
    assert np.isfinite(ret)
    assert obs.shape == (TOKAGYM_STEPS, OBS_DIM)
    assert u.shape == (TOKAGYM_STEPS, ACTION_DIM)
    assert rewards.shape == (TOKAGYM_STEPS,)
    np.testing.assert_allclose(ret, env.eval_return(zeros), rtol=1e-5, atol=1e-5)
