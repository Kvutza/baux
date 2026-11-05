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
    if device[0] == "mps":
        pytest.skip("the public JAX route targets CUDA; use Torch for Metal")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    platform = "METAL" if device[0] == "mps" else "cuda"
    platforms = {"metal"} if device[0] == "mps" else {"gpu", "cuda"}
    devices = [d for d in jax.devices() if d.platform.lower() in platforms]
    if not devices:
        if device[0] != "mps":
            pytest.fail("CUDA verification requires a JAX GPU runtime")
        pytest.skip(f"JAX {platform} runtime unavailable")
    return jax, devices[0]


def test_jax(device):
    jax, jax_dev = _jaxdevice(device)
    import jax.numpy as jnp

    cpu, native = models(device)
    query = jax.device_put(
        jnp.array([[0.125, 0.25, 0.375], [0.5, 0.625, 0.75]], dtype=jnp.float32),
        jax_dev,
    )
    params = ENNParams(3, 0.7, 0.13)
    snapshot = native.device_posterior(query, params)
    mean = jax.dlpack.from_dlpack(snapshot.mean, copy=False)
    draws = jax.dlpack.from_dlpack(snapshot.draw([41, 43]), copy=False).reshape(2, 2, 2)
    query_host = np.asarray(jax.device_get(query), dtype=float)
    expected = cpu.posterior(query_host, params=params)
    np.testing.assert_allclose(jax.device_get(mean), expected.mu, rtol=3e-5, atol=3e-6)
    reference, _ = cpu.posterior_draw(query_host, params, function_seeds=[41, 43])
    np.testing.assert_allclose(
        jax.device_get(draws), np.moveaxis(reference, -1, 0), rtol=5e-5, atol=5e-6
    )
    assert next(iter(mean.devices())).id == 0
    del snapshot, native
    gc.collect()
    assert jax.jit(lambda x: x + 1)(draws).shape == draws.shape
