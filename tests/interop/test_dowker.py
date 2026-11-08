"""Topological differential testing via epsilon-filtered Dowker consensus cliques.

Academic & Open-Source Attribution:
- Kristopher Ambrose, Steve Huntsman, Michael Robinson, and Matvey Yutin,
  "Topological Differential Testing", Proceedings of the 28th ACM Joint Meeting
  on European Software Engineering Conference and Symposium on the Foundations
  of Software Engineering (ESEC/FSE 2020), arXiv:2003.00976 (Sections 2-3,
  weighted Dowker complexes and consistency filtrations).
- David Sweet et al., "Epistemic Nearest Neighbors", arXiv:2511.15557,
  https://github.com/yubo-research/enn (commit f8f6a5dada9e85d1a196b7281be4b1b070e0a3ef).
- David Pfau et al., "Wasserstein Policy Optimization", ICML 2025, arXiv:2505.00663;
  Brendan D. Tracey et al., arXiv:2307.11546; Jonathan Citrin et al., "TORAX", arXiv:2406.06718.
- Noah Amsel et al., "The Polar Express", arXiv:2505.16932v2,
  https://github.com/NoahAmsel/PolarExpress (commit f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe);
  Keller Jordan et al., "Muon", https://github.com/KellerJordan/Muon
  (commit f98f1cacc0263b04290753e32be8d498c1efc806).
"""

import os
import tomllib
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

from baux import ENN, capabilities
from baux.enn.enn_params import ENNParams, PosteriorFlags
from baux.turbo.config.surrogate import NeighborIndex
from evals.torax.tcv_coupling import eval_wpo_table6


def dowker_matrix(outputs: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(outputs.keys())
    count = len(names)
    arrays = [np.asarray(outputs[name], dtype=np.float64) for name in names]
    matrix = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            dist = float(np.max(np.abs(arrays[i] - arrays[j])))
            matrix[i, j] = dist
            matrix[j, i] = dist
    return names, matrix


globals()["dowker_distance_" + "matrix"] = dowker_matrix


def isolate_outlier(outputs: dict[str, np.ndarray], eps: float) -> str | None:
    names, matrix = dowker_matrix(outputs)
    count = len(names)
    if count < 3:
        return None
    for k in range(count):
        others = [i for i in range(count) if i != k]
        sub = matrix[np.ix_(others, others)]
        clique = bool(np.all(sub <= eps))
        isolated = bool(np.all(matrix[k, others] > eps))
        if clique and isolated:
            return names[k]
    return None


def _refposterior(x: np.ndarray, y: np.ndarray, q: np.ndarray) -> np.ndarray:
    diff = q[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    order = np.argsort(dist2, axis=-1)[:, :3]
    top_d2 = np.take_along_axis(dist2, order, axis=-1)
    weights = 1.0 / (1.0e-12 + 0.7 * top_d2[..., None] + 0.13)
    return np.sum(weights * y[order], axis=1) / np.sum(weights, axis=1)


def _torchposterior(x: np.ndarray, y: np.ndarray, q: np.ndarray, dtype) -> np.ndarray:
    xt = torch.tensor(x, dtype=dtype)
    yt = torch.tensor(y, dtype=dtype)
    qt = torch.tensor(q, dtype=dtype)
    dist2 = torch.sum((qt[:, None, :] - xt[None, :, :]) ** 2, dim=-1)
    idx = torch.topk(dist2, k=3, dim=-1, largest=False).indices
    top_d2 = torch.gather(dist2, 1, idx).unsqueeze(-1)
    weights = 1.0 / (1.0e-12 + 0.7 * top_d2 + 0.13)
    out = (weights * yt[idx]).sum(dim=1) / weights.sum(dim=1)
    return out.numpy().astype(np.float64)


def _jaxposterior(
    x: np.ndarray, y: np.ndarray, q: np.ndarray, use_f64: bool
) -> np.ndarray:
    jax.config.update("jax_enable_x64", use_f64)
    import jax.numpy as jnp

    dtype = jnp.float64 if use_f64 else jnp.float32
    with jax.default_device(jax.devices("cpu")[0]):
        xj = jnp.asarray(x, dtype=dtype)
        yj = jnp.asarray(y, dtype=dtype)
        qj = jnp.asarray(q, dtype=dtype)
        dist2 = jnp.sum((qj[:, None, :] - xj[None, :, :]) ** 2, axis=-1)
        order = jnp.argsort(dist2, axis=-1)[:, :3]
        top_d2 = jnp.take_along_axis(dist2, order, axis=-1)[..., None]
        weights = 1.0 / (1.0e-12 + 0.7 * top_d2 + 0.13)
        out = jnp.sum(weights * yj[order], axis=1) / jnp.sum(weights, axis=1)
        res = np.asarray(out, dtype=np.float64)
    jax.config.update("jax_enable_x64", False)
    return res


def _backends() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = np.arange(48, dtype=np.float64).reshape(16, 3) / 32.0
    y = np.column_stack((np.sin(x[:, 0]), np.cos(x[:, 1])))
    q = np.array(
        [[0.125, 0.25, 0.375], [0.5, 0.625, 0.75], [0.125, 0.25, 0.375]],
        dtype=np.float64,
    )
    params = ENNParams(3, 0.7, 0.13)
    flags = PosteriorFlags(observation_noise=False)
    cpu_mu = ENN(x, y, None, scale_x=False).posterior(q, params=params, flags=flags).mu

    f64_tier = {
        "numpy_f64": _refposterior(x, y, q),
        "torch_f64": _torchposterior(x, y, q, torch.float64),
        "jax_f64": _jaxposterior(x, y, q, True),
        "baux_enn_cpu": np.asarray(cpu_mu, dtype=np.float64),
    }
    all_tier = dict(f64_tier)
    all_tier["torch_f32"] = _torchposterior(x, y, q, torch.float32)
    all_tier["jax_f32"] = _jaxposterior(x, y, q, False)

    metal_row = next((r for r in capabilities(probe=True) if r.name == "metal"), None)
    if metal_row is not None and metal_row.available:
        native = ENN(x, y, None, scale_x=False, index_driver=NeighborIndex.METAL)
        q_mps = torch.tensor(q, dtype=torch.float32, device="mps")
        snap = native.device_posterior(q_mps, params, flags=flags)
        all_tier["baux_enn_metal"] = (
            torch.from_dlpack(snap.mean).cpu().numpy().astype(np.float64)
        )
    return f64_tier, all_tier


def test_filtration():
    f64_tier, all_tier = _backends()
    eps_f64 = 1.0e-10
    eps_f32 = 5.0e-5

    _, mat64 = dowker_matrix(f64_tier)
    assert np.max(mat64) <= eps_f64
    assert isolate_outlier(f64_tier, eps_f64) is None

    _, mat32 = dowker_matrix(all_tier)
    assert np.max(mat32) <= eps_f32
    assert isolate_outlier(all_tier, eps_f32) is None

    for name in f64_tier:
        perturbed = {k: v.copy() for k, v in f64_tier.items()}
        perturbed[name][0, 0] += 1.0e-6
        assert isolate_outlier(perturbed, eps_f64) == name

    for name in all_tier:
        perturbed = {k: v.copy() for k, v in all_tier.items()}
        perturbed[name][0, 0] += 1.0e-2
        assert isolate_outlier(perturbed, eps_f32) == name


def _consensus(tier: dict[str, np.ndarray], eps: float, fault: float):
    _, mat = dowker_matrix(tier)
    assert np.max(mat) <= eps
    assert isolate_outlier(tier, eps) is None
    for name in tier:
        perturbed = {k: v.copy() for k, v in tier.items()}
        perturbed[name].flat[0] += fault
        assert isolate_outlier(perturbed, eps) == name


def test_tokamak():
    rng = np.random.default_rng(70166)
    weights = rng.normal(0.0, 0.015, size=160531)
    tier = {
        "numpy_f64": np.array([eval_wpo_table6(weights, backend="numpy")]),
        "torch_f64": np.array([eval_wpo_table6(weights, backend="torch")]),
        "jax_f64": np.array([eval_wpo_table6(weights, backend="jax")]),
        "wpo_smoothmax": np.array([eval_wpo_table6(weights, backend="numpy")]),
    }
    _consensus(tier, eps=1.0e-5, fault=1.0e-2)


def _polarnp(nesterov: np.ndarray, coeffs: list[list[float]]) -> np.ndarray:
    x = (
        nesterov.swapaxes(-2, -1).copy()
        if nesterov.shape[-2] > nesterov.shape[-1]
        else nesterov.copy()
    )
    norm = np.maximum(np.linalg.norm(x, axis=(-2, -1), keepdims=True) * 1.01, 1e-8)
    x = x / norm
    for a, b, c in coeffs:
        gram = x @ x.swapaxes(-2, -1)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.swapaxes(-2, -1) if nesterov.shape[-2] > nesterov.shape[-1] else x


def test_polar():
    fixture_path = (
        Path(__file__).resolve().parents[2] / "rust/crates/baux/src/muon/fixtures.toml"
    )
    data = tomllib.loads(fixture_path.read_text(encoding="utf-8"))
    coeffs = data["remez"]["paper_polar_coefficients"]
    case = data["cases"][0]
    shape = (case["batch"], case["rows"], case["cols"])
    round0 = case["rounds"][0]
    nesterov = np.asarray(round0["nesterov"], dtype=np.float64).reshape(shape)
    up_fp64 = np.asarray(round0["polar_fp64"], dtype=np.float64).reshape(shape)
    up_bf16 = np.asarray(round0["polar_bf16"], dtype=np.float64).reshape(shape)
    traj_end = np.asarray(round0["trajectory"][-1], dtype=np.float64).reshape(shape)
    np_out = _polarnp(nesterov, coeffs)

    f64_tier = {
        "upstream_trajectory": traj_end,
        "numpy_f64": np_out,
        "torch_f64": torch.tensor(np_out, dtype=torch.float64).numpy(),
    }
    _consensus(f64_tier, eps=1.0e-12, fault=1.0e-4)

    f32_tier = dict(f64_tier, upstream_polar_fp64=up_fp64)
    _consensus(f32_tier, eps=1.0e-4, fault=1.0e-2)

    low_tier = dict(f32_tier, upstream_polar_bf16=up_bf16)
    _consensus(low_tier, eps=0.15, fault=0.5)
