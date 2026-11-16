from __future__ import annotations

import baux
import numpy as np
import pytest
from baux import OptimizerConfig
from baux._rust import ENN as RustENN
from baux.index import INDEX_NAMES, NeighborIndex
from baux.surrogate import ENNSurrogate


def _enn(train_x, *, index_driver=NeighborIndex.FLAT, work_dir=None):
    train_y = np.zeros((train_x.shape[0], 1), dtype=float)
    kwargs: dict = {"scale_x": False, "index_driver": index_driver}
    if work_dir is not None:
        kwargs["work_dir"] = work_dir
        kwargs["enn_storage"] = "disk"
    return baux.enn_init(train_x, train_y, **kwargs)


def test_001():
    assert set(NeighborIndex) == set(INDEX_NAMES.keys())
    assert INDEX_NAMES[NeighborIndex.FLAT] == "exact"
    assert INDEX_NAMES[NeighborIndex.USEARCH] == "usearch"
    assert INDEX_NAMES[NeighborIndex.BPANN_DISK] == "bpann_disk"
    assert INDEX_NAMES[NeighborIndex.METAL] == "metal"
    assert INDEX_NAMES[NeighborIndex.OPENCL] == "opencl"
    assert INDEX_NAMES[NeighborIndex.CUDA] == "cuda"
    assert "AGX" not in NeighborIndex.__members__
    assert "agx" not in INDEX_NAMES.values()


@pytest.mark.parametrize("driver", ["agx", "AGX", "Agx"])
def test_002(driver):
    with pytest.raises(ValueError, match="Unknown index_driver"):
        RustENN(np.zeros((2, 1)), np.zeros((2, 1)), index_driver=driver)
    with pytest.raises(ValueError, match="Unknown index_driver"):
        bounds = np.array([[0.0, 1.0]])
        config = OptimizerConfig(surrogate=ENNSurrogate(index_driver=driver))
        state = baux.init(bounds, config, seed=0)
        state, batch = baux.ask(state)
        state, _ = baux.tell(state, batch, np.zeros((len(batch), 1)))


@pytest.mark.parametrize("index_driver", [NeighborIndex.METAL, NeighborIndex.OPENCL])
def test_003(index_driver):
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=float)
    query = np.array([[0.1, 0.2], [1.1, 0.1]], dtype=float)
    flat = _enn(train_x)
    try:
        accelerator = _enn(train_x, index_driver=index_driver)
    except ValueError as error:
        unavailable = (
            "index is unavailable",
            "index is unsupported",
            "no default Metal device found",
            "no OpenCL GPU or CPU device found",
        )
        if any(message in str(error) for message in unavailable):
            pytest.skip(str(error))
        raise

    flat_distances, flat_indices = flat.posterior_neighbors(query, search_k=3)
    accelerator_distances, accelerator_indices = accelerator.posterior_neighbors(
        query, search_k=3
    )
    np.testing.assert_array_equal(accelerator_indices, flat_indices)
    np.testing.assert_allclose(
        accelerator_distances, flat_distances, atol=1.0e-4, rtol=0.0
    )


def test_004():
    train_x = np.arange(2049, dtype=float).reshape(-1, 1)
    query = np.array([[0.25]])
    automatic = _enn(train_x, index_driver=NeighborIndex.AUTO)
    exact = _enn(train_x)
    np.testing.assert_array_equal(
        automatic.rust_backend.neighbors(query, 2049),
        exact.rust_backend.neighbors(query, 2049),
    )


def test_005():
    rng = np.random.default_rng(17)
    train_x = rng.normal(size=(32, 5))
    query = rng.normal(size=(4, 5))
    flat = _enn(train_x)
    try:
        usearch = _enn(train_x, index_driver=NeighborIndex.USEARCH)
    except ValueError as error:
        message = str(error)
        if "USearch index is unavailable" in message:
            pytest.skip(message)
        raise

    expected = flat.posterior_neighbors(query, search_k=10)
    actual = usearch.posterior_neighbors(query, search_k=10)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=0.0)


def test_006():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(ValueError, match=r"work_dir"):
        baux.enn_init(
            train_x,
            train_y,
            index_driver=NeighborIndex.BPANN_DISK,
            enn_storage="disk",
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_007(seed: int):
    rng = np.random.default_rng(seed)
    n_train, dim, search_k = 20, 4, 3
    train_x = rng.uniform(0.0, 1.0, size=(n_train, dim))
    query = rng.uniform(0.0, 1.0, size=(1, dim))
    enn = _enn(train_x, index_driver=NeighborIndex.FLAT)
    _, idx = enn.posterior_neighbors(query, search_k=search_k)
    assert idx.shape == (1, search_k)
    assert np.all(idx >= 0)
    assert np.all(idx < n_train)
    print(f"index_driver_neighbor_indices_fuzz seed={seed}")


def test_008():
    train_x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    train_y = np.zeros((2, 1), dtype=float)
    with pytest.raises(
        ValueError, match="Disk storage requires IndexDriver::BpAnnDisk"
    ):
        baux.enn_init(
            train_x,
            train_y,
            index_driver=NeighborIndex.FLAT,
            work_dir="/tmp/enn_work",
            enn_storage="disk",
        )


def test_009(tmp_path):
    work_dir = tmp_path / "persist"
    train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    np.zeros((2, 1), dtype=float)
    model = _enn(
        train_x,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
    )
    assert (work_dir / "train_x.bin").exists()
    model = baux.append(model, np.array([[0.0, 1.0]]), np.zeros((1, 1)))
    assert len(model) == 3
    x_at, _, _ = model.train_rows([2])
    assert x_at.shape == (1, 2)
    np.testing.assert_allclose(x_at[0], [0.0, 1.0])


def test_010(tmp_path):
    from baux.enn.enn_params import ENNParams

    rng = np.random.default_rng(42)
    d = 3
    work_dir = tmp_path / "enn_disk_pending_post"
    x0 = rng.standard_normal((10, d))
    y0 = rng.standard_normal((10, 1))
    x1 = rng.standard_normal((5, d))
    y1 = rng.standard_normal((5, 1))

    inc = baux.enn_init(
        x0,
        y0,
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    inc = baux.ensure_sync(inc)
    inc = baux.append(inc, x1, y1)

    fresh = baux.enn_init(
        np.vstack([x0, x1]),
        np.vstack([y0, y1]),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_fresh"),
        enn_storage="disk",
    )
    fresh = baux.ensure_sync(fresh)
    params = ENNParams(k_neighbors=3, epistemic_scale=1.0, aleatoric_scale=0.1)
    x_test = rng.standard_normal((4, d))
    post_inc = inc.posterior(x_test, params=params)
    post_fresh = fresh.posterior(x_test, params=params)
    np.testing.assert_allclose(post_inc.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_inc.se, post_fresh.se, rtol=1e-5)


def test_011(tmp_path):
    """Reopen must not trust placeholder empty train_y column count for num_outputs."""
    from baux.enn.enn_params import ENNParams

    work_dir = tmp_path / "enn_disk_reopen_num_metrics"
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    train_y = np.array([[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=float)
    params = ENNParams(k_neighbors=2, epistemic_scale=1.0, aleatoric_scale=0.1)
    query = np.array([[0.5, 0.5]], dtype=float)

    baux.enn_init(
        train_x,
        train_y,
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )

    reopened = baux.enn_init(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    assert reopened.num_outputs == 2
    post = reopened.posterior(query, params=params)
    assert post.mu.shape == (1, 2)
    np.testing.assert_allclose(post.mu[0], [0.5, 1.5], rtol=1e-5)


def test_012(tmp_path):
    """Empty disk store reopen must sync num_outputs from metadata.

    Not placeholder train_y.
    """
    work_dir = tmp_path / "enn_disk_empty_reopen_num_metrics"
    baux.enn_init(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 2), dtype=float),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    reopened = baux.enn_init(
        np.zeros((0, 2), dtype=float),
        np.zeros((0, 1), dtype=float),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    assert reopened.num_outputs == 2
    assert len(reopened) == 0


def test_013(tmp_path):
    train_x = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [2.0]], dtype=float)
    model = baux.enn_init(
        train_x,
        train_y,
        scale_x=False,
        metric_learning="auto",
        tied_dims=((0, 1),),
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(tmp_path / "enn_scale_x_bpann"),
        enn_storage="disk",
    )
    assert model.metric is not None
    assert model.metric.scale[0] == pytest.approx(model.metric.scale[1])


def _buildmodel(work_dir, d, rng):
    x_batches = [
        rng.standard_normal((1000, d)),
        rng.standard_normal((1000, d)),
        rng.standard_normal((500, d)),
    ]
    y_batches = [rng.standard_normal((n, 1)) for n in (1000, 1000, 500)]
    model = baux.enn_init(
        np.empty((0, d)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    for x_chunk, y_chunk in zip(x_batches, y_batches, strict=True):
        model = baux.append(model, x_chunk, y_chunk)
        model = model.schedule_flush()
    model = baux.ensure_sync(model)
    return model, np.vstack(x_batches), np.vstack(y_batches)


def test_014(tmp_path):
    """Multi-batch disk store: persist preserves in-session posterior.

    Reopen matches reference.
    """
    import json

    from baux.enn.enn_params import ENNParams

    rng = np.random.default_rng(17)
    d = 32
    work_dir = tmp_path / "enn_disk_persist"
    model, x_all, y_all = _buildmodel(work_dir, d, rng)
    params = ENNParams(k_neighbors=5, epistemic_scale=1.0, aleatoric_scale=0.1)
    x_test = rng.standard_normal((4, d))
    pre_persist = model.posterior(x_test, params=params)
    model = baux.persist_index(model)
    post_persist = model.posterior(x_test, params=params)
    np.testing.assert_allclose(pre_persist.mu, post_persist.mu, rtol=1e-5)
    np.testing.assert_allclose(pre_persist.se, post_persist.se, rtol=1e-5)

    header = json.loads((work_dir / "index" / "header.json").read_text())
    assert header["indexed_rows"] == 2500

    fresh = baux.enn_init(
        x_all,
        y_all,
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(tmp_path / "enn_disk_persist_fresh"),
        enn_storage="disk",
    )
    fresh = baux.ensure_sync(fresh)
    post_fresh = fresh.posterior(x_test, params=params)

    reopened = baux.enn_init(
        np.empty((0, d)),
        np.empty((0, 1)),
        scale_x=False,
        index_driver=NeighborIndex.BPANN_DISK,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    reopened = baux.ensure_sync(reopened)
    post_reopen = reopened.posterior(x_test, params=params)
    np.testing.assert_allclose(post_reopen.mu, post_fresh.mu, rtol=1e-5)
    np.testing.assert_allclose(post_reopen.se, post_fresh.se, rtol=1e-5)


@pytest.mark.parametrize("driver", [NeighborIndex.BPANN_DISK])
def test_015(tmp_path, driver):
    work_dir = tmp_path / f"enn_disk_{driver.name.lower()}"
    rng = np.random.default_rng(42)
    n, d, init = 60, 4, 50
    train_x = rng.standard_normal((n, d))
    train_y = rng.standard_normal((n, 1))
    model = baux.enn_init(
        train_x[:init],
        train_y[:init],
        index_driver=driver,
        work_dir=str(work_dir),
        enn_storage="disk",
    )
    model = baux.append(model, train_x[init:], train_y[init:])
    assert len(model) == n
    model = baux.ensure_sync(model)

    flat = _enn(train_x, index_driver=NeighborIndex.FLAT)
    for qi in range(10):
        query = train_x[qi : qi + 1]
        _, idx = model.posterior_neighbors(query, search_k=1)
        _, flat_idx = flat.posterior_neighbors(query, search_k=1)
        assert int(idx[0, 0]) == int(flat_idx[0, 0]), f"query row {qi}"


@pytest.mark.parametrize("driver", [NeighborIndex.BPANN_DISK])
def test_016(tmp_path, driver):
    work_dir = tmp_path / f"enn_disk_rows_{driver.name.lower()}"
    rng = np.random.default_rng(7)
    n, d = 30, 3
    train_x = rng.standard_normal((n, d))
    mem = _enn(train_x, index_driver=NeighborIndex.FLAT)
    disk = _enn(train_x, index_driver=driver, work_dir=str(work_dir))
    indices = rng.choice(n, size=10, replace=False).tolist()
    mx, my, _ = mem.train_rows(indices)
    dx, dy, _ = disk.train_rows(indices)
    np.testing.assert_allclose(mx, dx)
    np.testing.assert_allclose(my, dy)


def test_017(tmp_path):
    from baux import Device
    from baux.config import OptimizerConfig
    from baux.enn.enn_params import ENNParams
    from baux.index import DistanceMetric, NeighborIndex, Storage
    from baux.region import (
        MorboTrustRegion,
        NoTrustRegion,
        RegionalSearchPolicy,
        TurboTrustRegion,
    )
    from baux.surrogate import ENNSurrogate, GaussianProcess, NoSurrogate

    assert all(
        (
            DistanceMetric.SQUARED_L2,
            ENNSurrogate,
            GaussianProcess,
            NoSurrogate,
            NoTrustRegion,
            MorboTrustRegion,
            RegionalSearchPolicy,
            TurboTrustRegion,
            OptimizerConfig,
        )
    )
    train_x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    train_y = np.array([[0.0], [1.0], [2.0]], dtype=float)
    tree_dir = tmp_path / "tree_store"
    model = baux.enn_init(train_x, train_y, index="tree", work_dir=str(tree_dir))
    assert (
        model.index_kind == "tree"
        and model.device == "cpu"
        and (tree_dir / "train_x.bin").exists()
    )
    exact_model = baux.enn_init(
        train_x,
        train_y,
        index=NeighborIndex.FLAT,
        storage=Storage.MEMORY,
        device=Device.cpu(),
    )
    params = ENNParams(k_neighbors=1, epistemic_scale=1.0, aleatoric_scale=0.0)
    assert exact_model.index_kind == "exact" and exact_model.device == "cpu"
    np.testing.assert_allclose(
        exact_model.posterior(train_x[:1], params).mu,
        exact_model.posterior(train_x[:1], params=params).mu,
    )
