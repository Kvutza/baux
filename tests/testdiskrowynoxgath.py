"""row_y must not materialize train_x (RSS regression for large disk N)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import baux
import numpy as np
import pytest
from baux import NeighborIndex

pytestmark = pytest.mark.slow


def _rssbytes() -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out.decode().strip()) * 1024
    except (subprocess.SubprocessError, OSError, PermissionError):
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


@pytest.mark.skipif(
    sys.platform != "darwin", reason="ps RSS probe calibrated on darwin"
)
def test_001():
    """Iterating row_y over all rows must not charge ~N·D·8 bytes of train_x."""
    n = 100_000
    d = 128
    train_x_mib = n * d * 8 / (1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="_enn_row_y_") as work:
        model = baux.enn_init(
            np.empty((0, d)),
            np.empty((0, 1)),
            scale_x=False,
            index_driver=NeighborIndex.BPANN_DISK,
            work_dir=work,
            enn_storage="disk",
        )
        if not hasattr(model.rust_backend, "yvar_obs"):
            type(model.rust_backend).yvar_obs = lambda self: None
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, d))
        y = rng.normal(size=(n, 1))
        model = baux.append(model, x, y)
        model = baux.ensure_sync(model)
        before = _rssbytes()
        got = np.array(
            [
                float(np.asarray(model.rust_backend.row_y(i)).reshape(-1)[0])
                for i in range(n)
            ],
            dtype=float,
        )
        after = _rssbytes()
        delta_mib = (after - before) / (1024 * 1024)
        np.testing.assert_allclose(got, y[:, 0], rtol=0, atol=0)
        assert delta_mib < 0.55 * train_x_mib, (
            f"row_y iteration charged {delta_mib:.1f} MiB; train_x is {train_x_mib:.1f} MiB"
        )
