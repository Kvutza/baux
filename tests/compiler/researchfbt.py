"""Full FBT parity research; strict failures are intentional open findings.

Run through //admin/buck2/tests:fbt-research. No skip or expected-failure
markers: a numerical mismatch makes the research target fail. Shared compiler
contracts and identical-input sublayers remain in the required compiler suite.
See research/fbt-graph-2026-11-11.toml for measured results and workload scope.
"""

import numpy as np
import pytest
from baux.experimental import Fbt, ResidentModel, compile


@pytest.mark.parametrize(
    ("backend", "device"),
    [("xla", "cpu"), ("iree", "cpu"), ("iree", "metal"), ("tvm", "cpu")],
)
def test_forward(backend, device):
    # Full production weights, all seven visits, all 65 experts, full 262144
    # vocabulary. A causal prefix is compared with the native 4096-token call.
    model = ResidentModel(seed=19)
    rng = np.random.default_rng(84)
    tokens = rng.integers(0, 262144, model.rows, dtype=np.uint32)
    labels = np.roll(tokens, -1)
    mask = np.ones(model.rows, np.uint8)
    mask[::17] = 0
    reference = model.get(model(tokens, labels, mask)).token_losses[:128].copy()
    graph = Fbt(context=128, batch=1)
    values = (
        model.weights(),
        tokens[:128].reshape(1, 128),
        labels[:128].reshape(1, 128),
        mask[:128].reshape(1, 128).astype(bool),
    )
    executable = compile(graph, graph.inputs, backend=backend, device=device)
    actual, mean = executable.get(executable(executable.put(values)))
    error = float(np.max(np.abs(actual[0] - reference)))
    print(f"FBT {backend}/{device}: maximum loss error={error}", flush=True)
    np.testing.assert_allclose(actual[0], reference, rtol=0, atol=0.03)
    np.testing.assert_allclose(mean, actual[0][mask[:128] != 0].mean(), rtol=1e-5)
