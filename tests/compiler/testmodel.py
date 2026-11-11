"""The production 265M FBT/PISA forward through two native runtimes."""

import numpy as np
from baux.experimental import ResidentModel


def test_parity():
    reference = None
    for backend in ("native", "tvm"):
        model = ResidentModel(seed=19, backend=backend)
        rng = np.random.default_rng(84)
        tokens = rng.integers(0, 128, model.rows, dtype=np.uint32)
        labels = np.roll(tokens, -1)
        mask = np.ones(model.rows, np.uint8)
        mask[::17] = 0
        first = model.submit(tokens, labels, mask).wait()
        result = model.get(first)
        saved = (result.token_losses.copy(), result.sequence_nlls.copy())
        if reference is None:
            reference = saved
        else:
            np.testing.assert_array_equal(result.token_losses, reference[0])
            np.testing.assert_array_equal(result.sequence_nlls, reference[1])
        # Later execution must not reuse storage still owned by an observation.
        model.submit(tokens[::-1].copy(), labels, mask).wait()
        retained = model.get(first)
        np.testing.assert_array_equal(retained.token_losses, saved[0])
        np.testing.assert_array_equal(retained.sequence_nlls, saved[1])
