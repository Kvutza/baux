"""End-to-end consumer verification script for BAUX library.

Exercises stateful ask/tell, functional state transitions (init, step, scan),
PyTree parameter bounds, composable transform pipelines, and the ENN surrogate.
"""

from __future__ import annotations

import sys

import numpy as np

try:
    import baux

    print(f"[✓] Successfully imported baux from {baux.__file__}")
except ImportError as exc:
    print(f"[X] Import failed: {exc}")
    sys.exit(1)


def check_asktell() -> None:
    """Verify stateful Ask/Tell loop."""
    print("\n--- 1. Testing Stateful Ask / Tell Loop ---")
    bounds = np.array([[-5.0, 5.0]] * 8)
    opt = baux.create_optimizer(
        bounds=bounds,
        config=baux.turbo_enn(num_init=4),
        rng=np.random.default_rng(0),
    )
    for _ in range(5):
        x = opt.ask(2)
        y = -np.sum(x**2, axis=1)
        opt.tell(x, y)
    print("[✓] Completed ask/tell loop over 5 cycles.")


def check_pytree() -> None:
    """Verify functional pipeline and PyTree scan."""
    print("\n--- 2. Testing Functional Pipeline & PyTree Scan ---")
    spec = baux.pytree_bounds({"w": (4, 4), "b": (4,)}, low=-2.0, high=2.0)
    cfg = (
        baux.lhd(4)
        | baux.raasp(8, num_candidates=32)
        | baux.enn(k=3, num_samples=16)
        | baux.ucb(beta=1.5)
        | baux.tr_bounds(length_init=0.8, length_min=0.05, length_max=1.6)
        | baux.regions(2, sharing="ellipsoid_ball", allocation="least_trials")
    )
    state = baux.init(spec, cfg, seed=0)

    def objective(tree: dict[str, np.ndarray]) -> np.ndarray:
        return -np.sum(tree["w"] ** 2, axis=(-2, -1)) - np.sum(tree["b"] ** 2, axis=-1)

    final_state, hist = baux.scan(
        state,
        objective,
        steps=5,
        num_arms=2,
        use_pytree=True,
    )
    assert len(hist) == 5, f"Expected 5 history steps, got {len(hist)}"
    assert final_state.step_count == 5
    print(
        f"[✓] Successfully executed pure functional scan over PyTrees (steps={final_state.step_count})."
    )


def check_enn() -> None:
    """Verify standalone ENN surrogate lifecycle."""
    print("\n--- 3. Testing Standalone ENN Surrogate ---")
    x = np.random.default_rng(42).uniform(-2.0, 2.0, size=(16, 4))
    y = -np.sum(x**2, axis=1)

    st = baux.enn.init(x, y, scale_x=True)
    params = baux.enn.fit(st, k=3, num_samples=16, seed=42)
    post = baux.enn.posterior(st, x[:4], params)
    assert post.mu.shape[0] == 4, f"Expected 4 predictions, got {post.mu.shape}"
    draws, idx = baux.enn.sample(st, x[:4], params, seeds=[0, 1, 2])
    assert draws.shape == (4, 1, 3), f"Expected shape (4, 1, 3), got {draws.shape}"
    print(
        f"[✓] ENN surrogate fit, posterior (mean range: [{post.mu.min():.3f}, {post.mu.max():.3f}]), and sample verified."
    )


def main() -> None:
    """Execute all consumer integration checks."""
    check_asktell()
    check_pytree()
    check_enn()
    print(
        "\n[SUMMARY] ALL END-TO-END CONSUMER LIBRARY INTEGRATION CHECKS PASSED SUCCESSFULLY!"
    )


if __name__ == "__main__":
    main()
