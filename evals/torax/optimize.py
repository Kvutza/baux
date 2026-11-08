"""Bayesian optimization experiment over TORAX core transport parameters."""

import numpy as np
from baux import bo
from runner import ToraxRunner


def run_experiment(rounds: int = 5, seed: int = 42) -> None:
    """Run a Bayesian optimization loop over TORAX heating power and location."""
    runner = ToraxRunner(target_te=15.0, target_ti=12.0, max_steps=1)
    rng = np.random.default_rng(seed)
    cfg = bo.OptimizerConfig()
    opt = bo.create_optimizer(bounds=runner.bounds, config=cfg, rng=rng)

    print(
        f"[torax-opt] Starting BayesOptimizer over TORAX core transport | rounds={rounds} seed={seed}"
    )

    best_score = -float("inf")
    best_cand = None

    for step in range(1, rounds + 1):
        pts = opt.ask(1)
        p_heat, loc = pts[0, 0], pts[0, 1]
        score = runner.evaluate(p_heat, loc)
        opt.tell(pts, np.array([score]))

        is_best = score > best_score
        if is_best:
            best_score = score
            best_cand = pts[0]

        print(
            f"torax.round round={step}/{rounds} p_heat={p_heat:.2f}MW loc={loc:.2f} "
            f"score={score:.4f} best={best_score:.4f} accepted={is_best}"
        )

    print(
        f"[torax-opt] Completed: best P_heat={best_cand[0]:.2f}MW best_loc={best_cand[1]:.2f} "
        f"best_score={best_score:.4f}"
    )


if __name__ == "__main__":
    run_experiment(rounds=3, seed=42)
