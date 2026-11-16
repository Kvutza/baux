import baux
import numpy as np
import pytest
from baux.benchmarks import DoubleAckley
from baux.turbo.config import MorboTrustRegion, MultiObjectiveConfig, turbo_one


@pytest.mark.slow
def test_001():
    num_dim = 6
    num_arms = 3
    noise = 0.1
    num_metrics = 2
    rng = np.random.default_rng(42)
    objective = DoubleAckley(noise=noise, rng=rng)
    bounds = np.array([objective.bounds] * num_dim, dtype=float)
    config = turbo_one(
        trust_region=MorboTrustRegion(
            multi_objective=MultiObjectiveConfig(num_metrics=num_metrics)
        )
    )
    state = baux.init(bounds, config=config, seed=42)
    for iteration in range(2):
        state, batch = baux.ask(state, num_arms)
        x_arms = batch.points
        y_obs = objective(x_arms)
        state, _ = baux.tell(state, batch, y_obs)
        print(
            f"Iteration {iteration}: x_arms shape = {x_arms.shape}, y_obs shape = {y_obs.shape}"
        )
