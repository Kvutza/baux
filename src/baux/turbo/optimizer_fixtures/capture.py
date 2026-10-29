from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baux.turbo.config import OptimizerConfig

from .catalog import (
    FIXTURE_OBJECTIVES,
    PREFIX_CONFIG,
    FixtureGeneratorEntry,
    FixtureRunSpec,
    subdir_entry,
)


UPSTREAM_ENN_REVISION = "f8f6a5dada9e85d1a196b7281be4b1b070e0a3ef"
RNG_SEMANTICS = "ENN 0.4.1 owned StdRng fork after constructor draws"


def capture_optimizer(
    bounds: np.ndarray,
    config: OptimizerConfig,
    seed: int,
    run: FixtureRunSpec,
) -> dict[str, Any]:
    from baux import create_optimizer

    objective_fn = FIXTURE_OBJECTIVES[run.objective]
    rng = np.random.default_rng(seed)
    opt = create_optimizer(bounds=bounds, config=config, rng=rng)
    steps = []
    for _ in range(run.num_cycles):
        x = opt.ask(num_arms=run.num_arms)
        y = objective_fn(x)
        opt.tell(x, y)
        steps.append(
            {
                "ask": x.tolist(),
                "tell_y": y.tolist(),
                "tr_length": float(opt.tr_length),
                "tr_count": int(opt.region_count),
            }
        )
    return {
        "upstream_enn_revision": UPSTREAM_ENN_REVISION,
        "rng_semantics": RNG_SEMANTICS,
        "seed": seed,
        "num_cycles": run.num_cycles,
        "num_arms": run.num_arms,
        "objective": run.objective,
        "bounds": bounds.tolist(),
        "steps": steps,
    }


def build_fixture(entry: FixtureGeneratorEntry, seed: int) -> dict[str, Any]:
    config = PREFIX_CONFIG[entry.config_key]
    return capture_optimizer(entry.bounds, config, seed, entry.run)


def output_path(entry: FixtureGeneratorEntry, seed: int, root: Path) -> Path:
    subdir = subdir_entry(entry)
    return root / "tests" / "fixtures" / subdir / f"{entry.prefix}{seed}.json"
