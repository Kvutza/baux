#!/usr/bin/env python3
"""Verify TuRBO trust-region length adaptation from published paper equations (arXiv:1910.01739).

Clean-room mathematical reference implementation of the trust-region halving/doubling
and restart rules defined in Section 2 of Eriksson et al. (NeurIPS 2019) and MIT-licensed
BoTorch (`pytorch/botorch`) / `yubo-research/enn`.

Academic & Open-Source Attribution:
- David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, and Matthias Poloczek,
  "Scalable Global Optimization via Local Bayesian Optimization", Advances in Neural
  Information Processing Systems (NeurIPS 2019), arXiv:1910.01739 (Section 2).
- Maximilian Balandat et al., "BoTorch: A Framework for Efficient Monte-Carlo Bayesian
  Optimization", Advances in Neural Information Processing Systems (NeurIPS 2020),
  https://github.com/pytorch/botorch (MIT License).
"""

import json
import math
from pathlib import Path

import click
import numpy as np

PAPER_ARXIV = "1910.01739"
ENN_REVISION = "f8f6a5dada9e85d1a196b7281be4b1b070e0a3ef"


class PaperTurbo1:
    """Clean-room implementation of TuRBO-1 length adaptation (Eriksson et al., 2019, Sec. 2)."""

    def __init__(self, dim: int, batch_size: int = 1):
        self.dim = dim
        self.batch_size = batch_size
        self.succtol = 3
        self.failtol = max(4.0, float(dim) / float(batch_size))
        self.length_init = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.length = self.length_init
        self.succcount = 0
        self.failcount = 0
        self.best_fx = math.inf

    def adjust_length(self, fx_next: float):
        if fx_next < self.best_fx - 1e-3 * abs(self.best_fx):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += self.batch_size
        if self.succcount == self.succtol:
            self.length = min(2.0 * self.length, self.length_max)
            self.succcount = 0
        elif self.failcount >= self.failtol:
            self.length = self.length / 2.0
            self.failcount = 0
        self.best_fx = min(self.best_fx, fx_next)


class PaperTurboM:
    """Clean-room implementation of TuRBO-M per-region length adaptation (Eriksson et al., 2019, Sec. 2)."""

    def __init__(self, dim: int, n_trust_regions: int, batch_size: int = 1):
        self.dim = dim
        self.n_trust_regions = n_trust_regions
        self.batch_size = batch_size
        self.succtol = 3
        self.failtol = max(5, int(math.ceil(dim / batch_size)))
        self.length_init = 0.8
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.length = np.full(n_trust_regions, self.length_init, dtype=float)
        self.succcount = np.zeros(n_trust_regions, dtype=int)
        self.failcount = np.zeros(n_trust_regions, dtype=int)
        self.best_fx = np.full(n_trust_regions, math.inf, dtype=float)

    def adjust_length(self, fx_next: float, region_idx: int):
        best = self.best_fx[region_idx]
        if fx_next < best - 1e-3 * abs(best):
            self.succcount[region_idx] += 1
            self.failcount[region_idx] = 0
        else:
            self.succcount[region_idx] = 0
            self.failcount[region_idx] += 1
        if self.succcount[region_idx] == self.succtol:
            self.length[region_idx] = min(
                2.0 * self.length[region_idx], self.length_max
            )
            self.succcount[region_idx] = 0
        elif self.failcount[region_idx] >= self.failtol:
            self.length[region_idx] = self.length[region_idx] / 2.0
            self.failcount[region_idx] = 0
        self.best_fx[region_idx] = min(best, fx_next)


def verify_turbo1() -> dict:
    dim = 4
    turbo = PaperTurbo1(dim=dim, batch_size=1)
    assert (turbo.succtol, turbo.failtol) == (3, 4.0)
    assert (turbo.length_init, turbo.length_max, turbo.length_min, turbo.length) == (
        0.8,
        1.6,
        0.5**7,
        0.8,
    )
    assert (turbo.succcount, turbo.failcount) == (0, 0)
    turbo.best_fx = 10.0
    records = []

    def step(value: float):
        turbo.adjust_length(value)
        state = (turbo.length, turbo.succcount, turbo.failcount)
        records.append(
            {
                "fx": float(value),
                "length": float(turbo.length),
                "succcount": int(turbo.succcount),
                "failcount": int(turbo.failcount),
                "needs_restart": bool(turbo.length < turbo.length_min),
            }
        )
        return state

    expected_steps = [
        (9.0, (0.8, 1, 0)),
        (8.0, (0.8, 2, 0)),
        (7.0, (1.6, 0, 0)),
        (6.0, (1.6, 1, 0)),
        (5.0, (1.6, 2, 0)),
        (4.0, (1.6, 0, 0)),
        (3.9, (1.6, 1, 0)),
        (3.899, (1.6, 0, 1)),
        (4.0, (1.6, 0, 2)),
        (4.1, (1.6, 0, 3)),
        (4.2, (0.8, 0, 0)),
    ]
    for val, expected_state in expected_steps:
        assert step(val) == expected_state

    for expected_len in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        assert turbo.length >= turbo.length_min
        assert [step(5.0)[1:] for _ in range(3)] == [(0, 1), (0, 2), (0, 3)]
        length, succ, fail = step(5.0)
        assert math.isclose(length, expected_len, rel_tol=0.0, abs_tol=1e-15)
        assert (succ, fail) == (0, 0)
    assert turbo.length < turbo.length_min
    return {
        "dim": dim,
        "batch_size": 1,
        "succtol": int(turbo.succtol),
        "failtol": int(turbo.failtol),
        "length_init": float(turbo.length_init),
        "length_min": float(turbo.length_min),
        "length_max": float(turbo.length_max),
        "initial_fx": 10.0,
        "steps": records,
    }


def verify_turbom() -> dict:
    dim = 4
    turbom = PaperTurboM(dim=dim, n_trust_regions=2, batch_size=1)
    assert (turbom.succtol, turbom.failtol) == (3, 5)
    assert np.all(turbom.length == np.array([0.8, 0.8]))

    turbom.best_fx = np.array([10.0, 10.0], dtype=float)
    steps = []

    for region_idx, seq in ((0, (9.0, 8.0, 7.0)), (1, (10.5,) * 5)):
        for val in seq:
            turbom.adjust_length(val, region_idx)
            steps.append(
                {
                    "region": region_idx,
                    "fx": float(val),
                    "lengths": [float(x) for x in turbom.length],
                    "succcount": [int(x) for x in turbom.succcount],
                    "failcount": [int(x) for x in turbom.failcount],
                }
            )
    assert np.allclose(turbom.length, np.array([1.6, 0.4]))
    return {
        "dim": dim,
        "n_trust_regions": 2,
        "batch_size": 1,
        "succtol": int(turbom.succtol),
        "failtol": int(turbom.failtol),
        "initial_fx": [10.0, 10.0],
        "steps": steps,
    }


def main(output: Path):
    t1 = verify_turbo1()
    tm = verify_turbom()
    payload = {
        "paper_arxiv": PAPER_ARXIV,
        "enn_revision": ENN_REVISION,
        "turbo1": t1,
        "turbom": tm,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    click.echo(f"Verified clean-room TuRBO trajectories (arXiv:{PAPER_ARXIV}).")


for _decorator in (
    click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        default=Path("rust/crates/baux/src/trust_region/fixtures.json"),
        show_default=True,
    ),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
