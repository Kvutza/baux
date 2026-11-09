from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from baux import create_optimizer  # noqa: E402
from baux.turbo.config import (  # noqa: E402
    AcquisitionKind,
    ENNSurrogate,
    SurrogateFit,
    turbo_enn,
)

from scripts.optqualcore import y_best  # noqa: E402

CI_DIMS = (2, 8)
CI_SEEDS = tuple(range(5))
FULL_DIMS = (2, 8, 32)
FULL_SEEDS = tuple(range(20))


def _buildcells(dims: tuple[int, ...], seeds: tuple[int, ...]) -> list[dict]:
    cells = []
    for dim in dims:
        bounds = np.tile(np.array([[0.0, 1.0]], dtype=float), (dim, 1))
        config = turbo_enn(
            acquisition=AcquisitionKind.UCB,
            enn=ENNSurrogate(k=10, fit=SurrogateFit(num_samples=10)),
            num_init=min(10, 2 * dim),
        )
        for seed in seeds:
            cells.append(
                {
                    "objective": "sphere_centered_0.5",
                    "dim": dim,
                    "budget": 64,
                    "seed": seed,
                    "acquisition": "ucb",
                    "python_best_y": float(
                        y_best(
                            bounds,
                            config,
                            seed,
                            budget=64,
                            num_arms=2,
                            create_optimizer=create_optimizer,
                        )
                    ),
                }
            )
    return cells


def formatcell(cell: dict) -> str:
    lines = [
        "[[cells]]",
        f'objective = "{cell["objective"]}"',
        f"dim = {cell['dim']}",
        f"budget = {cell['budget']}",
        f"seed = {cell['seed']}",
        f'acquisition = "{cell["acquisition"]}"',
        f"python_best_y = {cell['python_best_y']}",
    ]
    return "\n".join(lines)


def formattoml(ci_cells: list[dict], full_cells: list[dict]) -> str:
    parts = [
        "y_range = 1.0",
        "",
        "[ci]",
        f"dims = {list(CI_DIMS)}",
        f"seeds = {list(CI_SEEDS)}",
        "",
    ]
    for cell in ci_cells:
        parts.append(formatcell(cell))
        parts.append("")
    for cell in full_cells:
        lines = [
            "[[full_cells]]",
            f'objective = "{cell["objective"]}"',
            f"dim = {cell['dim']}",
            f"budget = {cell['budget']}",
            f"seed = {cell['seed']}",
            f'acquisition = "{cell["acquisition"]}"',
            f"python_best_y = {cell['python_best_y']}",
        ]
        parts.append("\n".join(lines))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    out = ROOT / "tests" / "fixtures" / "optqualbase.toml"
    ci_cells = _buildcells(CI_DIMS, CI_SEEDS)
    full_cells = _buildcells(FULL_DIMS, FULL_SEEDS)
    out.write_text(formattoml(ci_cells, full_cells), encoding="utf-8")
    print("wrote", out, "cells", len(ci_cells))


if __name__ == "__main__":
    main()
