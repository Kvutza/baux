#!/usr/bin/env python3
"""Capture unchanged, CPU-executable definitions from pinned PolarExpress and Muon sources.

AST selection loads `optimal_cubic`, `optimal_quintic`, `optimal_composition`, and
`PolarExpress` from `PolarExpress/polar_express.py` alongside `zeropower_via_newtonschulz5`,
`muon_update`, and `adam_update` from `Muon/muon.py` without importing distributed runtime
dependencies.

Academic & Open-Source Attribution:
- Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower,
  "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm",
  arXiv:2505.16932v2 (Algorithm 1), https://github.com/NoahAmsel/PolarExpress
  (polar_express.py, commit f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe).
- Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks",
  https://github.com/KellerJordan/Muon (muon.py, commit f98f1cacc0263b04290753e32be8d498c1efc806).
"""

import ast
import hashlib
import json
import subprocess
from itertools import repeat
from math import inf, sqrt
from pathlib import Path

import click
import numpy as np
import torch
from numpy.polynomial import Polynomial

POLAR_REVISION = "f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe"
MUON_REVISION = "f98f1cacc0263b04290753e32be8d498c1efc806"


def check_repo(upstream: Path, expected: str, label: str) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected:
        raise click.ClickException(
            f"Expected {label} upstream {expected}, found {revision}"
        )
    if subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--"], text=True
    ):
        raise click.ClickException(f"{label} upstream source has local changes")
    return revision


def load_functions(sources: dict[Path, list[str]], namespace: dict):
    for path, wanted in sources.items():
        parsed = ast.parse(path.read_text())
        funcs = [
            stmt
            for stmt in parsed.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name in wanted
        ]
        if len(funcs) != len(wanted):
            raise click.ClickException(f"Missing symbols in {path}: {wanted}")
        exec(
            compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"),
            namespace,
        )


def values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().flatten().tolist()


def solve_cushion(optimal_quintic) -> float:
    """Solve for the 0.024*u cushion in `polar_express.py` where p(l)/p(1) = 0.1 (E = 9/11)."""
    lo, hi = 0.02, 0.03
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        a, b, c = optimal_quintic(mid, 1.0)
        if a * mid + b * (mid**3) + c * (mid**5) < 2.0 / 11.0:
            lo = mid
        else:
            hi = mid
    return mid


def apply_safety(
    coeffs_raw: list[np.ndarray], safety: float = 1.01
) -> list[list[float]]:
    scale = np.array([safety, safety**3, safety**5], dtype=np.float32)
    return [
        (np.asarray(row, dtype=np.float64).astype(np.float32) / scale)
        .astype(np.float64)
        .tolist()
        for row in coeffs_raw
    ]


def remez_fixture(api: dict) -> dict:
    repo_with_safety = api["optimal_composition"](
        l=1e-3, num_iters=5, degree=5, safety_factor_eps=1e-2, cushion=0.02
    )
    scale = np.array([1.01, 1.01**3, 1.01**5], dtype=np.float64)
    repo_raw = [
        (row * scale if idx < 4 else row).tolist()
        for idx, row in enumerate(repo_with_safety)
    ]
    paper_cushion = solve_cushion(api["optimal_quintic"])
    paper_raw = api["optimal_composition"](
        l=1e-3, num_iters=5, degree=5, safety_factor_eps=0.0, cushion=paper_cushion
    )
    paper_polar = apply_safety(paper_raw, safety=1.01)
    return {
        "l": 1e-3,
        "num_iters": 5,
        "degree": 5,
        "safety_factor": 1.01,
        "repo_cushion": 0.02,
        "repo_raw_coefficients": [[float(x) for x in row] for row in repo_raw],
        "repo_coefficients": [[float(x) for x in row] for row in repo_with_safety],
        "paper_cushion": paper_cushion,
        "paper_raw_coefficients": [[float(x) for x in row] for row in paper_raw],
        "paper_polar_coefficients": paper_polar,
    }


def draw_gradient(batch: int, rows: int, cols: int, round_idx: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026 + rows * 100 + cols * 10 + round_idx)
    grad = (
        torch.rand((batch, rows, cols), generator=generator, dtype=torch.float32) - 0.5
    ) * 0.125
    if batch > 1:
        # Zero the second bank to check batch-norm isolation from the first bank.
        grad[1].zero_()
    return grad


def polar_trajectory(
    nesterov: torch.Tensor, coeffs: list[list[float]], eps: float
) -> list[torch.Tensor]:
    x = (
        nesterov.mT.clone()
        if nesterov.size(-2) > nesterov.size(-1)
        else nesterov.clone()
    )
    norm = (x.norm(dim=(-2, -1), keepdim=True) * 1.01).clamp_min(eps)
    x = x / norm
    steps = []
    for a, b, c in coeffs:
        gram = x @ x.mT
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x
        out = x.mT.contiguous() if nesterov.size(-2) > nesterov.size(-1) else x.clone()
        steps.append(out)
    return steps


class ExactTensor(torch.Tensor):
    def bfloat16(self):
        return self


def nesterov_step(
    api: dict, grad: torch.Tensor, momentum: torch.Tensor, beta: float
) -> torch.Tensor:
    rows, cols = grad.size(-2), grad.size(-1)
    mom_probe = momentum.clone()
    api["zeropower_via_newtonschulz5"] = lambda g, steps: g.clone()
    upstream_nesterov = api["muon_update"](
        grad.clone(), mom_probe, beta=beta, ns_steps=5, nesterov=True
    ) / (max(1.0, rows / cols) ** 0.5)
    momentum.lerp_(grad, 1.0 - beta)
    nesterov = grad.clone().lerp_(momentum, beta)
    torch.testing.assert_close(mom_probe, momentum, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(upstream_nesterov, nesterov, rtol=0.0, atol=1e-15)
    return nesterov


def matrix_round(
    api: dict,
    polar_fn,
    polar_coeffs: list[list[float]],
    weights: torch.Tensor,
    momentum: torch.Tensor,
    round_idx: int,
) -> dict:
    batch, rows, cols = weights.shape
    rate, beta, decay, eps = 0.02, 0.95, 0.01, 1e-8
    grad_f32 = draw_gradient(batch, rows, cols, round_idx)
    nesterov = nesterov_step(api, grad_f32.to(torch.float64), momentum, beta)
    steps = polar_trajectory(nesterov, polar_coeffs, eps)
    fp64_polar = torch.Tensor(
        polar_fn(nesterov.clone().as_subclass(ExactTensor), steps=5)
    )
    bf16_polar = polar_fn(nesterov.to(torch.float32), steps=5).to(torch.float64)

    scale = 0.2 * sqrt(max(rows, cols))
    weights.mul_(1.0 - rate * decay).add_(steps[-1], alpha=-rate * scale)
    return {
        "round": round_idx,
        "gradient": values(grad_f32),
        "nesterov": values(nesterov),
        "trajectory": [values(step) for step in steps],
        "polar_fp64": values(fp64_polar),
        "polar_bf16": values(bf16_polar),
        "momentum": values(momentum),
        "weights": values(weights),
    }


def matrix_case(
    api: dict, polar_coeffs: list[list[float]], rows: int, cols: int, batch: int
) -> dict:
    weights = torch.full((batch, rows, cols), 0.5, dtype=torch.float64)
    momentum = torch.zeros((batch, rows, cols), dtype=torch.float64)
    api["coeffs_list"] = [tuple(row) for row in polar_coeffs]
    polar_fn = getattr(api["PolarExpress"], "__wrapped__", api["PolarExpress"])
    rounds = [
        matrix_round(api, polar_fn, polar_coeffs, weights, momentum, round_idx)
        for round_idx in range(4)
    ]
    return {
        "rows": rows,
        "cols": cols,
        "batch": batch,
        "rate": 0.02,
        "momentum": 0.95,
        "decay": 0.01,
        "epsilon": 1e-8,
        "rounds": rounds,
    }


def adam_case(api: dict) -> dict:
    rows, cols, batch = 4, 8, 2
    rate = 0.001
    betas = (0.9, 0.95)
    decay = 0.01
    eps = 1e-8
    weights = torch.full((batch, rows, cols), 0.5, dtype=torch.float64)
    buf1 = torch.zeros((batch, rows, cols), dtype=torch.float64)
    buf2 = torch.zeros((batch, rows, cols), dtype=torch.float64)
    rounds = []
    for round_idx in range(4):
        grad_f32 = draw_gradient(batch, rows, cols, round_idx)
        grad = grad_f32.to(torch.float64)
        update = api["adam_update"](
            grad, buf1, buf2, round_idx + 1, betas=betas, eps=eps
        )
        weights.mul_(1.0 - rate * decay).add_(update, alpha=-rate)
        rounds.append(
            {
                "round": round_idx,
                "gradient": values(grad_f32),
                "weights": values(weights),
            }
        )
    return {
        "rows": rows,
        "cols": cols,
        "batch": batch,
        "rate": rate,
        "momentum": betas[0],
        "beta2": betas[1],
        "decay": decay,
        "epsilon": eps,
        "rounds": rounds,
    }


def main(polar_upstream: Path, muon_upstream: Path, output: Path):
    polar_rev = check_repo(polar_upstream, POLAR_REVISION, "PolarExpress")
    muon_rev = check_repo(muon_upstream, MUON_REVISION, "Muon")

    polar_path = polar_upstream / "polar_express.py"
    muon_path = muon_upstream / "muon.py"

    api = {
        "repeat": repeat,
        "inf": inf,
        "sqrt": sqrt,
        "np": np,
        "Polynomial": Polynomial,
        "torch": torch,
        "__name__": "__main__",
    }
    load_functions(
        {
            polar_path: [
                "optimal_cubic",
                "optimal_quintic",
                "optimal_composition",
                "PolarExpress",
            ],
            muon_path: [
                "zeropower_via_newtonschulz5",
                "muon_update",
                "adam_update",
            ],
        },
        api,
    )
    api["degree_to_remez"] = {
        3: api["optimal_cubic"],
        5: api["optimal_quintic"],
    }

    remez = remez_fixture(api)
    polar_coeffs = remez["paper_polar_coefficients"]
    result = {
        "polar_revision": polar_rev,
        "muon_revision": muon_rev,
        "torch": torch.__version__,
        "device": "cpu",
        "source_hashes": {
            "PolarExpress/polar_express.py": hashlib.sha256(
                polar_path.read_bytes()
            ).hexdigest(),
            "Muon/muon.py": hashlib.sha256(muon_path.read_bytes()).hexdigest(),
        },
        "scope": (
            "PolarExpress Remez optimal_quintic/optimal_composition coefficients, "
            "5-step Gram polar trajectory, and Muon Nesterov + AdamW updates"
        ),
        "remez": remez,
        "cases": [
            matrix_case(api, polar_coeffs, 4, 8, 2),
            matrix_case(api, polar_coeffs, 8, 4, 2),
        ],
        "adam": adam_case(api),
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    click.echo(
        f"Captured upstream PolarExpress ({polar_rev[:12]}) and Muon ({muon_rev[:12]}) fixtures."
    )


for _decorator in (
    click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        required=True,
    ),
    click.option(
        "--muon-upstream",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True,
    ),
    click.option(
        "--polar-upstream",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True,
    ),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
