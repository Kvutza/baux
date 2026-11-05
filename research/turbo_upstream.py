#!/usr/bin/env python3
"""Verify unchanged TuRBO trust-region length adaptation from pinned upstream source.

AST extraction loads Turbo1._restart, Turbo1._adjust_length, and TurboM._adjust_length
from the pinned uber-research/TuRBO repository without importing GPyTorch.

Academic & Open-Source Attribution:
- David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, and Matthias Poloczek,
  "Scalable Global Optimization via Local Bayesian Optimization", Advances in Neural
  Information Processing Systems (NeurIPS 2019), arXiv:1910.01739,
  https://github.com/uber-research/TuRBO (turbo/turbo_1.py, turbo/turbo_m.py,
  commit de0db39f481d9505bb3610b7b7aa0ebf7702e4a5).
"""

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import click
import numpy as np
import torch

REVISION = "de0db39f481d9505bb3610b7b7aa0ebf7702e4a5"


def load_methods(
    path: Path, class_name: str, method_names: tuple[str, ...], namespace: dict
):
    tree = ast.parse(path.read_text())
    cls_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(cls_nodes) == 1, f"Expected class {class_name} in {path}"
    cls_node = cls_nodes[0]
    selected_methods = [
        stmt
        for stmt in cls_node.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name in method_names
    ]
    assert {m.name for m in selected_methods} == set(method_names)
    trimmed_cls = ast.ClassDef(
        name=cls_node.name,
        bases=cls_node.bases,
        keywords=cls_node.keywords,
        body=selected_methods,
        decorator_list=[],
        type_params=getattr(cls_node, "type_params", []),
    )
    ast.fix_missing_locations(trimmed_cls)
    module = ast.Module(body=[trimmed_cls], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def verify_turbo1(api: dict) -> dict:
    dim = 4
    turbo = api["Turbo1"](
        f=lambda x: 0.0,
        lb=np.zeros(dim),
        ub=np.ones(dim),
        n_init=2 * dim,
        max_evals=100,
        batch_size=1,
        verbose=False,
    )
    assert (turbo.succtol, turbo.failtol) == (3, 4.0)
    assert (turbo.length_init, turbo.length_max, turbo.length_min, turbo.length) == (
        0.8,
        1.6,
        0.5**7,
        0.8,
    )
    assert (turbo.succcount, turbo.failcount) == (0, 0)
    turbo._fX = np.array([[10.0]])
    records = []

    def step(value: float):
        fx_next = np.array([[value]])
        turbo._adjust_length(fx_next)
        turbo._fX = np.vstack((turbo._fX, fx_next))
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


def verify_turbom(api: dict) -> dict:
    dim = 4
    turbom = api["TurboM"](
        f=lambda x: 0.0,
        lb=np.zeros(dim),
        ub=np.ones(dim),
        n_init=2 * dim,
        max_evals=200,
        n_trust_regions=2,
        batch_size=1,
        verbose=False,
    )
    assert (turbom.succtol, turbom.failtol) == (3, 5)
    assert np.all(turbom.length == np.array([0.8, 0.8]))

    turbom.fX = np.array([[10.0], [10.0]])
    turbom._idx = np.array([[0], [1]])
    steps = []

    for region_idx, seq in ((0, (9.0, 8.0, 7.0)), (1, (10.5,) * 5)):
        for val in seq:
            fx_i = np.array([[val]])
            turbom._adjust_length(fx_i, region_idx)
            turbom.fX = np.vstack((turbom.fX, fx_i))
            turbom._idx = np.vstack((turbom._idx, [[region_idx]]))
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


def main(upstream: Path, output: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise click.ClickException(f"Expected upstream {REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--"], text=True
    ):
        raise click.ClickException("Upstream source has local changes")

    paths = [upstream / "turbo/turbo_1.py", upstream / "turbo/turbo_m.py"]
    api = {
        "math": math,
        "np": np,
        "sys": sys,
        "torch": torch,
        "__name__": "__main__",
    }
    load_methods(paths[0], "Turbo1", ("__init__", "_restart", "_adjust_length"), api)
    load_methods(paths[1], "TurboM", ("__init__", "_restart", "_adjust_length"), api)
    t1 = verify_turbo1(api)
    tm = verify_turbom(api)
    hashes = {
        str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    payload = {
        "upstream_revision": revision,
        "source_hashes": hashes,
        "turbo1": t1,
        "turbom": tm,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    click.echo(
        f"Verified upstream TuRBO _adjust_length trajectories at {revision} ({len(hashes)} files)."
    )


for _decorator in (
    click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        default=Path("rust/crates/baux/src/trust_region/fixtures.json"),
        show_default=True,
    ),
    click.option(
        "--upstream",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=Path(".cache/baux/upstream/TuRBO"),
        show_default=True,
    ),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
