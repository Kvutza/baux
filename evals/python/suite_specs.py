"""Suite B v4 tune-spec TOML matrix generator: Resident-Metal TuRBO-zero (R=10) vs TuRBO-ENN (R=1)."""

from __future__ import annotations

from pathlib import Path

ZERO_SHARING = (
    "origin-only",
    "box-contain",
    "ellipsoid-ball",
    "nearest-center",
)

ENN_GEOMETRIES = (
    "euclidean",
    "conformal",
    "hyperbolic",
    "wasserstein",
)

SIM_TOKENS = {
    "tokamak-tokagym": 1,
    "oceananigans": 1,
    "fuse": 2,
    "tokamak-tcv": 2,
}


def _header(experiment: str, rounds: int, repetitions: int) -> str:
    base = (
        "#:schema ../../schemas/tune-spec.schema.json\n"
        "version = 4\n"
        f'experiment = "{experiment}"\n'
        'backend = "resident-metal"\n'
        f"rounds = {rounds}\n"
        f"repetitions = {repetitions}\n"
    )
    if experiment in ("tokamak-tokagym", "tokamak-tcv"):
        return base
    if experiment == "oceananigans":
        return (
            base
            + "\n[ocean]\n"
            + 'task = "kelvin-helmholtz"\n'
            + "nx = 64\nny = 1\nnz = 64\n"
            + "actuators = 2\nsteps = 15\ndt = 0.02\n"
        )
    if experiment == "fuse":
        return (
            base
            + "\n[fuse]\n"
            + 'case = "SPARC"\n'
            + 'workflow = "pf-control"\n'
            + 'objective = "max_q95"\n'
            + 'constraints = ["min_q95"]\n'
            + "penalty = 25.0\n"
        )
    raise ValueError(f"Unsupported experiment: {experiment!r}")


def _zero(experiment: str, rounds: int, reps: int, sharing: str) -> str:
    hdr = _header(experiment, rounds, reps)
    return (
        f"{hdr}\n"
        "[optimizer]\n"
        'algorithm = "bayesian-search"\n'
        'distribution = "gaussian"\n'
        "candidates = 4\n"
        "arms = 1\n"
        'acquisition = "random"\n\n'
        "[optimizer.trust-region]\n"
        "initial = 0.8\n"
        "min = 0.0078125\n"
        "max = 1.6\n\n"
        "[optimizer.regional-search]\n"
        "count = 10\n"
        f'sharing = "{sharing}"\n'
        'allocation = "incumbent-ucb1"\n'
        'controller = "turbo"\n'
        "success-tolerance = 3\n"
        "failure-tolerance = 3\n"
    )


def _enn(experiment: str, rounds: int, reps: int, metric: str) -> str:
    hdr = _header(experiment, rounds, reps)
    local_line = "" if metric == "euclidean" else "local-neighbors = 4\n"
    return (
        f"{hdr}\n"
        "[optimizer]\n"
        'algorithm = "bayesian-search"\n'
        'distribution = "gaussian"\n'
        "candidates = 4\n"
        "arms = 1\n"
        'acquisition = "upper-confidence-bound"\n\n'
        "[optimizer.surrogate]\n"
        f'metric = "{metric}"\n'
        f"{local_line}"
        "neighbors = 10\n"
        'fit = "leave-one-out"\n'
        "fit-candidates = 10\n"
        "fit-samples = 5\n\n"
        "[optimizer.trust-region]\n"
        "initial = 0.8\n"
        "min = 0.0078125\n"
        "max = 1.6\n\n"
        "[optimizer.regional-search]\n"
        "count = 1\n"
        'sharing = "origin-only"\n'
        'allocation = "incumbent-ucb1"\n'
        'controller = "turbo"\n'
        "success-tolerance = 3\n"
        "failure-tolerance = 3\n"
    )


def write_specs(
    cfg_dir: Path, experiment: str, rounds: int, reps: int
) -> list[tuple[str, Path, int]]:
    """Write the 8 canonical Suite B v4 TOMLs and return (config_id, path, tokens)."""
    cfg_dir.mkdir(parents=True, exist_ok=True)
    tokens = SIM_TOKENS[experiment]
    specs: list[tuple[str, Path, int]] = []
    for sh in ZERO_SHARING:
        cid = f"turbo_zero_r10_{sh.replace('-', '_')}"
        path = cfg_dir / f"{experiment}__{cid}.toml"
        path.write_text(_zero(experiment, rounds, reps, sh), encoding="utf-8")
        specs.append((cid, path, tokens))
    for gm in ENN_GEOMETRIES:
        cid = f"turbo_enn_r1_{gm}"
        path = cfg_dir / f"{experiment}__{cid}.toml"
        path.write_text(_enn(experiment, rounds, reps, gm), encoding="utf-8")
        specs.append((cid, path, tokens))
    return specs
