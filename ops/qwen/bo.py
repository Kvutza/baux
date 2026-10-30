"""Correlated full-weight BO for the dense Qwen Coder Metal evaluator."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import click

from .bo_export import _ckptheader, _exportbest, _paired, _sha256file
from .config import Config
from .metal import Evaluator, proposal_stats
from .objective import SolutionObjective

_CONTEXT_TARGETS = (4096, 16384, 32768)


def _sample(rng: random.Random, population: int, size: int) -> list[int]:
    return rng.sample(range(population), size)


def _validatebo(output: Path, options: dict) -> dict:
    cfg = {
        "evaluations": int(options["evaluations"]),
        "candidates": int(options["candidates"]),
        "history": int(options["history"]),
        "minibatch_size": int(options["minibatch_size"]),
        "minibatch_refresh": int(options["minibatch_refresh"]),
        "seed": int(options["seed"]),
        "radius": float(options["radius"]),
        "export_checkpoint": bool(options.get("export_checkpoint", True)),
    }
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if cfg["evaluations"] < 1 or cfg["candidates"] != 4 or cfg["history"] < 1:
        raise ValueError(
            "evaluations must be positive, candidates must be 4, history positive"
        )
    if cfg["minibatch_size"] < 2:
        raise ValueError("minibatch_size must be at least 2")
    if cfg["minibatch_refresh"] < 1:
        raise ValueError("minibatch_refresh must be positive")
    return cfg


def _initreport(
    checkpoint: Path,
    cfg: dict,
    objective: SolutionObjective,
    evaluator: Evaluator,
    search,
    indices: list[int],
    *,
    baseline_losses: list[float],
    mean_loss: float,
) -> dict:
    perturbation_blocks = [
        {
            "key": int(key),
            "offset": int(offset),
            "elements": int(length),
            "rms_scale": float(scale),
            "distance_weight": float(weight),
        }
        for key, offset, length, scale, weight in evaluator.blocks
    ]
    max_tokens = int(objective.tokens.shape[1])
    return {
        "status": "running",
        "model": json.loads((checkpoint / "manifest.json").read_text()),
        "settings": {
            "evaluations": cfg["evaluations"],
            "candidates": cfg["candidates"],
            "history": cfg["history"],
            "history_policy": "fifo_absolute",
            "weight_controller": "turbo",
            "controller": search.controller_info(),
            "surrogate_fit_objective": "row_id_loocv_likelihood_fixed",
            "minibatch_size": cfg["minibatch_size"],
            "minibatch_refresh": cfg["minibatch_refresh"],
            "seed": cfg["seed"],
            "reference_seed": cfg["seed"],
            "radius": cfg["radius"],
            "sampler": "correlated",
            "perturbation_semantics": "dense_full_tensor_correlated_bf16",
            "runtime": "metal",
            "objective_evaluator": "teacher_forced_solution_loss",
            "objective_context_tokens": max_tokens,
            "context_targets": list(_CONTEXT_TARGETS),
            "context_target_reached": max_tokens in _CONTEXT_TARGETS,
            "long_context_loss_path": "cached_causal_attention_bf16_kv_above_256",
            "comparison_baseline": {
                "name": "EGGROLL",
                "method": "hyperscale_evolution_strategies",
                "evaluated_in_run": False,
            },
            "proposal_pipeline": "shared_metal_command_queue",
        },
        "objective": objective.metadata,
        "resources": {
            "model_bf16_elements": evaluator.weights_len,
            "search": search.memory_info(),
            "perturbation_layout": {
                "scale_scheme": "checkpoint_tensor_bf16_rms_fp32",
                "distance_scheme": "equal_tensor_weighted_relative_squared_l2",
                "rounding": "bf16_nearest_even",
                "blocks": perturbation_blocks,
            },
        },
        "baseline": {
            "indices": indices,
            "losses": baseline_losses,
            "mean_loss": mean_loss,
        },
        "objective_evaluations": 1,
        "accepted": 0,
        "events": [],
    }


def _refreshincumbent(
    step: int,
    state: dict,
    cfg: dict,
    evaluator: Evaluator,
    search,
    report: dict,
) -> tuple[float, object]:
    refresh = (
        cfg["minibatch_refresh"] == 1
        or step > 0
        and step % cfg["minibatch_refresh"] == 0
    )
    if not refresh:
        return 0.0, None
    state["indices"] = _sample(
        state["data_rng"], state["population"], cfg["minibatch_size"]
    )
    state["token_rows"], state["mask_rows"] = state["rows"](state["indices"])
    incumbent = search.incumbent()
    incumbent_start = time.perf_counter()
    state["incumbent_losses"] = evaluator.losses(
        incumbent, state["token_rows"], state["mask_rows"]
    )
    incumbent_seconds = time.perf_counter() - incumbent_start
    incumbent_loss_profile = evaluator.loss_profile
    del incumbent
    report["objective_evaluations"] += 1
    return incumbent_seconds, incumbent_loss_profile


def _telldecision(search, proposals, paired: dict) -> tuple[float, float, bool]:
    decision_started = time.perf_counter()
    search.tell_paired(
        proposals,
        -paired["candidate_loss"],
        paired["candidate_variance"],
        -paired["incumbent_loss"],
        paired["incumbent_variance"],
        paired["accepted"],
    )
    decision_seconds = time.perf_counter() - decision_started
    sync_started = time.perf_counter()
    accepted = search.sync()[0]
    sync_seconds = time.perf_counter() - sync_started
    if accepted != paired["accepted"]:
        raise RuntimeError("Native Qwen acceptance disagrees with paired decision")
    return decision_seconds, sync_seconds, accepted


def _bostep(
    step: int,
    state: dict,
    cfg: dict,
    evaluator: Evaluator,
    search,
    report: dict,
    *,
    events,
) -> dict:
    round_started = time.perf_counter()
    prev_evals = report["objective_evaluations"]
    incumbent_seconds, incumbent_profile = _refreshincumbent(
        step, state, cfg, evaluator, search, report
    )
    candidate_start = time.perf_counter()
    proposals, candidate_losses = evaluator.ask_losses(
        search,
        state["token_rows"],
        state["mask_rows"],
        candidates=cfg["candidates"],
        neighbors=cfg["history"],
        seed=state["rng"].getrandbits(64),
        draw_seed=state["rng"].getrandbits(64),
    )
    candidate_seconds = time.perf_counter() - candidate_start
    report["objective_evaluations"] += 1
    paired = _paired(candidate_losses, state["incumbent_losses"], state["population"])
    decision_seconds, sync_seconds, accepted = _telldecision(search, proposals, paired)
    if accepted:
        report["accepted"] += 1
        state["incumbent_losses"] = list(candidate_losses)
    event = {
        "step": step,
        "indices": state["indices"],
        "accepted": accepted,
        "candidate_losses": candidate_losses,
        "incumbent_losses": state["incumbent_losses"],
        "candidate_seconds": candidate_seconds,
        "incumbent_seconds": incumbent_seconds,
        "candidate_loss_profile": evaluator.loss_profile,
        "incumbent_loss_profile": incumbent_profile,
        "perturbation": proposal_stats(proposals, evaluator.weights_len),
        "controller": search.controller_info(),
        "objective_evaluations": report["objective_evaluations"],
        **paired,
    }
    logging_started = time.perf_counter()
    events.write(json.dumps(event, sort_keys=True) + "\n")
    events.flush()
    report["events"].append(event)
    click.echo(
        f"round={step + 1} accepted={str(accepted).lower()} "
        f"loss={paired['candidate_loss']:.7f} "
        f"objective_evaluations={report['objective_evaluations']}"
    )
    logging_seconds = time.perf_counter() - logging_started
    round_seconds = time.perf_counter() - round_started
    return {
        "step": step,
        "round_seconds": round_seconds,
        "incumbent_seconds": incumbent_seconds,
        "proposal_and_score_seconds": candidate_seconds,
        "decision_submit_seconds": decision_seconds,
        "decision_sync_seconds": sync_seconds,
        "logging_seconds": logging_seconds,
        "other_host_seconds": round_seconds
        - (
            incumbent_seconds
            + candidate_seconds
            + decision_seconds
            + sync_seconds
            + logging_seconds
        ),
        "acquisition_candidates": cfg["candidates"],
        "selected_candidates_evaluated": 1,
        "objective_evaluations": report["objective_evaluations"] - prev_evals,
        "minibatch_size": len(state["indices"]),
    }


def _setupbo(checkpoint: Path, objective_path: Path, cfg: dict) -> dict:
    document = json.loads(objective_path.read_text())
    objective = SolutionObjective.parse(document, Config())
    population = len(objective.tokens)
    if cfg["minibatch_size"] > population:
        raise ValueError("minibatch_size cannot exceed the objective population")
    evaluator = Evaluator(checkpoint, max_tokens=int(objective.tokens.shape[1]))
    data_rng = random.Random(cfg["seed"] ^ 0x4D424154)
    indices = _sample(data_rng, population, cfg["minibatch_size"])

    def rows(selected: list[int]) -> tuple[list[list[int]], list[list[bool]]]:
        return (
            [objective.tokens[index, :].tolist() for index in selected],
            [objective.mask[index, :].tolist() for index in selected],
        )

    token_rows, mask_rows = rows(indices)
    baseline_started = time.perf_counter()
    baseline_losses = evaluator.losses(evaluator.weights, token_rows, mask_rows)
    baseline_seconds = time.perf_counter() - baseline_started
    baseline = _paired(baseline_losses, baseline_losses, population)
    search = evaluator.search(
        -baseline["candidate_loss"],
        capacity=cfg["history"],
        radius=cfg["radius"],
        reference_seed=cfg["seed"],
    )
    del evaluator.weights
    report = _initreport(
        checkpoint,
        cfg,
        objective,
        evaluator,
        search,
        indices,
        baseline_losses=baseline_losses,
        mean_loss=baseline["candidate_loss"],
    )
    return {
        "rng": random.Random(cfg["seed"]),
        "data_rng": data_rng,
        "population": population,
        "indices": indices,
        "rows": rows,
        "token_rows": token_rows,
        "mask_rows": mask_rows,
        "incumbent_losses": list(baseline_losses),
        "evaluator": evaluator,
        "search": search,
        "report": report,
        "baseline_seconds": baseline_seconds,
    }


def run(
    checkpoint: Path,
    objective_path: Path,
    output: Path,
    **options: object,
) -> dict:
    run_started = time.perf_counter()
    cfg = _validatebo(output, options)
    state = _setupbo(checkpoint, objective_path, cfg)
    evaluator = state["evaluator"]
    search = state["search"]
    report = state["report"]
    output.mkdir(parents=True)
    started = time.perf_counter()
    with (output / "events.jsonl").open("w") as events:
        round_timings = [
            _bostep(step, state, cfg, evaluator, search, report, events=events)
            for step in range(cfg["evaluations"] - 1)
        ]

    bo_loop_seconds = time.perf_counter() - started
    export_started = time.perf_counter()
    if cfg["export_checkpoint"]:
        _exportbest(checkpoint, output, search.read_best())
    export_seconds = time.perf_counter() - export_started
    report.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "best_loss": float(-search.best),
            "best_reward_is_full_objective": False,
            "checkpoint": str(output / "model.safetensors")
            if cfg["export_checkpoint"]
            else None,
            "timing": {
                "setup_seconds": started - run_started,
                "baseline_seconds": state["baseline_seconds"],
                "baseline_included_in_setup": True,
                "bo_loop_seconds": bo_loop_seconds,
                "export_seconds": export_seconds,
                "rounds": round_timings,
            },
        }
    )
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(**kwargs: object):
    """Optimize Qwen weights against a Qwen-tokenized solution objective."""
    checkpoint = kwargs.pop("checkpoint")
    objective = kwargs.pop("objective")
    output = kwargs.pop("output")
    try:
        run(checkpoint, objective, output, **kwargs)  # type: ignore[arg-type]
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


for _decorator in (
    click.option(
        "--export/--no-export",
        "export_checkpoint",
        default=True,
        show_default=True,
        help="Export best weights as safetensors",
    ),
    click.option("--radius", default=0.01, show_default=True, type=float),
    click.option("--seed", default=0, show_default=True, type=int),
    click.option(
        "--minibatch-refresh",
        default=1,
        show_default=True,
        type=click.IntRange(1, 10000),
    ),
    click.option(
        "--minibatch-size", default=2, show_default=True, type=click.IntRange(2, 374)
    ),
    click.option(
        "--history", default=2, show_default=True, type=click.IntRange(1, 128)
    ),
    click.option(
        "--candidates", default=4, show_default=True, type=click.IntRange(4, 4)
    ),
    click.option(
        "--evaluations", default=8, show_default=True, type=click.IntRange(1, 10000)
    ),
    click.option(
        "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
    ),
    click.argument("objective", type=click.Path(dir_okay=False, path_type=Path)),
    click.argument("checkpoint", type=click.Path(file_okay=False, path_type=Path)),
    click.command(context_settings={"help_option_names": ["-h", "--help"]}),
):
    main = _decorator(main)


__all__ = [
    "_ckptheader",
    "_exportbest",
    "_paired",
    "_sample",
    "_sha256file",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
