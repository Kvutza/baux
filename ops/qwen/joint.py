"""Joint pilot for Qwen dense weight proposals and generated-code scoring."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import click

from .bo import _exportbest, _paired
from .config import Config
from .eval import validate_obj
from .joint_score import (
    FIM_BOUNDARIES,
    DecoderPolicy,
    ScoreTiming,
    _completiontext,
    _decoderpolicy,
    _score,
    _validatepolicy,
    _writejson,
)
from .metal import Evaluator, proposal_stats
from .objective import SolutionObjective
from .tokenizer import load


def _validatejoint(output: Path, options: dict[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    int_keys = (
        "rounds",
        "tasks",
        "samples",
        "max_new_tokens",
        "seed",
        "top_k",
        "batch_size",
    )
    cfg: dict[str, Any] = {k: int(options[k]) for k in int_keys}
    cfg.update(temperature=float(options["temperature"]), top_p=float(options["top_p"]))
    if min(cfg["rounds"], cfg["samples"], cfg["batch_size"]) < 1 or cfg["tasks"] < 2:
        raise ValueError(
            "rounds, samples, and batch_size must be positive; tasks must be at least 2"
        )
    return cfg


def _preparejoint(
    checkpoint: Path, objective_path: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    document = json.loads(objective_path.read_text())
    validate_obj(document)
    objective = SolutionObjective.parse(document, Config())
    examples = document["examples"]
    if cfg["tasks"] > len(examples):
        raise ValueError("tasks cannot exceed the objective population")
    prompts = []
    for index, example in enumerate(examples):
        boundary = next(
            position for position, scored in enumerate(example["loss_mask"]) if scored
        )
        prompts.append(objective.tokens[index, :boundary].tolist())
    max_tokens = max(
        len(prompt) + cfg["max_new_tokens"] for prompt in prompts[: cfg["tasks"]]
    )
    if max_tokens > Config().context:
        raise ValueError("prompt plus generation exceeds the Qwen context")
    evaluator = Evaluator(checkpoint, max_tokens=max_tokens)
    base_policy = DecoderPolicy(cfg["temperature"], cfg["top_p"], cfg["top_k"])
    _validatepolicy(base_policy)
    return {
        "objective": objective,
        "examples": examples,
        "prompts": prompts,
        "evaluator": evaluator,
        "base_policy": base_policy,
    }


def _initreport(
    checkpoint: Path,
    cfg: dict[str, Any],
    objective: SolutionObjective,
    evaluator: Evaluator,
    search,
    base_policy: DecoderPolicy,
    *,
    baseline_scores: list[float],
    baseline_details: list[dict],
    baseline_timing: ScoreTiming,
) -> dict[str, Any]:
    return {
        "status": "running",
        "model": json.loads((checkpoint / "manifest.json").read_text()),
        "objective": objective.metadata,
        "resources": {
            "model_bf16_elements": evaluator.weights_len,
            "search": search.memory_info(),
        },
        "settings": {
            "rounds": cfg["rounds"],
            "tasks": cfg["tasks"],
            "samples": cfg["samples"],
            "batch_size": cfg["batch_size"],
            "max_new_tokens": cfg["max_new_tokens"],
            "seed": cfg["seed"],
            "history": 2,
            "history_policy": "fifo_absolute",
            "weight_controller": "turbo",
            "controller": search.controller_info(),
            "sampler": "correlated_weight_proposals",
            "decoder_policy": base_policy.__dict__,
        },
        "baseline": {
            "policy": base_policy.__dict__,
            "scores": baseline_scores,
            "details": baseline_details,
            "timing": baseline_timing.as_dict(),
        },
        "objective_evaluations": 1,
        "accepted": 0,
        "events_file": "events.jsonl",
    }


def _askproposal(
    evaluator: Evaluator,
    search,
    policy: DecoderPolicy,
    prompts: list[list[int]],
    task_indices: list[int],
    cfg: dict[str, Any],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    root, draw = rng.getrandbits(64), rng.getrandbits(64)
    proposal_started = time.perf_counter()
    if policy.temperature == 0.0:
        initial_prompts = [
            prompts[index] for index in task_indices[: cfg["batch_size"]]
        ]
        proposals, initial_generated = evaluator.ask_generate(
            search, initial_prompts, cfg["max_new_tokens"], seed=root, draw_seed=draw
        )
        return {
            "proposals": proposals,
            "initial": initial_generated,
            "proposal_seconds": 0.0,
            "pipeline_seconds": time.perf_counter() - proposal_started,
        }
    proposals = search.ask(1, 4, 2, root, acquisition="thompson", draw_seed=draw)
    return {
        "proposals": proposals,
        "initial": None,
        "proposal_seconds": time.perf_counter() - proposal_started,
        "pipeline_seconds": 0.0,
    }


def _buildevent(
    step: int,
    policy: DecoderPolicy,
    state: dict[str, Any],
    candidate_scores: list[float],
    candidate_details: list[dict],
    accepted: bool,
    *,
    report: dict[str, Any],
    perturbation: dict[str, object],
    search,
    timings: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    native = getattr(search, "last_profile", None)
    return {
        "step": step,
        "decoder_point": None,
        "policy": policy.__dict__,
        "incumbent_scores": state["incumbent_scores"],
        "candidate_scores": candidate_scores,
        "incumbent_details": state["incumbent_details"],
        "candidate_details": candidate_details,
        "accepted": accepted,
        "decoder_reward": sum(candidate_scores) / len(candidate_scores),
        "objective_evaluations": report["objective_evaluations"],
        "perturbation": perturbation,
        "controller": search.controller_info(),
        "native_proposal_profile_ms": None
        if native is None
        else dict(
            zip(
                (
                    "gpu_pipeline_ms",
                    "host_selection_ms",
                    "host_materialize_ms",
                    "total_ms",
                ),
                map(float, native[:4]),
            )
        ),
        "timing": timings,
        **paired,
    }


def _telldecision(
    search, proposals, candidate_scores: list[float], state: dict[str, Any]
) -> dict[str, Any]:
    decision_started = time.perf_counter()
    paired = _paired(
        [1.0 - s for s in candidate_scores],
        [1.0 - s for s in state["incumbent_scores"]],
        len(state["examples"]),
    )
    decision_seconds = time.perf_counter() - decision_started
    tell_started = time.perf_counter()
    search.tell_paired(
        proposals,
        -paired["candidate_loss"],
        paired["candidate_variance"],
        -paired["incumbent_loss"],
        paired["incumbent_variance"],
        paired["accepted"],
    )
    tell_seconds = time.perf_counter() - tell_started
    sync_started = time.perf_counter()
    accepted = search.sync()[0]
    sync_seconds = time.perf_counter() - sync_started
    if accepted != paired["accepted"]:
        raise RuntimeError("Native Qwen acceptance disagrees with paired decision")
    return {
        "paired": paired,
        "decision_seconds": decision_seconds,
        "tell_seconds": tell_seconds,
        "sync_seconds": sync_seconds,
        "accepted": accepted,
    }


def _jointstep(
    step: int,
    state: dict[str, Any],
    cfg: dict[str, Any],
    evaluator: Evaluator,
    search,
    report: dict[str, Any],
    *,
    events,
) -> float:
    round_started = time.perf_counter()
    policy = state["base_policy"]
    prop = _askproposal(
        evaluator,
        search,
        policy,
        state["prompts"],
        state["task_indices"],
        cfg,
        rng=state["rng"],
    )
    proposals = prop["proposals"]
    candidate_timing = ScoreTiming()
    candidate_started = time.perf_counter()
    candidate_scores, candidate_details = _score(
        evaluator,
        state["tokenizer"],
        state["examples"],
        state["prompts"],
        proposals,
        indices=state["task_indices"],
        policy=policy,
        max_new_tokens=cfg["max_new_tokens"],
        seed=cfg["seed"],
        samples=cfg["samples"],
        batch_size=cfg["batch_size"],
        timing=candidate_timing,
        initial_generated=prop["initial"],
    )
    dec = _telldecision(search, proposals, candidate_scores, state)
    report["objective_evaluations"] += 1
    if dec["accepted"]:
        report["accepted"] += 1
    timings: dict[str, Any] = {
        "round_seconds": 0.0,
        "proposal_seconds": prop["proposal_seconds"],
        "candidate_pipeline_seconds": prop["pipeline_seconds"],
        "incumbent_score_seconds": 0.0,
        "candidate_score_seconds": time.perf_counter() - candidate_started,
        "decision_seconds": dec["decision_seconds"],
        "tell_seconds": dec["tell_seconds"],
        "sync_seconds": dec["sync_seconds"],
        "logging_seconds": 0.0,
        "incumbent_score": None,
        "candidate_score": candidate_timing.as_dict(),
    }
    event = _buildevent(
        step,
        policy,
        state,
        candidate_scores,
        candidate_details,
        dec["accepted"],
        report=report,
        perturbation=proposal_stats(proposals, evaluator.weights_len),
        search=search,
        timings=timings,
        paired=dec["paired"],
    )
    if dec["accepted"]:
        state["incumbent_scores"] = candidate_scores
        state["incumbent_details"] = candidate_details
    logging_started = time.perf_counter()
    timings["round_seconds"] = time.perf_counter() - round_started
    json.dumps(event, sort_keys=True)
    timings["logging_seconds"] = time.perf_counter() - logging_started
    timings["round_seconds"] = time.perf_counter() - round_started
    events_io_started = time.perf_counter()
    events.write(json.dumps(event, sort_keys=True) + "\n")
    events.flush()
    io_elapsed = time.perf_counter() - events_io_started
    for key in state["round_timing_totals"]:
        state["round_timing_totals"][key] += float(timings[key])
    click.echo(
        f"round={step + 1} temperature={policy.temperature:.3f} "
        f"accepted={str(dec['accepted']).lower()} score={event['decoder_reward']:.4f}"
    )
    return io_elapsed


def _setupjoint(
    checkpoint: Path, objective_path: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    prep = _preparejoint(checkpoint, objective_path, cfg)
    evaluator = prep["evaluator"]
    examples = prep["examples"]
    prompts = prep["prompts"]
    base_policy = prep["base_policy"]
    tokenizer = load(checkpoint)
    task_indices = list(range(cfg["tasks"]))
    baseline_started = time.perf_counter()
    baseline_timing = ScoreTiming()
    baseline_scores, baseline_details = _score(
        evaluator,
        tokenizer,
        examples,
        prompts,
        evaluator.weights,
        indices=task_indices,
        policy=base_policy,
        max_new_tokens=cfg["max_new_tokens"],
        seed=cfg["seed"],
        samples=cfg["samples"],
        batch_size=cfg["batch_size"],
        timing=baseline_timing,
    )
    baseline = _paired(
        [1.0 - score for score in baseline_scores],
        [1.0 - score for score in baseline_scores],
        len(examples),
    )
    search = evaluator.search(
        -baseline["candidate_loss"], capacity=2, reference_seed=cfg["seed"]
    )
    if hasattr(search, "set_profiling"):
        search.set_profiling(True)
    del evaluator.weights
    report = _initreport(
        checkpoint,
        cfg,
        prep["objective"],
        evaluator,
        search,
        base_policy,
        baseline_scores=baseline_scores,
        baseline_details=baseline_details,
        baseline_timing=baseline_timing,
    )
    return {
        **prep,
        "rng": random.Random(cfg["seed"]),
        "tokenizer": tokenizer,
        "task_indices": task_indices,
        "incumbent_scores": baseline_scores,
        "incumbent_details": baseline_details,
        "round_timing_totals": {
            "round_seconds": 0.0,
            "proposal_seconds": 0.0,
            "candidate_pipeline_seconds": 0.0,
            "incumbent_score_seconds": 0.0,
            "candidate_score_seconds": 0.0,
            "decision_seconds": 0.0,
            "tell_seconds": 0.0,
            "sync_seconds": 0.0,
            "logging_seconds": 0.0,
        },
        "search": search,
        "report": report,
        "baseline_started": baseline_started,
        "baseline_seconds": baseline_timing.elapsed_seconds,
    }


def run(checkpoint: Path, objective_path: Path, output: Path, **options: Any) -> dict:
    cfg = _validatejoint(output, options)
    run_started = time.perf_counter()
    state = _setupjoint(checkpoint, objective_path, cfg)
    evaluator, search, report = state["evaluator"], state["search"], state["report"]
    output.mkdir(parents=True)
    bo_loop_started = time.perf_counter()
    with (output / "events.jsonl").open("w") as events:
        events_io_seconds = sum(
            _jointstep(step, state, cfg, evaluator, search, report, events=events)
            for step in range(cfg["rounds"])
        )
    bo_loop_seconds = time.perf_counter() - bo_loop_started
    best_score = float(search.best)
    export_started = time.perf_counter()
    _exportbest(checkpoint, output, search.read_best())
    export_seconds = time.perf_counter() - export_started
    report["timing"] = {
        "setup_seconds": state["baseline_started"] - run_started,
        "baseline_seconds": state["baseline_seconds"],
        "bo_loop_seconds": bo_loop_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - run_started,
        "events_io_seconds": events_io_seconds,
        "round_totals": state["round_timing_totals"],
    }
    report.update(
        status="complete",
        elapsed_seconds=report["timing"]["total_seconds"],
        best_score=best_score,
        checkpoint=str(output / "model.safetensors"),
    )
    _writejson(output / "run.json", report, indent=2)
    return report


def main(**kwargs: Any):
    """Run full-weight BO with fixed decoder settings."""
    checkpoint = kwargs.pop("checkpoint")
    objective = kwargs.pop("objective")
    output = kwargs.pop("output")
    try:
        run(checkpoint, objective, output, **kwargs)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise click.ClickException(str(error)) from error


for _decorator in (
    click.option(
        "--top-k", default=0, show_default=True, type=click.IntRange(0, Config().vocab)
    ),
    click.option("--top-p", default=1.0, show_default=True, type=float),
    click.option("--temperature", default=0.0, show_default=True, type=float),
    click.option(
        "--seed", default=0, show_default=True, type=click.IntRange(0, 2**64 - 1)
    ),
    click.option(
        "--max-new-tokens",
        default=128,
        show_default=True,
        type=click.IntRange(1, Config().context),
    ),
    click.option(
        "--batch-size", default=2, show_default=True, type=click.IntRange(1, 32)
    ),
    click.option("--samples", default=1, show_default=True, type=click.IntRange(1, 16)),
    click.option("--tasks", default=2, show_default=True, type=click.IntRange(2, 374)),
    click.option(
        "--rounds", default=5, show_default=True, type=click.IntRange(1, 1000)
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
    "DecoderPolicy",
    "FIM_BOUNDARIES",
    "ScoreTiming",
    "_completiontext",
    "_decoderpolicy",
    "_score",
    "_validatepolicy",
    "_writejson",
    "main",
    "run",
]


if __name__ == "__main__":
    main()
