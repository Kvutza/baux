"""Optimization loop and candidate step execution for FLAME BO."""

from __future__ import annotations

import gc
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .bo_config import borrowed_weights, minibatch_rng
from .minibatch import paired_losses, sample_indices


@dataclass
class _LoopState:
    rewards: deque
    rng: np.random.Generator
    version: int = 0
    evaluations: int = 1
    stop: str = "evaluation_budget"
    restarts: int = 0
    objective_evaluations: int = 1
    last_accepted: int | None = None
    population: int = 0
    data_rng: Any = None
    indices: Any = None
    incumbent_losses: Any = None
    candidate_losses: Any = None
    paired: Any = None
    accepted: bool = False


def _initstate(search, evaluate, settings, base_reward, baseline) -> _LoopState:
    if settings.sampler == "correlated" and search.restarts != 0:
        raise RuntimeError("Correlated search must not use TuRBO restarts")
    if not math.isfinite(base_reward):
        raise ValueError("Baseline reward must be finite")
    state = _LoopState(
        rewards=deque([base_reward], maxlen=settings.history),
        rng=np.random.Generator(np.random.PCG64(settings.seed)),
        restarts=search.restarts,
    )
    if settings.minibatch_size is None:
        return state
    if baseline is None:
        raise ValueError("Minibatch optimization requires its baseline measurement")
    state.population = len(evaluate.tokens)
    state.data_rng = minibatch_rng(settings.minibatch_seed)
    initial_indices = sample_indices(
        state.data_rng, state.population, settings.minibatch_size
    )
    if initial_indices.tolist() != baseline["indices"]:
        raise ValueError("Baseline minibatch does not match its sampling seed")
    if settings.minibatch_refresh > 1:
        # The baseline already measured this exact batch. Reuse it until the
        # configured refresh boundary instead of paying for the same incumbent.
        state.indices = initial_indices
        state.incumbent_losses = np.asarray(baseline["losses"], dtype=np.float64).copy()
        if not np.all(np.isfinite(state.incumbent_losses)):
            raise RuntimeError("Nonfinite baseline loss; refusing to propose")
    search.enable_relative(failure_tolerance=settings.failure_tolerance)
    return state


def _emitbaseline(emit, search, settings, base_reward, baseline):
    correlated = settings.sampler == "correlated"
    minibatch = settings.minibatch_size is not None
    emit(
        {
            "event": "baseline",
            "evaluations": 1,
            "reward": base_reward,
            "base_version": 0,
            "sampler": settings.sampler,
            "controller": settings.controller,
            "reference_seed": settings.reference_seed,
            "reference_version": 0 if correlated else None,
            "reference_radius": float(search.length) if correlated else None,
            "minibatch_refresh": settings.minibatch_refresh if minibatch else None,
            **(
                {"minibatch": baseline, "objective_evaluations": 1} if minibatch else {}
            ),
        }
    )


def _evalincumbent(incumbent, evaluate, indices, backend):
    with borrowed_weights(incumbent, backend) as batch:
        started = time.perf_counter()
        losses = evaluate.losses(batch, indices)
        seconds = time.perf_counter() - started
    del batch
    return losses, seconds


def _refreshbatch(step: int, search, evaluate, settings, state: _LoopState) -> float:
    if settings.minibatch_size is None:
        return 0.0
    if settings.minibatch_refresh > 1 and (
        step == 0 or step % settings.minibatch_refresh != 0
    ):
        return 0.0
    state.indices = sample_indices(
        state.data_rng, state.population, settings.minibatch_size
    )
    incumbent = search.incumbent()
    try:
        state.incumbent_losses, seconds = _evalincumbent(
            incumbent, evaluate, state.indices, settings.backend
        )
    finally:
        del incumbent
        gc.collect()
    if not np.all(np.isfinite(state.incumbent_losses)):
        raise RuntimeError("Nonfinite incumbent loss; refusing to propose")
    state.objective_evaluations += 1
    return seconds


def _checkblocks(layout, changes):
    if len(changes) != len(layout.blocks):
        raise RuntimeError(
            "Native perturbation diagnostics do not match the tensor layout"
        )
    tensors = []
    distance = 0.0
    changed_total = 0
    for block, (changed, squared) in zip(layout.blocks, changes):
        if (
            type(changed) is not int
            or not 0 <= changed <= block.length
            or not math.isfinite(squared)
            or squared < 0
        ):
            raise RuntimeError("Invalid native perturbation diagnostics")
        tensors.append(
            {
                "name": block.name,
                "changed": changed,
                "relative_rms": math.sqrt(squared / block.length) / block.scale,
            }
        )
        distance += squared * block.weight
        changed_total += changed
    return tensors, distance, changed_total


def _verifyproposal(proposals, layout, settings):
    correlated = settings.sampler == "correlated"
    descriptions = proposals.describe()
    geometry = proposals.geometry()
    if len(descriptions) != 1 or len(geometry) != 1:
        raise RuntimeError("Expected diagnostics for exactly one selected proposal")
    seed, score, radius, changes = descriptions[0]
    if not math.isfinite(score) or not math.isfinite(radius) or radius <= 0:
        raise RuntimeError("Invalid native acquisition diagnostics")
    index, persistence = geometry[0]
    if (
        type(index) is not int
        or not 0 <= index < settings.candidates
        or persistence != (0.75 if correlated and index < 2 else 0.0)
    ):
        raise RuntimeError("Invalid native proposal geometry")
    tensors, distance, changed_total = _checkblocks(layout, changes)
    return {
        "seed": seed,
        "score": score,
        "radius": radius,
        "candidate_index": index,
        "persistence": persistence,
        "realized_distance": math.sqrt(distance),
        "changed": changed_total,
        "tensors": tensors,
        "evaluated": distance > 0 and changed_total > 0,
    }


def _askproposal(search, layout, settings, root: int, draw: int, y_scale: float):
    minibatch = settings.minibatch_size is not None
    started = time.perf_counter()
    proposals = search.ask(
        1,
        settings.candidates,
        settings.history,
        root,
        epistemic_scale=settings.paired_epistemic_scale
        if minibatch
        else 1.0 / settings.radius**2,
        aleatoric_scale=0.0 if minibatch else 0.05,
        y_scale=y_scale,
        acquisition="thompson",
        draw_seed=draw,
    )
    info = _verifyproposal(proposals, layout, settings)
    info.update(
        seed_root=root,
        draw_seed=draw,
        ask_seconds=time.perf_counter() - started,
    )
    return proposals, info


def _evalproposal(
    proposals, evaluate, search, settings, state: _LoopState, evaluated: bool
):
    minibatch = settings.minibatch_size is not None
    if not evaluated:
        state.stop = "selected_proposal_unchanged"
        if minibatch:
            state.candidate_losses = state.incumbent_losses.copy()
        return float(search.best), 0.0
    with borrowed_weights(proposals, settings.backend) as batch:
        started = time.perf_counter()
        if minibatch:
            state.candidate_losses = evaluate.losses(batch, state.indices)
            value = -float(np.mean(state.candidate_losses))
        else:
            value = float(evaluate(batch).block_until_ready())
        seconds = time.perf_counter() - started
    del batch
    gc.collect()
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite FLAME loss; refusing to update BAUX")
    state.evaluations += 1
    state.objective_evaluations += 1
    return value, seconds


def _tellproposal(
    search, proposals, settings, state: _LoopState, step: int, value: float
):
    minibatch = settings.minibatch_size is not None
    started = time.perf_counter()
    reject_failure = False
    if minibatch:
        state.paired = paired_losses(
            state.candidate_losses,
            state.incumbent_losses,
            state.population,
            acceptance_se=settings.acceptance_se,
        )
        value = -state.paired.candidate_loss
        reject_failure = (
            state.paired.deteriorated
            if settings.rejection_policy == "deterioration"
            else True
        )
        search.tell_relative(
            proposals,
            value,
            state.paired.candidate_variance,
            -state.paired.incumbent_loss,
            state.paired.incumbent_variance,
            state.paired.improvement,
            state.paired.improvement_variance,
            state.paired.accepted,
            reject_is_failure=reject_failure,
        )
    else:
        search.tell(proposals, [value], [0.0])
    state.accepted = search.sync()[0]
    if minibatch and state.accepted != state.paired.accepted:
        raise RuntimeError("Native acceptance disagrees with paired minibatch decision")
    if state.accepted:
        state.last_accepted = step + 1
        if minibatch and settings.minibatch_refresh > 1:
            state.incumbent_losses = state.candidate_losses.copy()
    tell_seconds = time.perf_counter() - started
    if settings.sampler == "correlated" and search.restarts != 0:
        raise RuntimeError("Correlated search must not use TuRBO restarts")
    return value, tell_seconds, reject_failure


def _minibatchfields(
    settings,
    state: _LoopState,
    reject_failure: bool,
    incumbent_sec: float,
    candidate_sec: float,
):
    if settings.minibatch_size is None:
        return {}
    paired = state.paired
    return {
        "outcome": "accepted"
        if paired.accepted
        else "deteriorated"
        if paired.deteriorated
        else "inconclusive",
        "counted_failure": not paired.accepted and reject_failure,
        "minibatch": {
            "indices": state.indices.tolist(),
            "candidate_losses": state.candidate_losses.tolist(),
            "incumbent_losses": state.incumbent_losses.tolist(),
            **asdict(paired),
        },
        "objective_evaluations": state.objective_evaluations,
        "incumbent_evaluate_seconds": incumbent_sec,
        "candidate_evaluate_seconds": candidate_sec,
        "minibatch_reused": (settings.minibatch_refresh > 1 and incumbent_sec == 0.0),
    }


def _buildprop(
    step: int,
    search,
    settings,
    state: _LoopState,
    info: dict,
    *,
    value: float,
    y_scale: float,
    reference_radius: float | None,
    timings: tuple[float, float, float],
    reject_failure: bool,
):
    correlated = settings.sampler == "correlated"
    incumbent_sec, candidate_sec, tell_sec = timings
    return {
        "event": "proposal",
        "step": step,
        "evaluations": state.evaluations,
        **info,
        "sampler": settings.sampler,
        "controller": settings.controller,
        "reference_seed": settings.reference_seed,
        "reference_version": state.version if correlated else None,
        "reference_radius": reference_radius,
        "next_reference_version": state.version + int(state.accepted)
        if correlated
        else None,
        "next_reference_radius": float(search.length) if correlated else None,
        "base_version": state.version,
        "accepted": state.accepted,
        "reward": value,
        "best_reward": float(search.best),
        "next_radius": float(search.length),
        "restarts": search.restarts,
        "history_len": search.history_len,
        "y_scale": y_scale,
        "evaluate_seconds": incumbent_sec + candidate_sec,
        "tell_seconds": tell_sec,
        **_minibatchfields(
            settings, state, reject_failure, incumbent_sec, candidate_sec
        ),
    }


def _stepproposal(step: int, search, layout, evaluate, settings, state: _LoopState):
    correlated = settings.sampler == "correlated"
    minibatch = settings.minibatch_size is not None
    root, draw = (int(value) for value in state.rng.bit_generator.random_raw(2))
    # Minibatch observation variances and outcomes share raw loss units.
    y_scale = 1.0 if minibatch else max(float(np.std(state.rewards)), 1e-6)
    reference_radius = float(search.length) if correlated else None
    incumbent_sec = _refreshbatch(step, search, evaluate, settings, state)
    proposals, info = _askproposal(search, layout, settings, root, draw, y_scale)
    value, candidate_sec = _evalproposal(
        proposals, evaluate, search, settings, state, info["evaluated"]
    )
    value, tell_sec, reject_failure = _tellproposal(
        search, proposals, settings, state, step, value
    )
    event = _buildprop(
        step,
        search,
        settings,
        state,
        info,
        value=value,
        y_scale=y_scale,
        reference_radius=reference_radius,
        timings=(incumbent_sec, candidate_sec, tell_sec),
        reject_failure=reject_failure,
    )
    state.version += int(state.accepted)
    state.rewards.append(value)
    if not correlated and search.restarts != state.restarts:
        state.rewards.clear()
        state.rewards.append(float(search.best))
        state.restarts = search.restarts
    return event, info["evaluated"]


def _finalsummary(search, settings, state: _LoopState):
    minibatch = settings.minibatch_size is not None
    return {
        "evaluations": state.evaluations,
        "best_reward": float(search.best),
        "base_version": state.version,
        "stop_reason": state.stop,
        **(
            {
                "objective_evaluations": state.objective_evaluations,
                "last_accepted_step": state.last_accepted,
                "checkpoint_selection": "paired_minibatch_training",
                "minibatch_refresh": settings.minibatch_refresh,
                "best_reward_is_full_objective": False,
                "final_minibatch": {
                    "indices": state.indices.tolist(),
                    "losses": (
                        state.candidate_losses
                        if state.accepted
                        else state.incumbent_losses
                    ).tolist(),
                    "variance": state.paired.candidate_variance
                    if state.accepted
                    else state.paired.incumbent_variance,
                },
            }
            if minibatch
            else {}
        ),
    }


def optimize(search, layout, evaluate, settings, base_reward, emit, *, baseline=None):
    """Sequential objective evaluations; candidates are ranked by native BAUX."""
    state = _initstate(search, evaluate, settings, base_reward, baseline)
    _emitbaseline(emit, search, settings, base_reward, baseline)
    for step in range(settings.evaluations - 1):
        event, evaluated = _stepproposal(
            step, search, layout, evaluate, settings, state
        )
        emit(event)
        if not evaluated:
            break
    return _finalsummary(search, settings, state)
