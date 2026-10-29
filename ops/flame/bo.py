"""Run full-weight FLAME BO with resident BAUX search engines."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import click

from . import bo_config
from .bo_config import (
    FORWARD_HEADROOM_BYTES,
    NOISE_LAWS,
    REJECTION_POLICIES,
    SAMPLERS,
    Settings,
    borrowed_weights,
    loss_evaluator,
    memory_budget,
    minibatch_rng,
    save_best,
)
from .bo_loop import optimize
from .config import Config
from .layout import Layout
from .minibatch import paired_losses, sample_indices
from .objective import SolutionEvaluator, SolutionObjective

__all__ = [
    "FORWARD_HEADROOM_BYTES",
    "NOISE_LAWS",
    "REJECTION_POLICIES",
    "SAMPLERS",
    "Config",
    "Layout",
    "Settings",
    "SolutionEvaluator",
    "SolutionObjective",
    "borrowed_weights",
    "loss_evaluator",
    "main",
    "memory_budget",
    "minibatch_rng",
    "optimize",
    "paired_losses",
    "run",
    "sample_indices",
    "save_best",
]


def _initobjective(tokens, settings: Settings):
    is_native = settings.backend != "jax"
    if is_native and not isinstance(tokens, dict):
        raise ValueError(
            "Native BO supports prepared solution-token corpora only; use --backend jax for legacy token batches"
        )
    solution = (
        SolutionObjective.parse(tokens, Config()) if isinstance(tokens, dict) else None
    )
    objective: dict[str, Any] = (
        solution.metadata
        if solution is not None
        else {"kind": "all_token_cross_entropy"}
    )
    if settings.minibatch_size is not None:
        if (
            not isinstance(tokens, dict)
            or settings.minibatch_size > objective["examples"]
        ):
            raise ValueError(
                "Minibatch BO requires a solution corpus at least as large as its batch"
            )
        objective.update(
            normalization="mean_per_problem_solution_token_loss",
            sampling="uniform_without_replacement_within_batch",
            minibatch_size=settings.minibatch_size,
            acceptance="paired_mean_improvement_exceeds_standard_error_threshold",
            acceptance_se=settings.acceptance_se,
            rejection_policy=settings.rejection_policy,
            minibatch_refresh=settings.minibatch_refresh,
            screening="paired_standard_error_heuristic_not_a_statistical_guarantee",
            uncertainty="finite_population_variance_of_mean_across_problems",
            checkpoint_policy="incumbent_in_gpu_memory_single_final_write",
        )
    return solution, objective


def _selectbackend(settings: Settings):
    if settings.backend == "metal":
        from . import metal as _metal

        return _metal, _metal.metal_device()
    if settings.backend != "jax":
        from . import native as _native

        return _native, _native.cuda_device()
    import jax

    if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") != "platform":
        raise RuntimeError(
            "Run BO in a fresh process with XLA_PYTHON_CLIENT_ALLOCATOR=platform"
        )
    device = jax.devices()[0]
    if device.platform != "gpu" or "T4" not in device.device_kind or device.id != 0:
        raise RuntimeError(
            f"This experiment requires CUDA device 0 to be a T4, got {device}"
        )
    return None, device.device_kind


def _freememory(native, is_metal: bool) -> int:
    if is_metal:
        return native.free_memory()
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(output) * 1024**2


def _checkmemory(native, solution, objective, settings: Settings):
    is_native = settings.backend != "jax"
    size = sum(math.prod(shape) for shape in Config().shapes().values())
    required = (
        native.memory_budget(size, settings.history, 0)
        if is_native
        else memory_budget(size, settings.history, settings.sampler)
        + objective.get("device_input_bytes", 0)
    )
    free = _freememory(native, settings.backend == "metal")
    if required > free:
        raise RuntimeError(
            f"BO needs an estimated {required / 1024**3:.2f} GiB, but only {free / 1024**3:.2f} GiB is free; reduce history"
        )
    if not is_native:
        return None, required, free
    # Allocate bounded engine scratch first, before any checkpoint upload.
    evaluate = native.NativeEvaluator(solution, Config())
    required = native.memory_budget(size, settings.history, evaluate.workspace_bytes)
    if required > free:
        raise RuntimeError(
            f"Native BO needs an estimated {required / 1024**3:.2f} GiB, "
            f"but only {free / 1024**3:.2f} GiB was free; reduce sequence length or history"
        )
    objective["native_workspace_bytes"] = evaluate.workspace_bytes
    objective["native_context_margin_bytes"] = native.CONTEXT_MARGIN_BYTES
    objective["device_input_bytes"] = 0
    return evaluate, required, free


def _loadweights(
    checkpoint: Path, tokens, native, evaluate, settings: Settings, zero_scale
):
    if settings.backend != "jax":
        _, params = native.load_checkpoint(checkpoint)
        layout = Layout.from_torch(params, zero_scale=zero_scale)
        if layout.size != evaluate.weights_len:
            raise RuntimeError("Native weight layout does not match checkpoint")
        flat = layout.flatten_torch(params)
        del params
        gc.collect()
        return layout, evaluate, native.upload_weights(flat)
    from . import model

    config, params = model.load_checkpoint(checkpoint)
    layout = Layout.from_params(params, zero_scale=zero_scale)
    evaluate = loss_evaluator(layout, tokens, config)
    flat = layout.flatten(params).block_until_ready()
    del params
    gc.collect()
    return layout, evaluate, flat


def _evalbaseline(evaluate, flat, objective, settings: Settings):
    if settings.minibatch_size is not None:
        indices = sample_indices(
            minibatch_rng(settings.minibatch_seed),
            objective["examples"],
            settings.minibatch_size,
        )
        losses = evaluate.losses(flat, indices)
        initial = paired_losses(
            losses, losses, objective["examples"], acceptance_se=settings.acceptance_se
        )
        base_reward = -initial.candidate_loss
        base_variance = initial.candidate_variance
        baseline = {
            "indices": indices.tolist(),
            "losses": losses.tolist(),
            "variance": base_variance,
            "mean_loss": initial.candidate_loss,
        }
    else:
        base_reward = float(evaluate(flat).block_until_ready())
        base_variance = 0.0
        baseline = None
    if not math.isfinite(base_reward):
        raise RuntimeError("Nonfinite baseline FLAME loss")
    return base_reward, base_variance, baseline


def _compilejax(
    evaluate, layout, objective, settings: Settings, required: int, free: int
):
    if settings.backend != "jax":
        return evaluate, required
    import jax
    import jax.numpy as jnp

    signature = jax.ShapeDtypeStruct((1, layout.size), jnp.bfloat16)
    if isinstance(evaluate, SolutionEvaluator):
        evaluate = evaluate.compile(signature)
        memory = evaluate.memory_analysis()
        required = memory_budget(
            required, FORWARD_HEADROOM_BYTES[settings.sampler], memory
        )
        if required > free:
            raise RuntimeError(
                f"Compiled coding BO needs an estimated {required / 1024**3:.2f} GiB, "
                f"but only {free / 1024**3:.2f} GiB was free; reduce sequence length or history"
            )
        objective["compiled_temporary_bytes"] = memory.temp_size_in_bytes
        objective["compiler_workspace_margin_bytes"] = 64 * 1024**2
        return evaluate, required
    return evaluate.lower(signature).compile(), required


def _makesearch(
    search_cls,
    block_cls,
    flat,
    layout,
    settings: Settings,
    *,
    base_reward: float,
    base_variance: float,
    native,
):
    is_native = settings.backend != "jax"
    blocks = [
        block_cls(b.key, b.offset, b.length, b.scale, b.weight) for b in layout.blocks
    ]
    search = search_cls(
        flat,
        base_reward,
        blocks,
        settings.history,
        max_pending=1,
        length_init=settings.radius,
        length_min=settings.radius_min,
        length_max=settings.radius_max,
        sampler=settings.sampler,
        reference_seed=settings.reference_seed,
        failure_tolerance=None
        if settings.sampler == "correlated"
        else settings.failure_tolerance,
        **(
            {"base_variance": base_variance}
            if settings.minibatch_size is not None
            else {}
        ),
    )
    # Native reference initialization is deferred until ask; free its input first.
    if not is_native:
        flat.delete()
    del flat
    gc.collect()
    if is_native and settings.backend != "metal":
        native.release_inputcache()
    return search


def _emitevent(stream, event):
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()
    click.echo(
        f"evaluations={event['evaluations']} reward={event['reward']:.7f}"
        + (
            f" outcome={event['outcome']}"
            f" counted_failure={str(event['counted_failure']).lower()}"
            if "outcome" in event
            else ""
        )
    )


def _executerun(
    search,
    layout,
    evaluate,
    settings: Settings,
    base_reward: float,
    baseline,
    *,
    source: dict,
    output: Path,
    metadata: dict,
    report: Path,
):
    minibatch = settings.minibatch_size is not None
    with (output / "events.jsonl").open("w") as stream:
        try:
            result = optimize(
                search,
                layout,
                evaluate,
                settings,
                base_reward,
                lambda event: _emitevent(stream, event),
                **({"baseline": baseline} if minibatch else {}),
            )
            save_best(search, layout, source, output, result)
        except Exception as error:
            report.write_text(
                json.dumps(
                    {**metadata, "status": "failed", "error": str(error)}, indent=2
                )
                + "\n"
            )
            raise
    report.write_text(
        json.dumps(
            {**metadata, **result, "status": "complete"}, indent=2, allow_nan=False
        )
        + "\n"
    )
    return result


def _preparestate(
    checkpoint: Path,
    tokens,
    native,
    evaluate,
    objective,
    settings: Settings,
    *,
    required: int,
    free: int,
    zero_scale,
    block_cls,
    search_cls,
):
    layout, evaluate, flat = _loadweights(
        checkpoint, tokens, native, evaluate, settings, zero_scale
    )
    base_reward, base_variance, baseline = _evalbaseline(
        evaluate, flat, objective, settings
    )
    evaluate, required = _compilejax(
        evaluate, layout, objective, settings, required, free
    )
    search = _makesearch(
        search_cls,
        block_cls,
        flat,
        layout,
        settings,
        base_reward=base_reward,
        base_variance=base_variance,
        native=native,
    )
    del flat
    gc.collect()
    return layout, evaluate, search, (base_reward, baseline, required)


def run(checkpoint: Path, tokens, output: Path, settings: Settings, *, zero_scale=None):
    solution, objective = _initobjective(tokens, settings)
    native, device_name = _selectbackend(settings)
    extension, block_cls, search_cls = bo_config.check_extension(settings)
    evaluate, required, free = _checkmemory(native, solution, objective, settings)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((checkpoint / "manifest.json").read_text())
    layout, evaluate, search, info = _preparestate(
        checkpoint,
        tokens,
        native,
        evaluate,
        objective,
        settings,
        required=required,
        free=free,
        zero_scale=zero_scale,
        block_cls=block_cls,
        search_cls=search_cls,
    )
    metadata = bo_config.build_metadata(
        checkpoint,
        source,
        tokens,
        layout,
        objective,
        settings,
        device_name=device_name,
        extension=extension,
        evaluate=evaluate,
        native=native,
        required=info[2],
    )
    report = output / "run.json"
    report.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return _executerun(
        search,
        layout,
        evaluate,
        settings,
        info[0],
        info[1],
        source=source,
        output=output,
        metadata=metadata,
        report=report,
    )


def _cli(checkpoint, tokens, output, zero_scale, **kwargs):
    if kwargs["backend"] in ("native", "metal") and kwargs["minibatch_size"] is None:
        kwargs["minibatch_size"] = 2
    try:
        settings = Settings(**kwargs)
    except ValueError as error:
        raise click.BadParameter(str(error)) from error
    run(
        checkpoint,
        json.loads(tokens.read_text()),
        output,
        settings,
        zero_scale=zero_scale,
    )


main = click.Command("main", callback=_cli, help=__doc__, params=bo_config.cli_params())


if __name__ == "__main__":
    main()
