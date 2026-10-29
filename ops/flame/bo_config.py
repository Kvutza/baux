"""Configuration, memory budgets, evaluators, and checkpoint saving for FLAME BO."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from .objective import SolutionObjective

SAMPLERS = ("correlated", "gaussian", "independent")
REJECTION_POLICIES = ("deterioration", "all")
# Incremental workspace estimates, excluding resident rows and the reference.
# Correlated covers the fixed 1x8 fixture's measured ~2234 MiB forward workspace
# plus ~24 MiB native context; this is not a guarantee for arbitrary batches.
FORWARD_HEADROOM_BYTES = {
    "correlated": 9 * 1024**3 // 4,
    "gaussian": 4 * 1024**3,
    "independent": 4 * 1024**3,
}
NOISE_LAWS = {
    "correlated": "correlated_gaussian",
    "gaussian": "independent_gaussian",
    "independent": "legacy_independent_signs",
}


def _checkintegers(settings: Settings):
    for name, low, high in (
        ("evaluations", 2, 1_000_000),
        ("candidates", 1, 8),
        ("history", 2, 128),
        ("seed", 0, 2**64 - 1),
        ("reference_seed", 0, 2**64 - 1),
        ("minibatch_seed", 0, 2**64 - 1),
        ("minibatch_refresh", 1, 1_000_000),
    ):
        value = getattr(settings, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _checksampler(settings: Settings):
    if settings.sampler == "correlated":
        if settings.candidates != 4:
            raise ValueError("correlated sampler requires candidates=4")
        if settings.failure_tolerance is not None and settings.minibatch_size is None:
            raise ValueError(
                "failure_tolerance requires paired minibatches or a baseline sampler"
            )
    if settings.sampler != "correlated" or settings.minibatch_size is not None:
        if settings.failure_tolerance is None:
            object.__setattr__(settings, "failure_tolerance", 4)
        if (
            type(settings.failure_tolerance) is not int
            or not 1 <= settings.failure_tolerance <= 2**32 - 1
        ):
            raise ValueError("failure_tolerance must be an integer in [1, 4294967295]")


def _checkminibatch(settings: Settings):
    if settings.minibatch_size is not None and (
        type(settings.minibatch_size) is not int
        or settings.minibatch_size < 2
        or settings.sampler != "correlated"
    ):
        raise ValueError(
            "minibatch_size requires correlated sampling and at least two problems"
        )
    if settings.minibatch_refresh != 1 and settings.minibatch_size is None:
        raise ValueError("minibatch_refresh requires minibatch_size")
    if settings.backend == "metal" and (
        settings.history != 2 or settings.failure_tolerance != 4
    ):
        raise ValueError(
            "Metal BO currently requires history=2 and failure_tolerance=4"
        )
    if (
        isinstance(settings.acceptance_se, bool)
        or not math.isfinite(settings.acceptance_se)
        or settings.acceptance_se < 0
    ):
        raise ValueError("acceptance_se must be finite and nonnegative")
    if (
        isinstance(settings.paired_epistemic_scale, bool)
        or not math.isfinite(settings.paired_epistemic_scale)
        or not float(np.finfo(np.float32).tiny)
        <= settings.paired_epistemic_scale
        <= float(np.finfo(np.float32).max)
    ):
        raise ValueError("paired_epistemic_scale must be positive and finite in FP32")


def _checkradii(settings: Settings):
    if not all(
        math.isfinite(value)
        for value in (settings.radius, settings.radius_min, settings.radius_max)
    ):
        raise ValueError("Radii must be finite")
    if (
        not 0 < settings.radius_min <= settings.radius <= settings.radius_max
        or settings.radius_min == settings.radius_max
    ):
        raise ValueError(
            "Require 0 < radius_min <= radius <= radius_max and distinct bounds"
        )
    smallest, largest = (
        float(np.finfo(np.float32).tiny),
        float(np.finfo(np.float32).max),
    )
    if (
        not smallest <= settings.radius_min <= largest
        or not smallest <= settings.radius_max <= largest
        or not smallest <= 1.0 / settings.radius**2 <= largest
    ):
        raise ValueError("Radius and metric scaling must be representable in FP32")
    if settings.sampler == "correlated":
        low, high = np.float32(settings.radius_min), np.float32(settings.radius_max)
        # Match the native inward rounding without altering requested settings.
        if float(low) < settings.radius_min:
            low = np.nextafter(low, np.float32(np.inf))
        if float(high) > settings.radius_max:
            high = np.nextafter(high, np.float32(-np.inf))
        if low >= high:
            raise ValueError(
                "Correlated radius bounds must contain two distinct FP32 radii"
            )


@dataclass(frozen=True)
class Settings:
    evaluations: int = 8
    candidates: int = 4
    history: int = 2
    radius: float = 0.01
    radius_min: float = 0.0001
    radius_max: float = 0.08
    sampler: str = "correlated"
    failure_tolerance: int | None = None
    seed: int = 0
    reference_seed: int = 0
    minibatch_size: int | None = None
    minibatch_seed: int = 0
    minibatch_refresh: int = 1
    acceptance_se: float = 2.0
    rejection_policy: str = "deterioration"
    paired_epistemic_scale: float = 1.0
    backend: str = "jax"

    def __post_init__(self):
        if self.rejection_policy not in REJECTION_POLICIES:
            raise ValueError("rejection_policy must be deterioration or all")
        if self.backend not in ("native", "jax", "metal"):
            raise ValueError("backend must be native, jax, or metal")
        if self.backend in ("native", "metal") and (
            self.sampler != "correlated" or self.minibatch_size is None
        ):
            raise ValueError(
                "Native BO requires correlated sampling and paired minibatches; use --backend jax for legacy objectives"
            )
        if self.sampler not in SAMPLERS:
            raise ValueError("sampler must be correlated, gaussian, or independent")
        _checkintegers(self)
        _checksampler(self)
        _checkminibatch(self)
        _checkradii(self)

    @property
    def controller(self):
        if self.sampler == "correlated":
            if self.minibatch_size is not None:
                return (
                    "paired_accepted_radius_with_deterioration_contraction"
                    if self.rejection_policy == "deterioration"
                    else "paired_accepted_radius_with_all_rejection_contraction_legacy"
                )
            return "acquisition_selected_radius"
        return "turbo_success_failure_with_explicit_failure_tolerance"


def memory_budget(size: int, history: int, sampler: Any = "correlated") -> int:
    """Resident rows, a dense BF16 reference, and estimated forward workspace."""
    if not isinstance(sampler, str):
        if sampler is None:
            raise RuntimeError("Compiler memory statistics are required for coding BO")
        workspace = sampler.temp_size_in_bytes + sampler.output_size_in_bytes
        if workspace < 0:
            raise RuntimeError("Invalid compiler memory statistics")
        return size + max(0, workspace + 64 * 1024**2 - history)
    if sampler not in SAMPLERS:
        raise ValueError("sampler must be correlated, gaussian, or independent")
    reference_bytes = 2 * size if sampler == "correlated" else 0
    return (
        (history + 2) * ((size + 127) & ~127) * 2
        + reference_bytes
        + FORWARD_HEADROOM_BYTES[sampler]
    )


def minibatch_rng(seed):
    # Separate problem sampling from the proposal/acquisition random stream.
    return np.random.default_rng(np.random.SeedSequence([seed, 0x4D424154]))


def loss_evaluator(layout, tokens, config):
    import jax
    import jax.numpy as jnp

    from . import model

    if isinstance(tokens, dict):
        return SolutionObjective.parse(tokens, config).evaluator(layout, config)
    tokens = np.asarray(tokens)
    if (
        tokens.ndim != 2
        or not np.issubdtype(tokens.dtype, np.integer)
        or tokens.shape[0] < 1
        or not 2 <= tokens.shape[1] <= config.context
    ):
        raise ValueError(
            "Expected a nonempty batch of unpadded integer token sequences"
        )
    if tokens.min() < 0 or tokens.max() >= config.vocab:
        raise ValueError("Token IDs are outside the model vocabulary")
    tokens = jnp.asarray(tokens, dtype=jnp.int32)

    @jax.jit
    def evaluate(flat):
        # Reshape inside the compiled forward to avoid an eager DLPack input copy.
        flat = flat.reshape((layout.size,))
        logits = model.forward(layout.unflatten(flat), tokens, config)
        return -jnp.mean(model.token_loss(logits, tokens))

    return evaluate


@contextmanager
def borrowed_weights(weights, backend):
    if backend in ("native", "metal"):
        # The native call releases its DLPack lease before returning or raising.
        yield weights
        return
    import jax

    batch = jax.dlpack.from_dlpack(weights)
    try:
        yield batch
    finally:
        batch.delete()


def save_best(search, layout, source, output, result):
    import torch
    from safetensors.torch import save_file

    from .checkpoint import digest

    bits = search.read_best()
    weights = torch.from_numpy(bits).view(torch.bfloat16)
    target = output / "best"
    target.mkdir()
    manifest = {
        **source,
        "tensors": {},
        "complete": False,
        "optimization": {**result, "events": "../events.jsonl"},
    }
    for index, block in enumerate(layout.blocks):
        filename = f"{index:03d}.safetensors"
        tensor = weights[block.offset : block.offset + block.length].reshape(
            block.shape
        )
        save_file({block.name: tensor}, str(target / filename))
        manifest["tensors"][block.name] = {
            "file": filename,
            "sha256": digest(target / filename),
        }
    manifest["complete"] = True
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


def _checkrelative(search_cls):
    if not all(
        hasattr(search_cls, name)
        for name in ("incumbent", "enable_relative", "tell_relative")
    ):
        raise RuntimeError(
            "Rebuild the BAUX extension with paired-relative minibatch support"
        )
    try:
        parameters = inspect.signature(search_cls.tell_relative).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "Rebuild the BAUX extension with inspectable paired-relative "
            "reject_is_failure keyword support"
        ) from error
    rejection_flag = parameters.get("reject_is_failure")
    if not (
        rejection_flag is not None
        and rejection_flag.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        raise RuntimeError(
            "Rebuild the BAUX extension with paired-relative "
            "reject_is_failure keyword support"
        )


def check_extension(settings: Settings):
    import baux.baux_rust as extension

    if settings.backend == "metal":
        from baux.experimental import MetalParamBlock as ParamBlock
        from baux.experimental import MetalSearchState as SearchState
    else:
        from baux.experimental import ParamBlock, SearchState

    if SearchState is None or not hasattr(SearchState, "read_best"):
        raise RuntimeError("Rebuild the BAUX extension for the selected backend")
    if settings.minibatch_size is not None:
        _checkrelative(SearchState)
    return extension, ParamBlock, SearchState


def build_metadata(
    checkpoint: Path,
    source: dict,
    tokens,
    layout,
    objective,
    settings: Settings,
    *,
    device_name: str,
    extension,
    evaluate,
    native,
    required: int,
):
    is_native = settings.backend != "jax"
    minibatch = settings.minibatch_size is not None
    if not is_native:
        import jax
    return {
        "status": "running",
        "settings": asdict(settings),
        "device": device_name,
        "backend": settings.backend,
        **({} if is_native else {"jax": jax.__version__}),
        "numpy": np.__version__,
        "blocks": layout.describe(),
        "tokens": tokens if isinstance(tokens, dict) else np.asarray(tokens).tolist(),
        "objective": objective,
        "source_revision": source["revision"],
        "source_manifest_sha256": hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest(),
        "extension_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "sampler": settings.sampler,
        "noise_law": NOISE_LAWS[settings.sampler],
        "rounding": "bf16_nearest_even",
        "scale_reference": "initial_checkpoint",
        "controller": settings.controller,
        "reference_seed": settings.reference_seed,
        "reference_storage": "dense_bf16" if settings.sampler == "correlated" else None,
        "reference_bytes": 2 * layout.size if settings.sampler == "correlated" else 0,
        "forward_headroom_bytes": (
            evaluate.workspace_bytes + native.CONTEXT_MARGIN_BYTES
            if is_native
            else FORWARD_HEADROOM_BYTES[settings.sampler]
        ),
        "history_policy": "incumbent_anchor_and_rejected_fifo_reset_on_accept"
        if minibatch
        else "resident_fifo",
        "surrogate_observation": "incumbent_relative_paired_improvement"
        if minibatch
        else "absolute_reward",
        "estimated_memory_bytes": required,
    }


def cli_params():
    return [
        click.Argument(
            ["checkpoint"],
            type=click.Path(exists=True, file_okay=False, path_type=Path),
        ),
        click.Option(
            ["--tokens"],
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="Unpadded token batch or a prepared solution-token objective JSON",
        ),
        click.Option(
            ["--output"],
            type=click.Path(file_okay=False, path_type=Path),
            required=True,
        ),
        click.Option(["--evaluations"], type=int, default=8, show_default=True),
        click.Option(
            ["--backend"],
            type=click.Choice(("native", "jax", "metal")),
            default="jax",
            show_default=True,
        ),
        click.Option(["--candidates"], type=int, default=4, show_default=True),
        click.Option(
            ["--sampler"],
            type=click.Choice(SAMPLERS),
            default="correlated",
            show_default=True,
            help="Correlated Gaussian, independent Gaussian, or legacy independent signs",
        ),
        click.Option(["--history"], type=int, default=2, show_default=True),
        click.Option(["--radius"], type=float, default=0.01, show_default=True),
        click.Option(["--radius-min"], type=float, default=0.0001, show_default=True),
        click.Option(["--radius-max"], type=float, default=0.08, show_default=True),
        click.Option(
            ["--failure-tolerance"],
            type=int,
            default=None,
            help="Counted failures before radius contraction for paired minibatches; baseline TuRBO failure tolerance otherwise. Defaults to 4",
        ),
        click.Option(["--seed"], type=int, default=0, show_default=True),
        click.Option(["--reference-seed"], type=int, default=0, show_default=True),
        click.Option(
            ["--minibatch-size"],
            type=int,
            default=None,
            help=(
                "Problems per paired minibatch; 2 gives four problem forwards per BO step. "
                "Correlated solution objectives only; defaults to 2 for native and metal"
            ),
        ),
        click.Option(["--minibatch-seed"], type=int, default=0, show_default=True),
        click.Option(
            ["--minibatch-refresh"],
            type=click.IntRange(1, 1_000_000),
            default=1,
            show_default=True,
            help=(
                "Reuse each paired minibatch for this many BO rounds. Values above 1 "
                "reduce incumbent forwards but increase minibatch reuse."
            ),
        ),
        click.Option(
            ["--rejection-policy"],
            type=click.Choice(REJECTION_POLICIES),
            default="deterioration",
            show_default=True,
            help="Count only paired deterioration as failure, or all rejections for a legacy ablation",
        ),
        click.Option(
            ["--acceptance-se"],
            type=float,
            default=2.0,
            show_default=True,
            help="Paired improvement screening threshold in estimated standard errors",
        ),
        click.Option(
            ["--paired-epistemic-scale"],
            type=float,
            default=1.0,
            show_default=True,
            help="Paired surrogate loss variance per unit squared tensor-normalized distance",
        ),
        click.Option(
            ["--zero-scale"],
            type=float,
            default=None,
            help="Explicit absolute scale for all-zero tensors",
        ),
    ]
