"""Compare the full checkpoint's JAX forward pass with an unfused CPU reference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import click
import jax
import jax.numpy as jnp
import numpy as np
import torch

from . import model, reference
from .config import REVISION


def finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}; parity cannot pass")


def _checkinputs(directory: Path, tokens, require_t4):
    device = jax.devices()[0]
    if require_t4 and (device.platform != "gpu" or "T4" not in device.device_kind):
        raise RuntimeError(f"Expected a CUDA T4, got {device}")
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("Tokens must be an unpadded rank-two integer array")
    if tokens.shape[0] < 1 or tokens.shape[1] < 2:
        raise ValueError("Parity requires a nonempty batch and at least two tokens")
    config, params = model.load_checkpoint(directory)
    if (
        tokens.min() < 0
        or tokens.max() >= config.vocab
        or tokens.shape[1] > config.context
    ):
        raise ValueError("Token IDs or sequence length are outside the model limits")
    return device, config, params, tokens.astype(np.int32)


def _comparelayer(layer, values, layer_trace):
    layer_errors = {}
    for name, expected_value in values.items():
        actual_value = np.asarray(layer_trace[name])
        expected_value = expected_value.numpy()
        finite(f"JAX {layer}/{name}", actual_value)
        finite(f"reference {layer}/{name}", expected_value)
        if name == "experts":
            np.testing.assert_array_equal(actual_value, expected_value)
            continue
        np.testing.assert_allclose(actual_value, expected_value, atol=2e-3, rtol=2e-4)
        layer_errors[name] = float(np.max(np.abs(actual_value - expected_value)))
    return layer_errors


def _comparetaps(taps, traces):
    return {
        layer: _comparelayer(layer, values, traces[layer])
        for layer, values in taps.items()
    }


def _checklosses(actual, expected, tokens, config):
    finite("JAX logits", actual)
    finite("reference logits", expected)
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-4)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
    losses = np.asarray(model.token_loss(jnp.asarray(actual), jnp.asarray(tokens)))
    finite("JAX loss", losses)
    reference_losses = (
        torch.nn.functional.cross_entropy(
            torch.from_numpy(expected[:, :-1].copy()).reshape(-1, config.vocab),
            torch.from_numpy(tokens[:, 1:].copy()).long().reshape(-1),
            reduction="none",
        )
        .numpy()
        .reshape(losses.shape)
    )
    finite("reference loss", reference_losses)
    np.testing.assert_allclose(losses, reference_losses, atol=2e-4, rtol=2e-5)
    return losses


def _hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def _forwardjax(params, tokens, config):
    evaluate = jax.jit(lambda p, t: model.forward(p, t, config, capture=True))
    print("Compiling JAX forward", flush=True)
    start = time.perf_counter()
    actual, traces = jax.block_until_ready(evaluate(params, jnp.asarray(tokens)))
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    actual, traces = jax.block_until_ready(evaluate(params, jnp.asarray(tokens)))
    forward_seconds = time.perf_counter() - start
    return np.asarray(actual), traces, compile_seconds, forward_seconds


def run(directory: Path, tokens, *, require_t4=True):
    device, config, params, tokens = _checkinputs(directory, tokens, require_t4)
    # Keep the independent reference on CPU so it cannot compete for T4 memory.
    cpu = {
        name: torch.from_numpy(np.asarray(value).astype(np.float32))
        for name, value in params.items()
    }
    print("Running independent CPU reference", flush=True)
    expected, taps = reference.forward(
        cpu, torch.from_numpy(tokens).long(), config, capture=True
    )
    del cpu
    actual, traces, compile_seconds, forward_seconds = _forwardjax(
        params, tokens, config
    )
    errors = _comparetaps(taps, traces)
    expected = expected.numpy()
    losses = _checklosses(actual, expected, tokens, config)
    report = {
        "revision": REVISION,
        "device": device.device_kind,
        "platform": device.platform,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "torch": torch.__version__,
        "tokens": tokens.tolist(),
        "evaluation_dtype": "float32",
        "storage_dtype": "bfloat16",
        "reference": "independent unfused PyTorch equations; not released Megatron runtime",
        "compile_and_first_forward_seconds": compile_seconds,
        "forward_seconds": forward_seconds,
        "max_logit_error": float(np.max(np.abs(actual - expected))),
        "mean_next_token_loss": float(losses.mean()),
        "greedy_tokens": actual.argmax(-1).tolist(),
        "layer_max_errors": errors,
        "device_memory": device.memory_stats(),
        "implementation_sha256": _hashes(),
        "passed": True,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def _cli(checkpoint: Path, tokens: Path, output: Path, allow_other_device: bool):
    result = run(
        checkpoint,
        json.loads(tokens.read_text()),
        require_t4=not allow_other_device,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


main = click.Command(
    "main",
    callback=_cli,
    help=__doc__,
    context_settings={"help_option_names": ["-h", "--help"]},
    params=[
        click.Argument(
            ["checkpoint"],
            type=click.Path(exists=True, file_okay=False, path_type=Path),
        ),
        click.Option(
            ["--tokens"],
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            required=True,
            help="JSON array of unpadded token-ID sequences",
        ),
        click.Option(
            ["--output"], type=click.Path(dir_okay=False, path_type=Path), required=True
        ),
        click.Option(["--allow-other-device"], is_flag=True),
    ],
)


if __name__ == "__main__":
    main()
