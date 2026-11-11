"""The production MHC4/FBT computation expressed as a portable tensor program.

JAX is used for tracing only. ``compile`` executes the emitted StableHLO using
the selected Rust-owned runtime. Packed weights and layer visits come from the
Rust model, and every operation is visible to the compiler: no custom call to
the Metal forward is emitted. See fbt/src/scorer.rs for the reference schedule.
"""

from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Any

import numpy as np

from .program import TensorSpec


@dataclass(frozen=True)
class Fbt:
    """Express experimental MHC4 FBT as a compiler-visible tensor program.

    Parameters
    ----------
    context : int, optional
        Tokens per sequence, a power of two of at least 64. Default is 4096.
    batch : int, optional
        Positive number of independent sequences. Default is 2.

    Raises
    ------
    ValueError
        A dimension violates the static shape constraints.

    Notes
    -----
    Uses all 265,623,320 FP16 parameters, full PISA, seven layer visits and
    the 262,144-entry tied vocabulary. Rust supplies the canonical parameter
    layout and visit order. JAX traces the graph; the selected native runtime
    compiles and executes it. All experts are evaluated before routing.

    Full-forward equivalence with the native Metal model remains unresolved.
    Four runtime paths have executed a 128-token prefix; 4096-token export is
    checked, but longer compiled-graph executions and latency are unvalidated.
    See ``research/fbt-graph-2026-11-11.toml`` and the ``fbt-research`` Buck target.

    Examples
    --------
    Export a graph without initializing or transferring model weights:

    >>> from baux.experimental import Fbt
    >>> graph = Fbt(context=128, batch=1)
    >>> graph.inputs[0].shape
    (265623320,)
    >>> source, outputs = graph.lower()
    >>> outputs[0].shape
    (1, 128)

    See Also
    --------
    baux.experimental.compile : Compile the graph with a native runtime.
    baux.experimental.ResidentModel : Initialize the native Metal model.
    """

    context: int = 4096
    batch: int = 2

    def __post_init__(self):
        if (
            type(self.context) is not int
            or self.context < 64
            or self.context & (self.context - 1)
        ):
            raise ValueError("FBT context must be a power of two of at least 64")
        if type(self.batch) is not int or self.batch < 1:
            raise ValueError("FBT batch must be a positive integer")

    @cached_property
    def layout(self):
        """Return canonical named parameter views from Rust.

        Returns
        -------
        tuple of (str, int, tuple of int)
            Parameter name, element offset and tensor shape. Offsets count
            FP16 elements, not bytes. Reading metadata transfers no weights.
        """
        from .. import baux_rust

        return tuple(
            (name, offset, tuple(shape))
            for name, offset, shape in baux_rust.ResidentModel.layout()
        )

    @cached_property
    def visits(self):
        """Weight-tied layer execution order from the Rust model."""
        from .. import baux_rust

        return tuple(baux_rust.ResidentModel.visits())

    @property
    def inputs(self):
        """Input specification accepted by ``baux.experimental.compile``."""
        _, offset, shape = self.layout[-1]
        sequence = (self.batch, self.context)
        return (
            TensorSpec((offset + prod(shape),), "float16"),
            TensorSpec(sequence, "uint32"),
            TensorSpec(sequence, "uint32"),
            TensorSpec(sequence, "bool"),
        )

    def lower(self):
        """Trace the model into StableHLO for native language callers.

        Returns
        -------
        source : str
            Static StableHLO module with a ``main`` entry and explicit weights.
        outputs : tuple of TensorSpec
            FP32 token losses of shape ``(batch, context)`` and sequence means
            of shape ``(batch,)``.

        Notes
        -----
        This synchronous operation traces only; it does not execute the model,
        initialize parameters or upload buffers. Requires JAX and the native
        BAUX extension for model metadata.
        """
        from ._trace import lower

        return lower(self, self.inputs)

    def __call__(self, arguments: Any, *, inspect: bool = False):
        """Trace or evaluate one forward tensor program.

        Parameters
        ----------
        arguments : tuple of array-like
            Packed FP16 weights with shape ``(265623320,)``, uint32 tokens and
            labels with shape ``(batch, context)``, and a bool mask of the same
            sequence shape. Callers must supply vocabulary IDs in [0, 262144).
        inspect : bool, optional
            Return per-visit tensor snapshots instead of readout losses.

        Returns
        -------
        tuple of arrays or dict of str to array
            Normally unmasked FP32 token losses and masked sequence means.
            An entirely masked sequence has a NaN mean. Inspection returns
            streams, normalized MoE inputs, router scores and selected IDs.

        Raises
        ------
        ValueError
            Input count, shape or dtype differs from ``inputs``.

        Notes
        -----
        Inputs are not mutated. Native resident execution uses ``compile``
        and its explicit ``put``, ``submit``/``wait`` and ``get`` operations.
        """
        import jax.numpy as jnp

        from . import _hyper, _routing

        packed, tokens, labels, mask = arguments
        for value, spec in zip(arguments, self.inputs, strict=True):
            if tuple(value.shape) != spec.shape or str(value.dtype) != spec.dtype:
                raise ValueError(f"FBT input differs from {spec}")
        weights = {
            name: packed[offset : offset + prod(shape)].reshape(shape)
            for name, offset, shape in self.layout
        }
        embedding = weights["embedding_readout"]
        hidden = jnp.take(embedding, tokens, axis=1, mode="clip").transpose((1, 2, 0))
        streams = jnp.repeat(hidden[..., None, :], 4, axis=-2)
        snapshots = {}
        for step, layer in enumerate(self.visits):
            streams = self._attention(streams, weights, layer)
            read, transport, write = _hyper.predict(
                streams,
                weights["mhc_moe_predictor"][layer],
                weights["mhc_moe_bias"][layer],
                weights["mhc_moe_control"][layer],
            )
            normalized = _hyper.mix(streams, read, weights["ffn_norm"][layer])
            branch = _routing.forward(
                normalized,
                weights["router"][layer],
                weights["expert_gate_up"][layer],
                weights["expert_down"][layer],
            )
            streams = _hyper.update(streams, branch, transport, write)
            if inspect:
                logits = _hyper.dot(
                    normalized.reshape((-1, 512)), weights["router"][layer]
                ).astype(jnp.float16)
                candidates = jnp.broadcast_to(
                    jnp.arange(logits.shape[-1], dtype=jnp.int32), logits.shape
                )
                experts, _ = _routing.select(logits.astype(jnp.float32), candidates, 3)
                snapshots[f"{step}/streams"] = streams
                snapshots[f"{step}/input"] = normalized
                snapshots[f"{step}/scores"] = logits
                snapshots[f"{step}/experts"] = experts.astype(jnp.uint32)
        if inspect:
            return snapshots
        hidden = _hyper.finalize(streams, weights["final_norm"])
        losses = readout(hidden, embedding, labels)
        counts = jnp.sum(mask.astype(jnp.float32), axis=1)
        means = jnp.sum(jnp.where(mask, losses, 0), axis=1) / counts
        return losses, means

    def _attention(self, streams, weights, layer):
        from . import _hyper, _pisa

        read, transport, write = _hyper.predict(
            streams,
            weights["mhc_attention_predictor"][layer],
            weights["mhc_attention_bias"][layer],
            weights["mhc_attention_control"][layer],
        )
        normalized = _hyper.mix(streams, read, weights["attention_norm"][layer])
        qkv = _hyper.dot(normalized, weights["qkv"][layer]).astype(np.float16)
        attention = _pisa.forward(_pisa.rope(qkv))
        branch = _hyper.dot(attention, weights["attention_output"][layer]).astype(
            np.float16
        )
        return _hyper.update(streams, branch, transport, write)


def readout(hidden, embedding, labels):
    """Full vocabulary loss with tiled, non-peak log-sum-exp statistics.

    Reference: fbt/src/loss_tensorops.metal. Keep the non-peak tail separate
    until log1p, including when the correct class dominates the vocabulary.
    """
    import jax.numpy as jnp

    from ._hyper import dot

    maximum = jnp.full(labels.shape, -jnp.inf, dtype=jnp.float32)
    tail = jnp.zeros_like(maximum)
    target = jnp.zeros_like(maximum)
    tile = min(4096, embedding.shape[-1])
    for start in range(0, embedding.shape[-1], tile):
        logits = (
            dot(hidden, embedding[:, start : start + tile])
            .astype(jnp.float16)
            .astype(jnp.float32)
        )
        peak = jnp.max(logits, axis=-1)
        indices = jnp.arange(logits.shape[-1], dtype=jnp.int32)
        first = jnp.min(
            jnp.where(logits == peak[..., None], indices, logits.shape[-1]), axis=-1
        )
        mass = jnp.sum(
            jnp.where(
                indices == first[..., None], 0, jnp.exp(logits - peak[..., None])
            ),
            axis=-1,
        )
        higher = peak > maximum
        scale = jnp.exp(jnp.minimum(peak, maximum) - jnp.maximum(peak, maximum))
        tail = jnp.where(higher, mass + (tail + 1) * scale, tail + (mass + 1) * scale)
        maximum = jnp.maximum(maximum, peak)
        local = jnp.clip(labels.astype(jnp.int32) - start, 0, logits.shape[-1] - 1)
        picked = jnp.take_along_axis(logits, local[..., None], axis=-1)[..., 0]
        target = target + jnp.where(
            (labels >= start) & (labels < start + tile), picked, 0
        )
    return (maximum - target) + jnp.log1p(tail)
