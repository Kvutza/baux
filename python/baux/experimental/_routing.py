"""Deterministic top-three routing and the production FP16 SwiGLU.

References: fbt/src/routing.metal and routing_tensorops.metal. The graph
evaluates experts in a batch; it preserves token-choice routing and has no
capacity limit. Physical dropless packing is a backend scheduling concern.
"""

import jax.numpy as jnp

from ._hyper import dot


def select(scores, candidates, count):
    """Descending score, ascending candidate ID; -1 denotes an empty slot."""
    chosen = []
    values = []
    for _ in range(count):
        maximum = jnp.max(scores, axis=-1, keepdims=True)
        eligible = (scores == maximum) & (scores > -jnp.inf) & (candidates >= 0)
        best = jnp.min(
            jnp.where(eligible, candidates, jnp.int32(2147483647)),
            axis=-1,
            keepdims=True,
        )
        valid = best != 2147483647
        chosen.append(jnp.where(valid, best, -1))
        values.append(maximum)
        scores = jnp.where(candidates == best, -jnp.inf, scores)
    return jnp.concatenate(chosen, axis=-1), jnp.concatenate(values, axis=-1)


def sigmoid(value):
    e = jnp.exp(-jnp.abs(value))
    return jnp.where(value >= 0, 1 / (1 + e), e / (1 + e))


def forward(value, router, gate, down):
    shape = value.shape
    value = value.reshape((-1, shape[-1]))
    logits = dot(value, router).astype(jnp.float16).astype(jnp.float32)
    ids = jnp.broadcast_to(jnp.arange(logits.shape[-1], dtype=jnp.int32), logits.shape)
    routes, scores = select(logits, ids, 3)
    probabilities = sigmoid(scores)
    probabilities = (
        probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    ).astype(jnp.float16)
    # Expert is a batch axis. Preserve half rounding after each activation op.
    projected = dot(value[None, ...], gate).astype(jnp.float16)
    width = projected.shape[-1] // 2
    activation = (projected[..., :width] * sigmoid(projected[..., :width])) * projected[
        ..., width:
    ]
    outputs = dot(activation, down).astype(jnp.float16)
    result = outputs[0].astype(jnp.float32)
    rows = jnp.arange(value.shape[0], dtype=jnp.int32)
    for route in range(3):
        branch = outputs[routes[:, route] + 1, rows].astype(jnp.float32)
        result = result + probabilities[:, route, None].astype(jnp.float32) * branch
    return result.astype(jnp.float16).reshape(shape)
