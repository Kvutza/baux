"""Full PISA binary refinement as ordinary StableHLO tensor operations.

Reference implementation: fbt/src/pisa1/parity.rs and pisa1.metal. Intermediate
scores are grouped-head log-sum-exp over child summaries, leaf scores over all
64 keys. First, previous and current blocks are forced, with lower-ID ties.
"""

import jax.numpy as jnp
import numpy as np

from ._routing import select


def rope(qkv):
    context = qkv.shape[1]
    base = 10000.0 * max(1.0, context / 8192) ** 2
    phase = np.arange(context, dtype=np.float64)[:, None] * np.power(
        base, -2 * np.arange(32, dtype=np.float64)[None, :] / 64
    )
    cosine = jnp.asarray(np.cos(phase).astype(np.float32))[None, :, None, :]
    sine = jnp.asarray(np.sin(phase).astype(np.float32))[None, :, None, :]
    heads = qkv[..., :576].reshape((*qkv.shape[:-1], 9, 64)).astype(jnp.float32)
    left, right = heads[..., :32], heads[..., 32:]
    rotated = jnp.concatenate(
        (left * cosine - right * sine, right * cosine + left * sine), axis=-1
    )
    return jnp.concatenate(
        (rotated.astype(jnp.float16).reshape((*qkv.shape[:-1], 576)), qkv[..., 576:]),
        axis=-1,
    )


def lse(logits):
    maximum = jnp.max(logits, axis=-1, keepdims=True)
    return (
        maximum + jnp.log(jnp.sum(jnp.exp(logits - maximum), axis=-1, keepdims=True))
    )[..., 0]


def contract(query, key):
    return jnp.einsum(
        "bthd,btcsd->bthcs", query, key, preferred_element_type=jnp.float32
    ) * jnp.float32(0.125)


def blocks(query, key, positions=None):
    batch, context = key.shape[:2]
    leaves = context // 64
    keys = key.reshape((batch, leaves, 64, 64))
    summaries = [jnp.mean(keys.astype(jnp.float32), axis=2).astype(jnp.float16)]
    while summaries[-1].shape[1] > 1:
        previous = summaries[-1].astype(jnp.float32)
        summaries.append(
            ((previous[:, ::2] + previous[:, 1::2]) * jnp.float32(0.5)).astype(
                jnp.float16
            )
        )
    level = max(0, (leaves - 1).bit_length() - 4)
    rows = query.shape[1]
    candidates = jnp.broadcast_to(
        jnp.arange(leaves >> level, dtype=jnp.int32), (batch, rows, leaves >> level)
    )
    if positions is None:
        positions = jnp.arange(rows, dtype=jnp.int32)
    current = (positions // 64)[None, :, None]
    previous = jnp.maximum(current - 1, 0)
    sequences = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
    for depth in range(level, -1, -1):
        begin = candidates * (1 << depth)
        end = begin + (1 << depth) - 1
        forced = (
            (begin == 0)
            | ((begin <= previous) & (previous <= end))
            | ((begin <= current) & (current <= end))
        )
        if depth:
            children = jnp.stack((candidates * 2, candidates * 2 + 1), axis=-1)
            vectors = summaries[depth - 1][
                sequences[..., None], jnp.maximum(children, 0)
            ]
        else:
            vectors = keys[sequences, jnp.maximum(candidates, 0)]
        score = jnp.sum(lse(contract(query, vectors)), axis=2)
        score = jnp.where(
            candidates < 0,
            -jnp.inf,
            jnp.where(forced, jnp.inf, jnp.where(end < current, score, -jnp.inf)),
        )
        chosen, _ = select(score, candidates, 8)
        if depth:
            candidates = jnp.stack((chosen * 2, chosen * 2 + 1), axis=-1)
            candidates = jnp.where(chosen[..., None] >= 0, candidates, -1).reshape(
                (batch, rows, 16)
            )
    return chosen


def forward(qkv):
    batch, context = qkv.shape[:2]
    query = qkv[..., :512].reshape((batch, context, 8, 64))
    key, value = qkv[..., 512:576], qkv[..., 576:]
    selected = blocks(query, key)
    sequences = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
    positions = jnp.arange(context, dtype=jnp.int32)[None, :, None]
    maximum = jnp.full((batch, context, 8), -jnp.inf, dtype=jnp.float32)
    total = jnp.zeros_like(maximum)
    result = jnp.zeros((batch, context, 8, 64), jnp.float32)
    for slot in range(8):
        block = selected[..., slot, None]
        tokens = jnp.maximum(block, 0) * 64 + jnp.arange(64, dtype=jnp.int32)
        keys, values = key[sequences, tokens], value[sequences, tokens]
        logits = contract(query, keys[..., None, :, :])[..., 0, :]
        valid = ((block >= 0) & (tokens <= positions))[:, :, None, :]
        logits = jnp.where(valid, logits, -jnp.inf)
        nextmax = jnp.maximum(maximum, jnp.max(logits, axis=-1))
        alpha = jnp.exp(maximum - nextmax)
        probability = jnp.where(valid, jnp.exp(logits - nextmax[..., None]), 0)
        total = total * alpha + jnp.sum(probability, axis=-1)
        contribution = jnp.einsum(
            "bths,btsd->bthd",
            probability.astype(jnp.float16),
            values,
            preferred_element_type=jnp.float32,
        )
        result = result * alpha[..., None] + contribution
        maximum = nextmax
    return (
        (result / total[..., None]).astype(jnp.float16).reshape((batch, context, 512))
    )
