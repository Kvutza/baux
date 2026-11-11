"""Compiler-visible MHC4 operations, matching fbt/src/moe.metal.

The parameter-dependent Sinkhorn projection is twenty row/column iterations.
FP16 stores are explicit; prediction and transport accumulate in FP32.
"""

import jax.numpy as jnp
from jax import lax


def dot(left, right):
    return jnp.matmul(left, right, preferred_element_type=jnp.float32)


def rms(value, weight):
    value = value.astype(jnp.float16).astype(jnp.float32)
    scale = lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + 1e-5)
    return (value * scale * weight.astype(jnp.float32)).astype(jnp.float16)


def predict(streams, predictor, bias, control):
    flat = streams.reshape((*streams.shape[:-2], -1))
    values = flat.astype(jnp.float32)
    scale = lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + 1e-5)
    gains = jnp.concatenate(
        (
            jnp.repeat(control[0], 4),
            jnp.repeat(control[1], 16),
            jnp.repeat(control[2], 4),
        )
    ).astype(jnp.float32)
    raw = bias.astype(jnp.float32) + gains * dot(flat, predictor) * scale
    read = 1 / (1 + jnp.exp(-raw[..., :4]))
    read = read / jnp.maximum(jnp.sum(read, axis=-1, keepdims=True), 1e-20)
    transport = raw[..., 4:20].reshape((*raw.shape[:-1], 4, 4))
    transport = jnp.exp(
        jnp.maximum(transport - jnp.max(transport, axis=(-2, -1), keepdims=True), -80)
    )
    for _ in range(20):
        transport = transport / jnp.maximum(
            jnp.sum(transport, axis=-1, keepdims=True), 1e-20
        )
        transport = transport / jnp.maximum(
            jnp.sum(transport, axis=-2, keepdims=True), 1e-20
        )
    amount = jnp.clip(control[3].astype(jnp.float32), 0, 1)
    transport = jnp.eye(4, dtype=jnp.float32) * (1 - amount) + transport * amount
    write = 2 / (1 + jnp.exp(-raw[..., 20:]))
    return read, transport, write


def mix(streams, read, norm):
    value = jnp.zeros(streams.shape[:-2] + (streams.shape[-1],), jnp.float32)
    for index in range(4):
        value = value + read[..., index, None] * streams[..., index, :].astype(
            jnp.float32
        )
    return rms(value, norm)


def update(streams, branch, transport, write):
    value = write[..., None] * branch[..., None, :].astype(jnp.float32)
    for index in range(4):
        value = value + transport[..., index, None] * streams[
            ..., index, None, :
        ].astype(jnp.float32)
    return value.astype(jnp.float16)


def finalize(streams, norm):
    value = jnp.zeros(streams.shape[:-2] + (streams.shape[-1],), jnp.float32)
    for index in range(4):
        value = value + jnp.float32(0.25) * streams[..., index, :].astype(jnp.float32)
    return rms(value, norm)
