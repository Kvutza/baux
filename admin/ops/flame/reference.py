"""Independent, unfused PyTorch comparator for the pinned Megatron equations.

This does not run Megatron/TransformerEngine, and is not a substitute for testing
against that runtime. It provides a second implementation and per-block taps.
"""

import math

import torch
import torch.nn.functional as F


def _norm(x, weight, config):
    return F.rms_norm(x, (config.width,), weight.float(), config.epsilon)


def _dense(x, first, second):
    gate, up = F.linear(x, first.float()).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, second.float())


def _rope(sequence, dim, base, device):
    positions = torch.arange(sequence, device=device).float()
    frequencies = base ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    angles = torch.outer(positions, frequencies)
    return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]


def _rotate(item, rope):
    cosine, sine = rope
    left, right = item.chunk(2, dim=-1)
    return torch.cat(
        (left * cosine - right * sine, right * cosine + left * sine), dim=-1
    ).transpose(1, 2)


def _attention(x, params, prefix, config, rope):
    dim = config.width // config.heads
    normalized = _norm(
        x, params[prefix + "self_attention.linear_qkv.layer_norm_weight"], config
    )
    packed = F.linear(
        normalized, params[prefix + "self_attention.linear_qkv.weight"].float()
    ).reshape(x.shape[0], x.shape[1], config.heads, 3 * dim)
    q, k, v = packed.split(dim, dim=-1)
    scores = (_rotate(q, rope) @ _rotate(k, rope).transpose(-1, -2)) / math.sqrt(dim)
    mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
    scores.masked_fill_(mask, -torch.inf)
    attended = (scores.softmax(-1) @ v.transpose(1, 2)).transpose(1, 2).reshape_as(x)
    return F.linear(
        attended, params[prefix + "self_attention.linear_proj.weight"].float()
    )


def _routed(flat, ids, probs, params, prefix, config):
    routed = torch.zeros_like(flat)
    for expert in range(config.experts):
        rows, slots = torch.where(ids.reshape(-1, config.top_k) == expert)
        if rows.numel():
            output = _dense(
                flat[rows],
                params[prefix + "mlp.experts.experts.linear_fc1.weight"][expert],
                params[prefix + "mlp.experts.experts.linear_fc2.weight"][expert],
            )
            routed.index_add_(
                0,
                rows,
                output * probs.reshape(-1, config.top_k)[rows, slots, None],
            )
    return routed


def _moe(x, params, prefix, config):
    normalized = _norm(x, params[prefix + "pre_mlp_layernorm.weight"], config)
    logits = F.linear(normalized, params[prefix + "mlp.router.weight"].float())
    scores = logits.softmax(-1)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., : config.top_k]
    probs = scores.gather(-1, ids)
    shared = _dense(
        normalized,
        params[prefix + "mlp.shared_experts.linear_fc1.weight"],
        params[prefix + "mlp.shared_experts.linear_fc2.weight"],
    )
    # Dispatch by expert, independently of the JAX token-by-token path.
    routed = _routed(
        normalized.reshape(-1, config.width), ids, probs, params, prefix, config
    )
    return (
        shared + routed.reshape_as(x),
        {"router_logits": logits, "routing_weights": probs, "experts": ids},
    )


@torch.inference_mode()
def forward(params, tokens, config, *, capture=False):
    x = F.embedding(tokens, params["embedding.word_embeddings.weight"].float())
    rope = _rope(x.shape[1], config.width // config.heads, config.rope_base, x.device)
    traces = {}
    for layer in range(config.layers):
        prefix = f"decoder.layers.{layer}."
        attended = _attention(x, params, prefix, config, rope)
        x = x + attended
        trace = {"attention": attended}
        if layer == 0:
            normalized = _norm(
                x, params[prefix + "mlp.linear_fc1.layer_norm_weight"], config
            )
            update = _dense(
                normalized,
                params[prefix + "mlp.linear_fc1.weight"],
                params[prefix + "mlp.linear_fc2.weight"],
            )
        else:
            update, extra = _moe(x, params, prefix, config)
            trace.update(extra)
        x = x + update
        if capture:
            traces[str(layer)] = dict(trace, output=x)
    logits = F.linear(
        _norm(x, params["decoder.final_layernorm.weight"], config),
        params["output_layer.weight"].float(),
    )
    return (logits, traces) if capture else logits
