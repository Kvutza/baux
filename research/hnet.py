#!/usr/bin/env python3
"""Capture unchanged, CPU-executable definitions from the pinned H-Net source.

AST selection avoids importing CUDA-only mixers. It does not substitute their
implementations: DeChunkLayer.forward and the full model are not exercised here.

Academic & Open-Source Attribution:
- Sukjun Hwang, Brandon Wang, and Albert Gu, "Dynamic Chunking for End-to-End
  Hierarchical Sequence Modeling", arXiv:2507.07955 (2025),
  https://github.com/goombalab/hnet (commit 3673fe1217ebeb0d1438c7c71d58d32bdd190ec2).
"""

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REVISION = "3673fe1217ebeb0d1438c7c71d58d32bdd190ec2"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )


def values(tensor):
    return tensor.detach().flatten().tolist()


def routing_case(api, name, hidden, mask, q, k):
    router = api["RoutingModule"](hidden.shape[-1], dtype=torch.float32)
    with torch.no_grad():
        router.q_proj_layer.weight.copy_(q)
        router.k_proj_layer.weight.copy_(k)
    output = router(hidden, mask=mask)
    chunks, _, _, next_mask = api["ChunkLayer"]()(
        hidden, output.boundary_mask, mask=mask
    )
    return {
        "name": name,
        "batch": hidden.shape[0],
        "length": hidden.shape[1],
        "width": hidden.shape[2],
        "hidden": values(hidden),
        "mask": values(mask),
        "q": values(q),
        "k": values(k),
        "probability": values(output.boundary_prob[..., 1]),
        "boundary": values(output.boundary_mask),
        "selected": values(output.selected_probs),
        "chunks": values(chunks),
        "chunk_count": values(output.boundary_mask.sum(dim=1)),
        "chunk_mask": values(next_mask),
    }


def routing_cases(api):
    identity = torch.eye(4)
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    cases = [
        routing_case(
            api, "ties", hidden, torch.ones(1, 5, dtype=torch.bool), identity, identity
        )
    ]
    generator = torch.Generator().manual_seed(1729)
    for width in (4, 33):
        hidden = torch.randn(2, 37, width, generator=generator)
        mask = torch.arange(37)[None, :] < torch.tensor([[37], [23]])
        q = torch.randn(width, width, generator=generator)
        k = torch.randn(width, width, generator=generator)
        cases.append(routing_case(api, f"projected{width}", hidden, mask, q, k))
    return cases


def ste_case(api):
    probability = torch.tensor([1.0, 0.1, 0.5, 0.8], requires_grad=True)
    pair = torch.stack((1 - probability, probability), dim=-1)
    chosen = pair.argmax(dim=-1, keepdim=True)
    selected = pair.gather(-1, chosen)
    dechunk = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 8
    gradient = torch.tensor([[1.0, -2.0, 3.0, 1.0]]).expand(4, -1)
    (dechunk * api["ste_func"](selected) * gradient).sum().backward()
    return {
        "probability": values(probability),
        "boundary": [True, False, False, False],
        "dechunk": values(dechunk),
        "gradient": values(gradient),
        "expected": values(probability.grad),
        "note": "Last token is masked; selected probability still uses raw argmax.",
    }


def recurrent_case(api):
    router = api["RoutingModule"](4, dtype=torch.float32)
    state = router.allocate_inference_cache(1, 8, "cpu", dtype=torch.float32)
    hidden = torch.tensor(
        [
            [
                [1.0, 2.0, 0.0, 1.0],
                [-1.0, -2.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 1.0, 1.0, 0.0],
            ]
        ]
    )
    steps = []
    for token in hidden.unbind(dim=1):
        out = router.step(token[:, None], state)
        steps.append(
            {
                "hidden": values(token),
                "probability": values(out.boundary_prob[..., 1]),
                "boundary": values(out.boundary_mask),
                "selected": values(out.selected_probs),
                "last_hidden": values(state.last_hidden_state),
                "seen": values(state.has_seen_tokens),
            }
        )
    return steps


def half_words(tensor):
    return values(tensor.detach().half().view(torch.int16).to(torch.int32) & 65535)


def flatten(tensors):
    return torch.cat([tensor.flatten() for tensor in tensors])


def bridge_scan(api, route, core):
    """Verify a differentiable FP32 scan against unchanged upstream step()."""
    width = core.shape[-1]
    layer = api["DeChunkLayer"](width, dtype=torch.float32, headdim=1)
    state = layer.allocate_inference_cache(1, 37, "cpu", dtype=torch.float32)
    oracle, outputs, last, cursor = [], [], torch.zeros(1, width), 0
    for t in range(37):
        boundary = route.boundary_mask[:, t]
        selected = core[:, cursor : cursor + 1] if boundary.item() else core[:0, :1]
        with torch.no_grad():
            oracle.append(
                layer.step(selected, boundary, route.boundary_prob[:, t], state).clone()
            )
        if boundary.item():
            p = route.boundary_prob[:, t, 1:2].clamp(1e-4, 1 - 1e-4)
            last = p * core[:, cursor] + (1 - p) * last
            cursor += 1
        outputs.append(last)
    ema = torch.stack(outputs, dim=1)
    torch.testing.assert_close(ema, torch.cat(oracle, dim=1))
    return ema


class Bridge:
    """Half-stored weights, FP32 hierarchy math, differentiable test core."""

    def __init__(self, api, width):
        self.api, self.width = api, width
        self.generator = torch.Generator().manual_seed(927 + width)
        self.hidden = self.draw(1, 37, width).requires_grad_()
        self.router = api["RoutingModule"](width, dtype=torch.float32)
        with torch.no_grad():
            self.router.q_proj_layer.weight.copy_(self.draw(width, width))
            self.router.k_proj_layer.weight.copy_(self.draw(width, width))
        self.residual = (self.draw(width, width) / 8).requires_grad_()
        self.bias = (self.draw(width) / 8).requires_grad_()
        self.parameters = (
            self.router.q_proj_layer.weight,
            self.router.k_proj_layer.weight,
            self.residual,
            self.bias,
        )

    def draw(self, *shape):
        return torch.randn(*shape, generator=self.generator).half().float()

    def forward(self):
        mask = torch.ones(1, 37, dtype=torch.bool)
        self.route = self.router(self.hidden, mask=mask)
        self.chunks, _, _, _ = self.api["ChunkLayer"]()(
            self.hidden, self.route.boundary_mask, mask=mask
        )
        # Multiplication by 0.5 preserves representability in FP16, allowing
        # FP32 core gradients to be checked independently of cast adjoints.
        self.core = self.chunks * 0.5
        self.core.retain_grad()
        self.ema = bridge_scan(self.api, self.route, self.core)
        self.combined = (
            (
                self.ema * self.api["ste_func"](self.route.selected_probs)
                + F.linear(self.hidden, self.residual, self.bias)
            )
            .half()
            .float()
        )
        self.gradient = self.draw(1, 37, self.width) / 16
        (self.combined * self.gradient).sum().backward()
        return self.record()

    def record(self):
        tensors = {
            "hidden": self.hidden,
            "weights": flatten(self.parameters),
            "probability": self.route.boundary_prob[..., 1],
            "boundary": self.route.boundary_mask,
            "chunks": self.chunks,
            "core": self.core,
            "ema": self.ema,
            "combined": self.combined,
            "gradient": self.gradient,
            "core_gradient": self.core.grad,
            "input_gradient": self.hidden.grad,
            "weight_gradient": flatten(
                [parameter.grad for parameter in self.parameters]
            ),
        }
        result = {name: values(tensor) for name, tensor in tensors.items()}
        for name in ("hidden", "weights", "core", "chunks", "combined"):
            result[f"{name}_bits"] = half_words(tensors[name])
        result.update(width=self.width, length=37)
        return result


def tokenizer_cases(api):
    tok = api["ByteTokenizer"]()
    assert (tok.vocab_size, tok.bos_idx, tok.eos_idx) == (256, 254, 255)
    cases = []
    for text in ("", "hello, world!", "é 🦀 λ\n\t0123"):
        raw = tok.encode([text], add_bos=False, add_eos=False)[0]["input_ids"]
        assert tok.decode(raw) == text
        for add_bos in (False, True):
            for add_eos in (False, True):
                ids = tok.encode([text], add_bos=add_bos, add_eos=add_eos)[0][
                    "input_ids"
                ]
                cases.append(
                    {
                        "text": text,
                        "add_bos": add_bos,
                        "add_eos": add_eos,
                        "tokens": [int(x) for x in ids.tolist()],
                    }
                )
    return cases


def ratio_cases(api):
    specs = [
        ("balanced", [1.0, 0.2, 0.8, 0.3], [True, False, True, False], 2.0),
        (
            "dense",
            [1.0, 0.9, 0.7, 0.6, 0.2, 0.8],
            [True, True, True, True, False, True],
            2.5,
        ),
        (
            "sparse",
            [1.0, 0.1, 0.15, 0.2, 0.05, 0.3, 0.1, 0.25],
            [True, False, False, False, False, False, False, False],
            4.0,
        ),
        (
            "fractional",
            [1.0, 0.55, 0.45, 0.75, 0.25, 0.65, 0.35, 0.85],
            [True, True, False, True, False, True, False, True],
            3.5,
        ),
    ]
    cases = []
    for name, prob_list, mask_list, target_ratio in specs:
        p = torch.tensor(prob_list, dtype=torch.float32)
        bmask = torch.tensor(mask_list, dtype=torch.bool)
        pair = torch.stack((1.0 - p, p), dim=-1)
        selected = torch.where(bmask, p, 1.0 - p).unsqueeze(-1)
        out = api["RoutingModuleOutput"](
            boundary_prob=pair,
            boundary_mask=bmask,
            selected_probs=selected,
        )
        loss = api["load_balancing_loss"](out, target_ratio)
        cases.append(
            {
                "name": name,
                "probability": values(p),
                "boundary": values(bmask),
                "selected": values(selected),
                "target_ratio": target_ratio,
                "loss": float(loss.item()),
            }
        )
    return cases


def main(upstream: Path, output: Path):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise click.ClickException(f"Expected upstream {REVISION}, found {revision}")
    if subprocess.check_output(
        ["git", "-C", str(upstream), "diff", "HEAD", "--"], text=True
    ):
        raise click.ClickException("Upstream source has local changes")
    api = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "dataclass": dataclass,
        "__name__": "__main__",
    }
    paths = [
        upstream / "hnet/modules/dc.py",
        upstream / "hnet/models/hnet.py",
        upstream / "hnet/utils/tokenizers.py",
        upstream / "hnet/utils/train.py",
    ]
    definitions(
        paths[0],
        [
            "RoutingModuleOutput",
            "RoutingModuleState",
            "RoutingModule",
            "ChunkLayer",
            "DeChunkState",
            "DeChunkLayer",
        ],
        api,
    )
    definitions(paths[1], ["STE", "ste_func"], api)
    definitions(paths[2], ["ByteTokenizer"], api)
    definitions(paths[3], ["load_balancing_loss"], api)
    result = {
        "revision": revision,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "source_hashes": {
            str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "scope": "routing forward/step, chunk packing, upstream recurrent EMA, STE, byte tokenizer, ratio loss, and FP32 hierarchy adjoints with half-stored weights",
        "full_model_parity": False,
        "routing": routing_cases(api),
        "recurrent": recurrent_case(api),
        "ste": ste_case(api),
        "tokenizer": tokenizer_cases(api),
        "ratio_loss": ratio_cases(api),
        "bridge": [Bridge(api, width).forward() for width in (4, 33)],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    click.echo(
        f"Captured upstream CPU components at {revision}; full model parity is not established."
    )


for _decorator in (
    click.option(
        "--output",
        type=click.Path(dir_okay=False, writable=True, path_type=Path),
        required=True,
    ),
    click.option(
        "--upstream",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True,
    ),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
