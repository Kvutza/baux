"""Validated solution-token objectives for Qwen optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

FORMAT = "baux.solution_tokens.v1"


def _checkexample(example, seen, config):
    if not isinstance(example, dict):
        raise TypeError("Each objective example must be an object")
    key = example.get("id")
    if not isinstance(key, str) or not key or key in seen:
        raise ValueError("Objective example IDs must be nonempty and unique")
    seen.add(key)
    tokens, mask = example.get("tokens"), example.get("loss_mask")
    if (
        not isinstance(tokens, list)
        or not 2 <= len(tokens) <= config.context
        or any(type(t) is not int or not 0 <= t < config.vocab for t in tokens)
    ):
        raise ValueError(f"Invalid token sequence for {key}")
    if (
        not isinstance(mask, list)
        or len(mask) != len(tokens)
        or any(type(value) is not bool for value in mask)
        or mask[0]
        or not mask[-1]
        or any(mask[i] and not mask[i + 1] for i in range(len(mask) - 1))
    ):
        raise ValueError(
            f"Expected an unscored prompt then a scored solution for {key}"
        )
    return len(tokens), sum(mask)


@dataclass(frozen=True)
class SolutionObjective:
    tokens: np.ndarray
    mask: np.ndarray
    metadata: dict

    @classmethod
    def parse(cls, document, config):
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            raise ValueError(f"Expected objective format {FORMAT}")
        examples = document.get("examples")
        if not isinstance(examples, list) or not examples:
            raise ValueError("Objective requires a nonempty examples list")
        if not isinstance(document.get("provenance"), dict):
            raise TypeError("Objective requires provenance metadata")
        seen = set()
        lengths, counts = [], []
        for example in examples:
            length, count = _checkexample(example, seen, config)
            lengths.append(length)
            counts.append(count)
        padded = np.zeros((len(examples), max(lengths)), dtype=np.int32)
        mask = np.zeros(padded.shape, dtype=np.bool_)
        for row, (example, length) in enumerate(zip(examples, lengths, strict=True)):
            padded[row, :length] = example["tokens"]
            mask[row, :length] = example["loss_mask"]
        serialized = json.dumps(document, sort_keys=True, allow_nan=False).encode()
        metadata = {
            "kind": "solution_token_cross_entropy",
            "normalization": "total_scored_tokens",
            "sha256": hashlib.sha256(serialized).hexdigest(),
            "examples": len(examples),
            "example_ids": [example["id"] for example in examples],
            "input_tokens": sum(lengths),
            "scored_tokens": sum(counts),
            "sequence_lengths": lengths,
            "solution_tokens_per_example": counts,
            "padded_sequence_length": max(lengths),
            "microbatch_size": 1,
            "device_input_bytes": padded.nbytes + mask[:, 1:].nbytes,
            "provenance": document["provenance"],
        }
        return cls(padded, mask, metadata)
