"""Scoring helpers for the Qwen joint weight/decoder pilot."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codecheck import check_solution
from .config import Config
from .metal import Evaluator

FIM_BOUNDARIES = (
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|repo_name|>",
)


@dataclass(frozen=True)
class DecoderPolicy:
    temperature: float
    top_p: float
    top_k: int = 0


@dataclass
class ScoreTiming:
    elapsed_seconds: float = 0.0
    generation_seconds: float = 0.0
    decode_seconds: float = 0.0
    checker_seconds: float = 0.0
    batches: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0

    def as_dict(self) -> dict[str, float | int]:
        accounted = self.generation_seconds + self.decode_seconds + self.checker_seconds
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "generation_seconds": self.generation_seconds,
            "decode_seconds": self.decode_seconds,
            "checker_seconds": self.checker_seconds,
            "python_overhead_seconds": max(0.0, self.elapsed_seconds - accounted),
            "batches": self.batches,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }


def _decoderpolicy(point) -> DecoderPolicy:
    if len(point) != 2:
        raise ValueError("decoder policy requires temperature and top_p")
    temperature, top_p = (float(value) for value in point)
    if not 0.0 < temperature <= 1.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("decoder policy is outside the valid sampling domain")
    return DecoderPolicy(temperature, top_p)


def _validatepolicy(policy: DecoderPolicy) -> None:
    if not 0.0 <= policy.temperature <= 1.0:
        raise ValueError("decoder temperature must be in [0, 1]")
    if not 0.0 < policy.top_p <= 1.0:
        raise ValueError("decoder top_p must be in (0, 1]")
    if policy.top_k < 0 or policy.top_k > Config().vocab:
        raise ValueError("decoder top_k exceeds the vocabulary")


def _completiontext(tokenizer, token_ids: list[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    boundary = min(
        (position for marker in FIM_BOUNDARIES if (position := text.find(marker)) >= 0),
        default=len(text),
    )
    return text[:boundary]


def _writejson(path: Path, value: dict, *, indent: int | None = None) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=indent, sort_keys=indent is None) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generatebatch(
    evaluator: Evaluator,
    weights,
    batch_indices: list[int],
    batch_prompts: list[list[int]],
    policy: DecoderPolicy,
    *,
    max_new_tokens: int,
    seed: int,
    sample: int,
) -> list[list[int]]:
    if len(batch_indices) == 1:
        if policy.temperature == 0.0:
            return [evaluator.generate(weights, batch_prompts[0], max_new_tokens)]
        sample_seed = (seed + batch_indices[0] + sample * 0x9E3779B97F4A7C15) & (
            (1 << 64) - 1
        )
        return [
            evaluator.sample(
                weights,
                batch_prompts[0],
                max_new_tokens,
                temperature=policy.temperature,
                top_p=policy.top_p,
                top_k=policy.top_k,
                seed=sample_seed,
            )
        ]
    if policy.temperature == 0.0:
        return evaluator.generate_batch(weights, batch_prompts, max_new_tokens)
    seeds = [
        (seed + index + sample * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        for index in batch_indices
    ]
    return evaluator.sample_batch(
        weights,
        batch_prompts,
        max_new_tokens,
        temperature=policy.temperature,
        top_p=policy.top_p,
        top_k=policy.top_k,
        seeds=seeds,
    )


def _scorebatch(
    tokenizer,
    examples: list[dict],
    batch_indices: list[int],
    generated_batch: list[list[int]],
    batch_prompts: list[list[int]],
    *,
    start: int,
    sample: int,
    score_samples: list[list[float]],
    details: list[dict],
    timing: ScoreTiming | None,
) -> None:
    for local, (index, generated, prompt) in enumerate(
        zip(batch_indices, generated_batch, batch_prompts)
    ):
        example = examples[index]
        decode_started = time.perf_counter()
        text = _completiontext(tokenizer, generated[len(prompt) :])
        if timing is not None:
            timing.decode_seconds += time.perf_counter() - decode_started
        checker_started = time.perf_counter()
        check = check_solution(text, example["setup"], example["tests"])
        if timing is not None:
            timing.checker_seconds += time.perf_counter() - checker_started
        if check["status"] == "sandbox_unavailable":
            raise RuntimeError("MBPP checker sandbox is unavailable")
        score = check["tests_passed"] / check["tests_total"]
        score_samples[start + local].append(score)
        details[start + local]["samples"].append(
            {"sample": sample, "score": score, "check": check, "text": text}
        )


def _runbatch(
    evaluator: Evaluator,
    tokenizer,
    examples: list[dict],
    prompts: list[list[int]],
    weights,
    *,
    batch_indices: list[int],
    start: int,
    sample: int,
    cfg: dict[str, Any],
    score_samples: list[list[float]],
    details: list[dict],
) -> None:
    batch_prompts = [prompts[index] for index in batch_indices]
    generation_started = time.perf_counter()
    initial = cfg["initial_generated"]
    if initial is not None and sample == 0 and start == 0:
        generated_batch = initial
    else:
        generated_batch = _generatebatch(
            evaluator,
            weights,
            batch_indices,
            batch_prompts,
            cfg["policy"],
            max_new_tokens=cfg["max_new_tokens"],
            seed=cfg["seed"],
            sample=sample,
        )
    timing = cfg["timing"]
    if timing is not None:
        timing.generation_seconds += time.perf_counter() - generation_started
        timing.batches += 1
        timing.prompt_tokens += sum(len(prompt) for prompt in batch_prompts)
        timing.generated_tokens += sum(
            max(0, len(generated) - len(prompt))
            for generated, prompt in zip(generated_batch, batch_prompts, strict=True)
        )
    _scorebatch(
        tokenizer,
        examples,
        batch_indices,
        generated_batch,
        batch_prompts,
        start=start,
        sample=sample,
        score_samples=score_samples,
        details=details,
        timing=timing,
    )


def _unpackargs(positional: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    names = (
        "indices",
        "policy",
        "max_new_tokens",
        "seed",
        "samples",
        "batch_size",
        "timing",
        "initial_generated",
    )
    cfg = {"timing": None, "initial_generated": None, **kwargs}
    for name, value in zip(names, positional):
        cfg[name] = value
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    return cfg


def _score(
    evaluator: Evaluator,
    tokenizer,
    examples: list[dict],
    prompts: list[list[int]],
    weights,
    *positional: Any,
    **kwargs: Any,
) -> tuple[list[float], list[dict]]:
    cfg = _unpackargs(positional, kwargs)
    indices = cfg["indices"]
    samples = cfg["samples"]
    batch_size = cfg["batch_size"]
    started = time.perf_counter()
    score_samples: list[list[float]] = [[] for _ in indices]
    details: list[dict] = [
        {"id": examples[index]["id"], "samples": []} for index in indices
    ]
    for sample in range(samples):
        for start in range(0, len(indices), batch_size):
            _runbatch(
                evaluator,
                tokenizer,
                examples,
                prompts,
                weights,
                batch_indices=indices[start : start + batch_size],
                start=start,
                sample=sample,
                cfg=cfg,
                score_samples=score_samples,
                details=details,
            )
    if cfg["timing"] is not None:
        cfg["timing"].elapsed_seconds = time.perf_counter() - started
    per_problem = [sum(values) / samples for values in score_samples]
    return per_problem, details
