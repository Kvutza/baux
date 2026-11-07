"""Compare greedy Metal completions from the original and a saved BO checkpoint."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click
import numpy as np

from . import coding, metal
from .config import Config
from .gen_support import (
    artifact,
    cases_for,
    check_outputs,
    first_candidates,
    greedy,
    render_report,
    write_json,
)
from .layout import Block, Layout
from .native import load_checkpoint

__all__ = [
    "artifact",
    "cases_for",
    "check_command",
    "check_outputs",
    "first_candidates",
    "greedy",
    "main",
    "render_report",
    "run",
    "run_command",
    "write_json",
]


def _checksaved(run_directory, training):
    saved = json.loads((run_directory / "best/manifest.json").read_text())
    optimization = saved.get("optimization")
    if (
        not isinstance(optimization, dict)
        or not optimization
        or any(
            key != "events" and (key not in training or training[key] != value)
            for key, value in optimization.items()
        )
        or not {"final_minibatch", "base_version", "evaluations"} <= optimization.keys()
    ):
        raise ValueError(
            "Saved checkpoint optimization metadata differs from the BO run"
        )


def _checkbounds(training, output, count, max_new_tokens, candidates):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer in [1, 512]")
    if type(candidates) is not int or not 1 <= candidates <= 16:
        raise ValueError("candidates must be an integer in [1, 16]")
    if type(count) is not int or not 1 <= count <= len(training["tokens"]["examples"]):
        raise ValueError("Count must select a nonempty prefix of the corpus")
    if output.exists():
        raise FileExistsError("Refusing to overwrite generation output")


def _checkrun(original, run_directory, output, count, max_new_tokens, candidates):
    from baux.experimental import MetalFlameEvaluator

    if not hasattr(MetalFlameEvaluator, "next_logits"):
        raise ValueError("Rebuild the Metal extension with next_logits support")
    record_path = run_directory / "run.json"
    training = json.loads(record_path.read_text())
    if training.get("status") != "complete" or training.get("backend") != "metal":
        raise ValueError("Generation requires a completed Metal BO run")
    original_hash = coding.sha256((original / "manifest.json").read_bytes())
    if original_hash != training["source_manifest_sha256"]:
        raise ValueError("Original checkpoint does not match the BO run")
    _checksaved(run_directory, training)
    _checkbounds(training, output, count, max_new_tokens, candidates)
    return record_path, training


def _loadinputs(output, training, count, max_new_tokens, config):
    from tokenizers import Tokenizer

    inputs = output.parent / "generation-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = artifact(
        inputs, "tokenizer.json", coding.TOKENIZER_URL, coding.TOKENIZER_SHA256
    )
    dataset_bytes = artifact(
        inputs, "mbpp.jsonl", coding.DATASET_URL, coding.DATASET_SHA256
    )
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vocab = tokenizer.get_vocab_size()
    if vocab != 50277 or tokenizer.token_to_id("<|endoftext|>") != coding.EOS_ID:
        raise ValueError("Tokenizer vocabulary/EOS mismatch")
    rows = [json.loads(line) for line in dataset_bytes.splitlines()]
    cases = cases_for(
        training["tokens"], tokenizer, rows, count, max_new_tokens, config
    )
    return tokenizer, vocab, cases


def _initresult(
    record_path,
    training,
    layout,
    cases,
    *,
    vocab,
    max_new_tokens,
    candidates,
    config,
) -> dict[str, Any]:
    import baux.baux_rust as extension

    return {
        "status": "running",
        "scope": "Fixed TRAIN prefix sanity check; public tests in prompts, not hidden-test accuracy",
        "backend": "metal",
        "device": metal.metal_device(),
        "parameters": layout.size,
        "tokenizers_version": version("tokenizers"),
        "numpy_version": np.__version__,
        "selection": "first_n_corpus_problems_before_generation",
        "decoding": "top_first_token_candidates_then_greedy_full_prefix_recompute",
        "first_candidates": candidates,
        "max_new_tokens": max_new_tokens,
        "tokenizer_vocab": vocab,
        "padded_model_vocab": config.vocab,
        "padded_token_ids_excluded": True,
        "run_sha256": coding.sha256(record_path.read_bytes()),
        "bo_extension_sha256": training["extension_sha256"],
        "generation_extension_sha256": coding.sha256(
            Path(extension.__file__).read_bytes()
        ),
        "generation_source_sha256": coding.sha256(Path(__file__).read_bytes()),
        "tokenizer_sha256": coding.TOKENIZER_SHA256,
        "dataset_sha256": coding.DATASET_SHA256,
        "cases": cases,
        "results": {"original": {}, "optimized": {}},
        "checkpoint_manifest_sha256": {},
        "reference_checks": {},
        "next_logits_parity": {},
        "parity_problem_forwards": 0,
        "reload_problem_forwards": 0,
        "problem_forwards": 0,
        "forward_counts_complete": False,
    }


def _checkrefs(cases, result, report):
    from .codecheck import check_solution

    for case in cases:
        result["reference_checks"][case["id"]] = check_solution(
            case["reference"], case["setup"], case["tests"]
        )
    write_json(report, result)
    if any(
        check["status"] != "passed" for check in result["reference_checks"].values()
    ):
        raise RuntimeError("Reference checks failed; no model generation was started")


def _checkreload(training, weights, result, config):
    from baux.experimental import MetalFlameEvaluator

    check = training["final_minibatch"]
    examples = [training["tokens"]["examples"][i] for i in check["indices"]]
    # Reload comparison is independent of the generation sample.
    reload_engine = MetalFlameEvaluator(
        asdict(config),
        max_tokens=max(len(example["tokens"]) for example in examples),
    )
    losses = reload_engine.losses(
        weights,
        [e["tokens"] for e in examples],
        [e["loss_mask"] for e in examples],
    )
    result["reload_problem_forwards"] += len(examples)
    np.testing.assert_array_equal(losses, check["losses"])
    del reload_engine


def _generatecase(
    engine,
    weights,
    case,
    tokenizer,
    *,
    vocab,
    max_new_tokens,
    candidates,
    config,
    on_forward,
):
    candidate_results = first_candidates(
        engine,
        weights,
        case["prompt_token_ids"],
        vocab=vocab,
        eos_id=coding.EOS_ID,
        max_new_tokens=max_new_tokens,
        context=config.context,
        candidates=candidates,
        on_forward=on_forward,
    )
    for candidate in candidate_results:
        text_ids = candidate["token_ids"]
        if candidate["stop_reason"] == "eos":
            text_ids = text_ids[:-1]
        candidate["text"] = tokenizer.decode(text_ids, skip_special_tokens=False)
        candidate["check"] = {"status": "not_checked"}
    completion = candidate_results[0].copy()
    completion["candidates"] = candidate_results
    completion["problem_forwards"] = sum(
        candidate["problem_forwards"] for candidate in candidate_results
    )
    completion["seconds"] = sum(candidate["seconds"] for candidate in candidate_results)
    return completion


def _runcases(
    label,
    engine,
    weights,
    result,
    report,
    *,
    tokenizer,
    vocab,
    max_new_tokens,
    candidates,
    config,
):
    def on_forward():
        result["problem_forwards"] += 1

    for case in result["cases"]:
        completion = _generatecase(
            engine,
            weights,
            case,
            tokenizer,
            vocab=vocab,
            max_new_tokens=max_new_tokens,
            candidates=candidates,
            config=config,
            on_forward=on_forward,
        )
        result["results"][label][case["id"]] = completion
        write_json(report, result)
        click.echo(
            f"{label} {case['id']}: {len(completion['candidates'])} candidates, "
            f"{completion['problem_forwards']} forwards, "
            f"{completion['stop_reason']}, {completion['check']['status']}"
        )


def _runcheckpoint(
    label,
    directory,
    layout,
    training,
    result,
    report,
    *,
    tokenizer,
    vocab,
    max_new_tokens,
    candidates,
    config,
):
    from baux.experimental import MetalFlameEvaluator

    cases = result["cases"]
    max_tokens = max(len(case["prompt_token_ids"]) for case in cases) + max_new_tokens
    loaded_config, params = load_checkpoint(directory)
    if loaded_config != config:
        raise ValueError("Checkpoint architecture mismatch")
    flat = layout.flatten_torch(params)
    del params
    gc.collect()
    weights = metal.upload_weights(flat)
    del flat
    gc.collect()
    engine = MetalFlameEvaluator(asdict(config), max_tokens=max_tokens)
    if label == "optimized":
        _checkreload(training, weights, result, config)
    probe = cases[0]["prompt_token_ids"]
    next_row = engine.next_logits(weights, probe)
    result["parity_problem_forwards"] += 1
    full_row = engine.logits(weights, probe)[-1]
    result["parity_problem_forwards"] += 1
    np.testing.assert_array_equal(next_row, full_row)
    result["next_logits_parity"][label] = "exact_full_logits_last_row"
    result["checkpoint_manifest_sha256"][label] = coding.sha256(
        (directory / "manifest.json").read_bytes()
    )
    _runcases(
        label,
        engine,
        weights,
        result,
        report,
        tokenizer=tokenizer,
        vocab=vocab,
        max_new_tokens=max_new_tokens,
        candidates=candidates,
        config=config,
    )
    del engine, weights
    gc.collect()


def _execute(
    original,
    run_directory,
    output,
    layout,
    training,
    result,
    *,
    tokenizer,
    vocab,
    max_new_tokens,
    candidates,
    config,
):
    report = output / "generation.json"
    _checkrefs(result["cases"], result, report)
    for label, directory in (
        ("original", original),
        ("optimized", run_directory / "best"),
    ):
        _runcheckpoint(
            label,
            directory,
            layout,
            training,
            result,
            report,
            tokenizer=tokenizer,
            vocab=vocab,
            max_new_tokens=max_new_tokens,
            candidates=candidates,
            config=config,
        )
    result["status"] = "complete"
    assert result["problem_forwards"] == sum(
        value["problem_forwards"]
        for values in result["results"].values()
        for value in values.values()
    )
    result["generation_seconds"] = sum(
        value["seconds"]
        for values in result["results"].values()
        for value in values.values()
    )
    result["reference_checks_valid"] = all(
        check["status"] == "passed" for check in result["reference_checks"].values()
    )
    result["test_passes"] = None
    result["forward_counts_complete"] = True
    write_json(report, result)
    (output / "report.html").write_text(render_report(result))


def run(original, run_directory, output, *, count=8, max_new_tokens=128, candidates=4):
    record_path, training = _checkrun(
        original, run_directory, output, count, max_new_tokens, candidates
    )
    config = Config()
    tokenizer, vocab, cases = _loadinputs(
        output, training, count, max_new_tokens, config
    )
    layout = Layout(
        tuple(
            Block(
                name=b["name"],
                key=b["key"],
                offset=b["offset"],
                shape=tuple(b["shape"]),
                scale=b["scale"],
                weight=b["weight"],
            )
            for b in training["blocks"]
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    result = _initresult(
        record_path,
        training,
        layout,
        cases,
        vocab=vocab,
        max_new_tokens=max_new_tokens,
        candidates=candidates,
        config=config,
    )
    report = output / "generation.json"
    write_json(report, result)
    try:
        _execute(
            original,
            run_directory,
            output,
            layout,
            training,
            result,
            tokenizer=tokenizer,
            vocab=vocab,
            max_new_tokens=max_new_tokens,
            candidates=candidates,
            config=config,
        )
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(report, result)
        raise
    return result


def run_command(original, run_directory, output, count, max_new_tokens, candidates):
    run(
        original,
        run_directory,
        output,
        count=count,
        max_new_tokens=max_new_tokens,
        candidates=candidates,
    )


def check_command(output):
    check_outputs(output)


main = click.Group("main", help=__doc__)
main.add_command(
    click.Command(
        "run",
        callback=run_command,
        params=[
            click.Argument(
                ["original"],
                type=click.Path(exists=True, file_okay=False, path_type=Path),
            ),
            click.Argument(
                ["run_directory"],
                type=click.Path(exists=True, file_okay=False, path_type=Path),
            ),
            click.Option(
                ["--output"],
                required=True,
                type=click.Path(file_okay=False, path_type=Path),
            ),
            click.Option(
                ["--count"], default=8, type=click.IntRange(1, 312), show_default=True
            ),
            click.Option(
                ["--max-new-tokens"],
                default=128,
                type=click.IntRange(1, 512),
                show_default=True,
            ),
            click.Option(
                ["--candidates"],
                default=4,
                type=click.IntRange(1, 16),
                show_default=True,
                help="Distinct first-token paths to complete; candidate 0 remains greedy.",
            ),
        ],
    )
)
main.add_command(
    click.Command(
        "check",
        callback=check_command,
        help="Run inspected completions in OS confinement; no model forwards.",
        params=[
            click.Argument(
                ["output"],
                type=click.Path(exists=True, file_okay=False, path_type=Path),
            )
        ],
    )
)


if __name__ == "__main__":
    main()
