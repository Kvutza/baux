"""Decoding, case preparation, HTML rendering, and post-generation checking."""

from __future__ import annotations

import html
import json
import time

import click
import numpy as np

from . import coding
from .objective import SolutionObjective


def _checkdecode(prompt, vocab, eos_id, max_new_tokens, context, candidates=1):
    for name, value in (
        ("vocab", vocab),
        ("context", context),
        ("max_new_tokens", max_new_tokens),
        ("candidates", candidates),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(eos_id) is not int or not 0 <= eos_id < vocab:
        raise ValueError("EOS must be in the tokenizer vocabulary")
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or not 0 <= token < vocab for token in prompt)
    ):
        raise ValueError("Prompt must be a nonempty list of valid token IDs")
    if len(prompt) + max_new_tokens > context:
        raise ValueError("Prompt plus generation budget exceeds context; no truncation")


def _checkedlogits(engine, weights, prefix, vocab, on_forward):
    logits = np.asarray(engine.next_logits(weights, prefix))
    if on_forward is not None:
        on_forward()
    if logits.ndim != 1 or logits.size < vocab or logits.dtype != np.float32:
        raise ValueError("Expected one FP32 vocabulary vector from Metal")
    if not np.isfinite(logits).all():
        raise ValueError("Cannot generate from nonfinite logits")
    return logits[:vocab]


def greedy(
    engine, weights, prompt, *, vocab, eos_id, max_new_tokens, context, on_forward=None
):
    """Feed only prompt and already generated tokens; ties choose the lowest ID."""
    _checkdecode(prompt, vocab, eos_id, max_new_tokens, context)
    prefix = prompt.copy()
    generated = []
    stop = "token_limit"
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = _checkedlogits(engine, weights, prefix, vocab, on_forward)
        # The model's embedding vocabulary is padded beyond the tokenizer's IDs.
        token = int(np.argmax(logits))
        generated.append(token)
        if token == eos_id:
            stop = "eos"
            break
        prefix.append(token)
    return {
        "token_ids": generated,
        "stop_reason": stop,
        "problem_forwards": len(generated),
        "seconds": time.perf_counter() - started,
    }


def _decodebranch(
    engine,
    weights,
    prompt,
    token,
    logits,
    started,
    *,
    vocab,
    eos_id,
    max_new_tokens,
    on_forward,
):
    prefix = prompt.copy()
    prefix.append(token)
    generated = [token]
    score = float(logits[token] - np.logaddexp.reduce(logits))
    stop = "eos" if token == eos_id else "token_limit"
    while stop != "eos" and len(generated) < max_new_tokens:
        logits = _checkedlogits(engine, weights, prefix, vocab, on_forward)
        normalizer = float(np.logaddexp.reduce(logits))
        token = int(np.argmax(logits))
        score += float(logits[token] - normalizer)
        generated.append(token)
        if token == eos_id:
            stop = "eos"
        else:
            prefix.append(token)
    return {
        "token_ids": generated,
        "stop_reason": stop,
        "problem_forwards": len(generated),
        "seconds": time.perf_counter() - started,
        "log_probability": score,
    }


def first_candidates(
    engine,
    weights,
    prompt,
    *,
    vocab,
    eos_id,
    max_new_tokens,
    context,
    candidates,
    on_forward=None,
):
    """Complete distinct top-first-token paths without executing them.

    Candidate zero is exactly the existing greedy decode. Branching only at the
    first token keeps the extra work bounded while recovering nearby code starts
    that greedy decoding can discard immediately.
    """
    _checkdecode(prompt, vocab, eos_id, max_new_tokens, context, candidates)
    first_started = time.perf_counter()
    first = _checkedlogits(engine, weights, prompt.copy(), vocab, on_forward)
    first_order = np.argsort(-first, kind="stable")[:candidates]
    results = []
    for index, first_token in enumerate(first_order.tolist()):
        branch_started = first_started if index == 0 else time.perf_counter()
        logits = (
            first
            if index == 0
            else _checkedlogits(engine, weights, prompt.copy(), vocab, on_forward)
        )
        results.append(
            _decodebranch(
                engine,
                weights,
                prompt,
                int(first_token),
                logits,
                branch_started,
                vocab=vocab,
                eos_id=eos_id,
                max_new_tokens=max_new_tokens,
                on_forward=on_forward,
            )
        )
    return results


def artifact(directory, name, url, expected):
    path = directory / name
    data = path.read_bytes() if path.exists() else coding.read_public(url)
    if coding.sha256(data) != expected:
        raise ValueError(f"Pinned artifact checksum mismatch: {name}")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(data)
    return data


def _indexdataset(dataset):
    by_id = {}
    for row in dataset:
        if row["task_id"] in by_id:
            raise ValueError("Duplicate dataset task ID")
        by_id[row["task_id"]] = row
    return by_id


def cases_for(document, tokenizer, dataset, count, max_new_tokens, config):
    objective = SolutionObjective.parse(document, config)
    if type(count) is not int or not 1 <= count <= len(document["examples"]):
        raise ValueError("Count must select a nonempty prefix of the corpus")
    by_id = _indexdataset(dataset)
    cases = []
    for index, example in enumerate(document["examples"][:count]):
        row = by_id.get(example["task_id"])
        if row is None or row["task_id"] not in coding.TRAIN_IDS:
            raise ValueError("Generation sample must match the pinned TRAIN data")
        prompt = coding.prompt_for(row)
        if prompt != example["prompt"] or row["code"] != example["solution"]:
            raise ValueError("Corpus prompt/reference differs from the pinned dataset")
        if coding.text_hash(prompt) != example["prompt_sha256"]:
            raise ValueError("Corpus prompt checksum mismatch")
        boundary = int(np.flatnonzero(objective.mask[index])[0])
        prompt_ids = objective.tokens[index, :boundary].tolist()
        if tokenizer.encode(prompt, add_special_tokens=False).ids != prompt_ids:
            raise ValueError("Standalone prompt tokenization differs from training")
        if len(prompt_ids) + max_new_tokens > config.context:
            raise ValueError("Generation exceeds model context; no truncation")
        cases.append(
            {
                "id": example["id"],
                "task_id": row["task_id"],
                "prompt": prompt,
                "prompt_token_ids": prompt_ids,
                "reference": row["code"],
                "setup": row["test_setup_code"],
                "tests": row["test_list"],
            }
        )
    return cases


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def render_report(record):
    esc = html.escape
    sections = []
    for case in record["cases"]:
        columns = []
        for label in ("original", "optimized"):
            result = record["results"][label][case["id"]]
            status = result.get("check", {}).get("status", "not_checked")
            text = result.get("selected_text", result["text"])
            candidate_count = len(result.get("candidates", [result]))
            selected = result.get("selected_candidate")
            selection = (
                f"; selected candidate {selected}" if selected is not None else ""
            )
            columns.append(
                f"<div><h3>{esc(label.title())}: {esc(status)}</h3>"
                f"<p>{candidate_count} candidates{selection}</p>"
                f"<pre>{esc(text)}</pre></div>"
            )
        sections.append(
            f"<section><h2>{esc(case['id'])}</h2><pre>{esc(case['prompt'])}</pre>"
            f'<div class="pair">{"".join(columns)}</div></section>'
        )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>FLAME generation comparison</title><style>"
        "body{font:15px system-ui;margin:24px auto;padding:0 20px;max-width:1400px;color:#172121;background:#fff}"
        "h1{font-size:26px}h2{font-size:20px}h3{font-size:16px}section{border-top:1px solid #ccd4d4;padding:20px 0}"
        ".pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.5}"
        "@media(max-width:700px){.pair{grid-template-columns:1fr}}"
        "</style><body><h1>FLAME generation comparison</h1>"
        f"<p>{len(record['cases'])}-problem TRAIN sanity check. Public tests are in the prompts. "
        "Candidate zero is greedy; nearby first-token candidates are completed with "
        "the same model and budget. No reference tokens are fed to the model. "
        "The checker selects a passing raw candidate without repair. This is not "
        "hidden-test accuracy.</p>" + "".join(sections) + "</body></html>"
    )


def _scorecase(label, case, completion, report, result, check_solution):
    candidates = completion.get("candidates", [completion])
    for index, candidate in enumerate(candidates):
        candidate["check"] = check_solution(
            candidate["text"], case["setup"], case["tests"]
        )
        click.echo(
            f"{label} {case['id']} candidate {index}: {candidate['check']['status']}"
        )
        write_json(report, result)
    selected = next(
        (
            (index, candidate)
            for index, candidate in enumerate(candidates)
            if candidate["check"]["status"] == "passed"
        ),
        (0, candidates[0]),
    )
    completion["selected_candidate"] = selected[0]
    completion["selected_text"] = selected[1]["text"]
    completion["check"] = selected[1]["check"]
    write_json(report, result)


def check_outputs(output):
    """Explicit post-generation scoring after inspection; never runs a model."""
    from .codecheck import check_solution

    report = output / "generation.json"
    result = json.loads(report.read_text())
    if result.get("status") != "complete" or not result.get("reference_checks_valid"):
        raise ValueError(
            "Scoring requires complete generation and passing reference checks"
        )
    if result.get("test_passes") is not None:
        raise ValueError("These completions have already been scored")
    for label, values in result["results"].items():
        for case in result["cases"]:
            _scorecase(label, case, values[case["id"]], report, result, check_solution)
    result["test_passes"] = {
        label: sum(value["check"]["status"] == "passed" for value in values.values())
        for label, values in result["results"].items()
    }
    result["all_completions_checked"] = all(
        value["check"].get("isolation_verified") is True
        for values in result["results"].values()
        for value in values.values()
    )
    write_json(report, result)
    (output / "report.html").write_text(render_report(result))
    return result
