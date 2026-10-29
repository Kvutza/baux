"""Build immutable Python-centered Stack v3 pretraining streams."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import tomllib
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import click

from ops import (
    pretrain_base as _base,
    pretrain_stage as _stage,
    pretrain_study as _study,
)
from ops.pretrain_base import (
    LOG,
    Document,
    canonical_id,
    classify,
    pack_stream,
    repository_split,
    required_characters,
    spool_tokens,
    train_tokenizer,
)
from ops.pretrain_study import (
    canonical_model,
    lower_study,
    resolved_toml,
    source_identity,
    validate_cache,
)

globals().update(
    {
        k: v
        for mod in (_base, _study)
        for k, v in vars(mod).items()
        if not k.startswith("_")
    }
)


def _logcollect(
    scanned: int, seen_count: int, characters: dict, targets: dict
) -> float:
    quota = sum(target for split in targets.values() for target in split.values())
    filled = sum(
        min(characters[split][bucket], target)
        for split, buckets in targets.items()
        for bucket, target in buckets.items()
    )
    LOG.info(
        "collect | repositories scanned %d | documents kept %d | character quota %.1f%%",
        scanned,
        seen_count,
        100 * filled / quota,
    )
    return time.monotonic()


def _validfile(
    file: dict,
    bucket: str | None,
    split_chars: int,
    target_chars: int,
    seen_content: set,
    remaining: int,
) -> bool:
    if file.get("is_vendor") or file.get("license_type") != "permissive":
        return False
    text = file["content"]
    return not (
        bucket is None
        or split_chars >= target_chars
        or file["content_id"] in seen_content
        or not isinstance(text, str)
        or not 64 <= len(text) <= _base.MAX_FILE_CHARS
        or "\0" in text
        or len(text) > remaining
    )


def _collectrepo(
    record: dict,
    split: str,
    targets: dict,
    documents: dict,
    characters: dict,
    seen_content: set,
) -> None:
    repository, commit = record["repo_path"], record["commit_id"]
    repo_chars = 0
    for file in record["files"]:
        if repo_chars >= _base.MAX_REPOSITORY_CHARS:
            break
        bucket = classify(file["file_path"], file["language"])
        target_chars = targets[split].get(bucket, 0) if bucket else 0
        if not _validfile(
            file,
            bucket,
            characters[split][bucket],
            target_chars,
            seen_content,
            _base.MAX_REPOSITORY_CHARS - repo_chars,
        ):
            continue
        assert bucket is not None
        text, cid = file["content"], file["content_id"]
        seen_content.add(cid)
        documents[split][bucket].append(
            Document(repository, commit, cid, file["file_path"], bucket, text)
        )
        characters[split][bucket] += len(text)
        repo_chars += len(text)


def collect(source, targets: dict[str, dict[str, int]]):
    documents = {split: defaultdict(list) for split in targets}
    characters = {split: defaultdict(int) for split in targets}
    seen_content: set[str] = set()
    reported = time.monotonic()
    for scanned, record in enumerate(source, 1):
        if time.monotonic() - reported >= 5:
            reported = _logcollect(scanned, len(seen_content), characters, targets)
        split = repository_split(record["repo_path"], record["commit_id"])
        if split not in targets or record.get("github_metadata", {}).get("is_fork"):
            continue
        _collectrepo(record, split, targets, documents, characters, seen_content)
        if all(
            characters[part][bucket] >= targets[part][bucket]
            for part in targets
            for bucket in _base.QUOTAS
        ):
            break
    missing = {
        f"{split}/{bucket}": targets[split][bucket] - characters[split][bucket]
        for split in targets
        for bucket in _base.QUOTAS
        if characters[split][bucket] < targets[split][bucket]
    }
    if missing:
        raise ValueError(
            f"Stack v3 stream ended before corpus quotas were met: {missing}"
        )
    LOG.info(
        "collect | documents kept %d | all character quotas met", len(seen_content)
    )
    return documents, characters


def corpus_recipe(rounds: int = 10, preset: str = _base.CORPUS_PRESET) -> dict:
    del rounds
    is_gemma = preset == _base.CORPUS_GEMMA4
    return {
        "preset": preset,
        "format": "baux.pretraining.v1",
        "source": {"id": _base.SOURCE_ID, "revision": _base.SOURCE_REVISION},
        "collector_version": _base.COLLECTOR_VERSION,
        "split_policy": _base.SPLIT_POLICY,
        "shuffle_repositories": _base.SHUFFLE_REPOSITORIES,
        "parquet_batch_repositories": _base.PARQUET_BATCH,
        "tokenizer": {
            "kind": "gemma4" if is_gemma else "byte_level_bpe",
            "vocabulary": _base.GEMMA4_VOCAB if is_gemma else _base.VOCAB,
            "package": "tokenizers==0.21.4",
        },
        "datasets_package": "datasets==5.0.1",
        "context": _base.CONTEXT,
        "seed": _base.SEED,
        "sequences": {**_base.CORPUS_SEQUENCES[preset]},
        "mixture_tokens_per_sequence": _base.QUOTAS,
    }


def token_stage(characters: dict, preset: str = _base.CORPUS_PRESET) -> dict:
    from ops import gemma4

    is_gemma = preset == _base.CORPUS_GEMMA4
    specials = list(gemma4.SPECIAL_TAGS if is_gemma else _base.SPECIALS)
    return {
        "format": "baux.token-pools.v1",
        "collector_version": _base.COLLECTOR_VERSION,
        "split_policy": _base.SPLIT_POLICY,
        "source": {"id": _base.SOURCE_ID, "revision": _base.SOURCE_REVISION},
        "filter": {
            "license_type": "permissive",
            "exclude_forks": True,
            "exclude_vendor": True,
            "max_file_characters": _base.MAX_FILE_CHARS,
            "max_repository_characters": _base.MAX_REPOSITORY_CHARS,
        },
        "required_characters": characters,
        "tokenizer": {
            "kind": "gemma4" if is_gemma else "byte_level_bpe",
            "vocabulary": _base.GEMMA4_VOCAB if is_gemma else _base.VOCAB,
            "package": "tokenizers",
            "package_version": "0.21.4",
            "special_tokens": specials,
        },
        "seed": _base.SEED,
        "shuffle_repositories": _base.SHUFFLE_REPOSITORIES,
        "parquet_batch_repositories": _base.PARQUET_BATCH,
        "mixture_tokens_per_sequence": _base.QUOTAS,
    }


def _checkstudy(document: dict) -> tuple[str, int]:
    if document.get("version") != 1:
        raise ValueError("version must be 1")
    if document.get("study") != "pretrain":
        raise ValueError("resolved corpus studies require study = 'pretrain'")
    if document.get("model") not in _base.MODEL_PRESETS:
        choices = ", ".join(repr(model) for model in _base.MODEL_PRESETS)
        raise ValueError(f"model must be one of {choices}")
    corpus = document.get("corpus")
    if corpus not in _base.CORPUS_SEQUENCES:
        choices = ", ".join(repr(name) for name in _base.CORPUS_SEQUENCES)
        raise ValueError(f"corpus must be one of {choices}")
    forbidden = sorted(
        {"output", "dataset", "validation_dataset"}.intersection(document)
    )
    if forbidden:
        raise ValueError(
            f"pretrain paths are generated automatically; remove {', '.join(forbidden)}"
        )
    rounds = document.get("rounds", 3)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    return corpus, rounds


def plan_study(document: dict, root: Path, source: str) -> dict:
    document = lower_study(document)
    canonical_model(document)
    corpus, rounds = _checkstudy(document)
    recipe = corpus_recipe(rounds, corpus)
    generation = document.get("generation")
    if generation is not None and generation.get("corpus_prompt_tokens") is not None:
        recipe["episodes"] = {
            "schema": "baux.document_episodes.v2",
            "prompt_tokens": generation["corpus_prompt_tokens"],
            "generated_tokens": generation["max_tokens"],
            "counts": {"train": 16, "validation": 3, "test": 3},
            "sources": {
                "train": ["train"],
                "validation": ["validation"],
                "test": ["test"],
            },
            "buckets": ["implementation", "tests"],
        }
    corpus_id = canonical_id("baux-corpus-v1", recipe)
    stage_recipe = token_stage(required_characters(recipe["sequences"]), preset=corpus)
    stage_id = canonical_id("baux-token-pools-v1", stage_recipe)
    experiment = {
        "config": document,
        "corpus_id": corpus_id,
        "stage_id": stage_id,
        "source": source,
    }
    experiment_id = canonical_id("baux-experiment-v1", experiment)
    cache = root / ".cache" / "baux"
    return {
        "corpus_id": corpus_id,
        "experiment_id": experiment_id,
        "corpus": cache / "corpora" / corpus_id,
        "stage": cache / "token-pools" / stage_id,
        "output": cache / "runs" / "pretrain" / experiment_id,
        "resolved": cache / "studies" / f"{experiment_id}.toml",
        "recipe": recipe,
    }


def _resolvegen(document: dict, parent: Path) -> None:
    if not isinstance(document.get("generation"), dict):
        return
    generation = dict(document["generation"])
    for key in ("checkpoint", "save_checkpoint", "qualification_manifest"):
        if generation.get(key) is not None:
            generation[key] = str((parent / generation[key]).resolve())
    reward = dict(generation["reward"])
    for key in ("checkpoint", "tokenizer_program", "program"):
        if reward.get(key) is not None:
            reward[key] = str((parent / reward[key]).resolve())
    generation["reward"] = reward
    document["generation"] = generation


def resolve_study(config: Path, root: Path) -> Path:
    document = lower_study(tomllib.loads(config.read_text()))
    canonical_model(document)
    _resolvegen(document, config.resolve().parent)
    plan = plan_study(document, root.resolve(), source_identity(root))
    plan["resolved"].parent.mkdir(parents=True, exist_ok=True)
    log_path = plan["resolved"].with_suffix(".prepare.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [pretrain] %(message)s"))
    LOG.addHandler(handler)
    try:
        LOG.info("corpus %s | preparation log %s", plan["corpus_id"], log_path)
        return resolve_plan(document, plan)
    except Exception as error:
        LOG.error("resolution failed | %s", error)
        raise
    finally:
        LOG.removeHandler(handler)
        handler.close()


def _runprepare(corpus: Path, plan: dict) -> None:
    sequences = plan["recipe"]["sequences"]
    LOG.info(
        "cache | miss | training batches %d | context %d",
        sequences["train"] // 2,
        _base.CONTEXT,
    )
    try:
        prepare(
            corpus,
            sequences["train"],
            sequences["validation"],
            sequences["test"],
            recipe_id=plan["corpus_id"],
            stage=plan["stage"],
            episodes=plan["recipe"].get("episodes"),
            preset=plan["recipe"].get("preset", _base.CORPUS_PRESET),
        )
    except TypeError:
        prepare(
            corpus,
            sequences["train"],
            sequences["validation"],
            sequences["test"],
            recipe_id=plan["corpus_id"],
            stage=plan["stage"],
            episodes=plan["recipe"].get("episodes"),
        )


def resolve_plan(document: dict, plan: dict) -> Path:
    corpus = plan["corpus"]
    manifest_path = corpus / "manifest.json"
    if corpus.exists():
        LOG.info("cache | hit | %s", corpus)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid cached corpus {corpus}: {error}") from error
        if manifest.get("recipe_id") != plan["corpus_id"]:
            raise ValueError(
                f"cached corpus recipe does not match its identity: {corpus}"
            )
        validate_cache(corpus, manifest, plan["recipe"])
    else:
        _runprepare(corpus, plan)
    plan["resolved"].parent.mkdir(parents=True, exist_ok=True)
    temporary = plan["resolved"].with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(resolved_toml(document, plan))
    os.replace(temporary, plan["resolved"])
    LOG.info(
        "ready | rounds %d | study %s", document.get("rounds", 3), plan["resolved"]
    )
    return plan["resolved"]


def prepare(
    output: Path,
    train_sequences: int,
    validation_sequences: int,
    test_sequences: int,
    recipe_id: str | None = None,
    stage: Path | None = None,
    *,
    episodes: dict | None = None,
    preset: str = _base.CORPUS_PRESET,
):
    sequences = {
        "train": train_sequences,
        "validation": validation_sequences,
        "test": test_sequences,
    }
    if any(count < 2 or count % 2 for count in sequences.values()):
        raise ValueError("every split requires a positive even sequence count")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_recipe = token_stage(required_characters(sequences), preset=preset)
    stage_id = canonical_id("baux-token-pools-v1", stage_recipe)
    stage = (stage or output.parent / "token-pools" / stage_id).resolve()
    stage.parent.mkdir(parents=True, exist_ok=True)
    vocab = stage_recipe["tokenizer"]["vocabulary"]
    hooks = (version, train_tokenizer, spool_tokens, pack_stream)
    if stage.exists():
        LOG.info("token pools | hit | %s", stage)
    else:
        LOG.info("token pools | miss | %s", stage)
        _stage.create_stage(
            stage, (stage_id, stage_recipe, sequences, vocab), preset, hooks
        )
    stage_state = _stage.load_stage(stage, stage_id, stage_recipe, sequences, vocab)
    manifest = _stage.init_manifest(
        recipe_id, stage_id, stage_recipe, stage_state[0], _base.SPLIT_POLICY, vocab
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _stage.write_corpus(
            temporary,
            output,
            manifest,
            stage,
            stage_state,
            (sequences, episodes, vocab, pack_stream),
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(
    context: click.Context,
    output: Path | None,
    train_sequences: int,
    validation_sequences: int,
    test_sequences: int,
) -> None:
    logging.basicConfig(level=logging.WARNING, format="[pretrain] %(message)s")
    LOG.setLevel(logging.INFO)
    if context.invoked_subcommand is not None:
        return
    if output is None:
        raise click.UsageError("Missing option '--output'.")
    try:
        manifest = prepare(
            output, train_sequences, validation_sequences, test_sequences
        )
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(manifest["splits"], sort_keys=True))


for _dec in (
    click.pass_context,
    click.option(
        "--test-sequences", type=click.IntRange(min=1), default=256, show_default=True
    ),
    click.option(
        "--validation-sequences",
        type=click.IntRange(min=1),
        default=256,
        show_default=True,
    ),
    click.option(
        "--train-sequences", type=click.IntRange(min=1), default=4096, show_default=True
    ),
    click.option("--output", type=click.Path(file_okay=False, path_type=Path)),
    click.group(invoke_without_command=True, help=__doc__),
):
    main = _dec(main)


def resolve(config: Path) -> None:
    try:
        path = resolve_study(config, Path.cwd())
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(path)


for _dec in (
    click.argument(
        "config", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    main.command(help="Resolve a path-free pretrain study."),
):
    resolve = _dec(resolve)


if __name__ == "__main__":
    main()
