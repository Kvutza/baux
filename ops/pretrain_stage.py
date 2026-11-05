"""Token-pool stage cache creation, loading, and corpus stream packing."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from ops.pretrain_base import (
    COLLECTOR_VERSION,
    CONTEXT,
    CORPUS_GEMMA4,
    MAX_FILE_CHARS,
    MAX_REPOSITORY_CHARS,
    QUOTAS,
    SEED,
    SOURCE_ID,
    SOURCE_REVISION,
    file_digest,
    phase,
    required_characters,
)


def _checkpool(
    stage: Path, split: str, bucket: str, pool_records: dict, vocab: int
) -> tuple[Path, int]:
    extension = "u16" if vocab <= 65536 else "u32"
    stride = 2 if vocab <= 65536 else 4
    record = pool_records.get(f"{split}/{bucket}")
    path = stage / "pools" / f"{split}-{bucket}.{extension}"
    if (
        not isinstance(record, dict)
        or type(record.get("count")) is not int
        or record["count"] < 0
        or not path.is_file()
        or path.stat().st_size != record["count"] * stride
        or file_digest(path) != record.get("sha256")
    ):
        raise ValueError(f"token pool is missing or corrupt: {path}")
    return path, record["count"]


def _readmeta(stage: Path, stage_id: str, stage_recipe: dict) -> dict:
    metadata_path = stage / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid token-pool cache {stage}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid token-pool cache metadata: {stage}")
    if metadata.get("stage_id") != stage_id or metadata.get("recipe") != stage_recipe:
        raise ValueError(f"token-pool cache identity does not match: {stage}")
    tokenizer_path = stage / "tokenizer.json"
    if not tokenizer_path.is_file() or file_digest(tokenizer_path) != metadata.get(
        "tokenizer_sha256"
    ):
        raise ValueError(
            f"tokenizer is missing or corrupt in token-pool cache: {stage}"
        )
    if metadata.get("document_index_sha256") is not None:
        index_path = stage / "pools" / "document-index.json"
        if (
            not index_path.is_file()
            or file_digest(index_path) != metadata["document_index_sha256"]
        ):
            raise ValueError(f"document index is missing or corrupt: {index_path}")
    return metadata


def load_stage(
    stage: Path, stage_id: str, stage_recipe: dict, sequences: dict, vocab: int
):
    metadata = _readmeta(stage, stage_id, stage_recipe)
    pool_records = metadata.get("pools")
    split_records = metadata.get("splits")
    if not isinstance(pool_records, dict) or not isinstance(split_records, dict):
        raise ValueError(f"invalid token-pool cache metadata: {stage}")
    paths = {split: {} for split in sequences}
    counts = {split: {} for split in sequences}
    for split in sequences:
        for bucket in QUOTAS:
            path, count = _checkpool(stage, split, bucket, pool_records, vocab)
            paths[split][bucket] = path
            counts[split][bucket] = count
        if not isinstance(split_records.get(split), dict):
            raise ValueError(f"invalid token-pool split metadata: {stage}/{split}")
    return metadata, paths, counts


def _stagemeta(
    stage_temp: Path,
    stage_info: tuple[str, dict, dict, int],
    collected: tuple[dict, dict],
    pools: tuple[dict, dict],
    version_fn,
) -> dict:
    stage_id, stage_recipe, sequences, _ = stage_info
    documents, characters = collected
    pool_paths, pool_counts = pools
    pool_metadata = {
        f"{split}/{bucket}": {
            "count": pool_counts[split][bucket],
            "sha256": file_digest(pool_paths[split][bucket]),
        }
        for split in sequences
        for bucket in QUOTAS
    }
    split_metadata = {}
    for split in sequences:
        rows = [doc for bucket in QUOTAS for doc in documents[split][bucket]]
        split_metadata[split] = {
            "collected_characters": dict(characters[split]),
            "documents": len(rows),
            "repositories": len({doc.repository for doc in rows}),
        }
    return {
        "stage_id": stage_id,
        "recipe": stage_recipe,
        "tokenizer_sha256": file_digest(stage_temp / "tokenizer.json"),
        "tokenizer_package_version": version_fn("tokenizers"),
        "collector_package_version": "native",
        "pools": pool_metadata,
        "splits": split_metadata,
        "document_index_sha256": file_digest(
            stage_temp / "pools" / "document-index.json"
        ),
    }


def _publishstage(stage_temp: Path, stage: Path) -> None:
    try:
        os.replace(stage_temp, stage)
    except OSError:
        if not stage.exists():
            raise
    finally:
        if stage_temp.exists():
            shutil.rmtree(stage_temp)


def _runspool(spool_fn, tokenizer, documents: dict, pools_dir: Path, vocab: int):
    with phase("tokenize"):
        try:
            return spool_fn(tokenizer, documents, pools_dir, vocab=vocab)
        except TypeError:
            return spool_fn(tokenizer, documents, pools_dir)


def _writestage(
    stage_temp: Path,
    stage: Path,
    stage_info: tuple[str, dict, dict, int],
    collected: tuple[dict, dict],
    tokenizer,
    hooks: tuple,
) -> None:
    vocab = stage_info[3]
    version_fn, _, spool_fn, _ = hooks
    pools = _runspool(spool_fn, tokenizer, collected[0], stage_temp / "pools", vocab)
    tokenizer.save(str(stage_temp / "tokenizer.json"))
    metadata = _stagemeta(stage_temp, stage_info, collected, pools, version_fn)
    (stage_temp / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    _publishstage(stage_temp, stage)


def create_stage(
    stage: Path,
    stage_info: tuple[str, dict, dict, int],
    preset: str,
    hooks: tuple,
) -> None:
    from ops.corpus import collect_native

    _, train_fn, _, _ = hooks
    sequences = stage_info[2]
    with phase("collect and filter"):
        collected = collect_native(required_characters(sequences))
    with phase("train tokenizer"):
        if preset == CORPUS_GEMMA4:
            from ops import gemma4

            tokenizer = gemma4.load_gemma4()
        else:
            tokenizer = train_fn(collected[0])
    stage_temp = Path(tempfile.mkdtemp(prefix=f".{stage.name}.", dir=stage.parent))
    try:
        _writestage(stage_temp, stage, stage_info, collected, tokenizer, hooks)
    except BaseException:
        shutil.rmtree(stage_temp, ignore_errors=True)
        raise


def _runpack(
    pack_fn,
    path: Path,
    split_paths: dict,
    split_counts: dict,
    count: int,
    seed: int,
    *,
    vocab: int,
):
    with phase(f"pack {path.stem} ({count} sequences)"):
        try:
            return pack_fn(path, split_paths, split_counts, count, seed, vocab=vocab)
        except TypeError:
            return pack_fn(path, split_paths, split_counts, count, seed)


def _packsplits(
    temporary: Path,
    manifest: dict,
    sequences: dict,
    stage_state: tuple[dict, dict, dict],
    vocab: int,
    pack_fn,
) -> None:
    stage_metadata, pool_paths, pool_counts = stage_state
    for index, (split, count) in enumerate(sequences.items()):
        path = temporary / f"{split}.bauxptn"
        digest, mixture = _runpack(
            pack_fn,
            path,
            pool_paths[split],
            pool_counts[split],
            count,
            SEED + index * 1_000_000,
            vocab=vocab,
        )
        manifest["splits"][split] = {
            "sequences": count,
            "batches": count // 2,
            "processed_tokens": count * CONTEXT,
            "causal_targets": count * (CONTEXT - 1),
            "mixture_tokens": mixture,
            **stage_metadata["splits"][split],
            "sha256": digest,
        }


def init_manifest(
    recipe_id: str | None,
    stage_id: str,
    stage_recipe: dict,
    stage_metadata: dict,
    split_policy: str,
    vocab: int,
) -> dict:
    specials = stage_recipe["tokenizer"]["special_tokens"]
    return {
        "format": "baux.pretraining.v1",
        "recipe_id": recipe_id,
        "stage_id": stage_id,
        "source": {"id": SOURCE_ID, "revision": SOURCE_REVISION},
        "collector_version": COLLECTOR_VERSION,
        "filter": {
            "license_type": "permissive",
            "exclude_forks": True,
            "exclude_vendor": True,
            "max_file_characters": MAX_FILE_CHARS,
            "max_repository_characters": MAX_REPOSITORY_CHARS,
        },
        "split_policy": split_policy,
        "split": "sha256(repository) modulo 20: train 0..17, validation 18, test 19",
        "deduplication": "content ID across all admitted splits",
        "mixture_tokens_per_sequence": QUOTAS,
        "tokenizer": {
            "kind": stage_recipe["tokenizer"]["kind"],
            "vocabulary": vocab,
            "special_tokens": dict(zip(specials, range(len(specials)))),
            "package_version": stage_metadata["tokenizer_package_version"],
            "sha256": stage_metadata["tokenizer_sha256"],
        },
        "seed": SEED,
        "splits": {},
    }


def _writeepisodes(
    temporary: Path, pool_paths: dict, episodes: dict, manifest: dict
) -> None:
    from ops.pretrain_episodes import write_episodes

    with phase("document-aligned generation episodes"):
        episode_path = temporary / "episodes.json"
        write_episodes(
            episode_path,
            pool_paths,
            episodes["counts"],
            QUOTAS,
            episodes["prompt_tokens"],
            episodes["generated_tokens"],
            SEED,
            sources=episodes["sources"],
            buckets=episodes["buckets"],
        )
        manifest["episodes"] = {**episodes, "sha256": file_digest(episode_path)}


def write_corpus(
    temporary: Path,
    output: Path,
    manifest: dict,
    stage: Path,
    stage_state: tuple[dict, dict, dict],
    job: tuple[dict, dict | None, int, object],
) -> None:
    sequences, episodes, vocab, pack_fn = job
    shutil.copyfile(stage / "tokenizer.json", temporary / "tokenizer.json")
    _packsplits(temporary, manifest, sequences, stage_state, vocab, pack_fn)
    if episodes is not None:
        _writeepisodes(temporary, stage_state[1], episodes, manifest)
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    os.replace(temporary, output)
