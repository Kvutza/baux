"""Core constants, document classification, and token stream packing for pretrain."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sys
import threading
import time
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SOURCE_ID = "HuggingFaceCode/stack-v3-train"
SOURCE_REVISION = "1f61b735bc0a5698345ce2196730f24bfa467f33"
CORPUS_PRESET = "stack_v3_python_pilot_v1"
CORPUS_800K = "stack_v3_python_800k_v1"
CORPUS_GEMMA4 = "stack_v3_python_gemma4_v1"
MODEL_PRESET = "fbt_pisa1_legacy_v1"
MODEL_ALIASES = {
    "fbt-pisa1-legacy-v1": MODEL_PRESET,
    "fbt-pisa1-residual1-v1": "fbt_pisa1_residual1_v1",
    "fbt-pisa1-projected-boundary-v1": "fbt_pisa1_projected_boundary_v1",
    "fbt-pisa1-hc4-v1": "fbt_pisa1_hc4_v1",
    "fbt-pisa1-mhc4-v1": "fbt_pisa1_mhc4_v1",
    "fbt_pisa1_moe_v1": MODEL_PRESET,
}
CORPUS_ALIASES = {
    "stack-v3-python-pilot-v1": CORPUS_PRESET,
    "stack-v3-python-800k-v1": CORPUS_800K,
    "stack-v3-python-gemma4-v1": CORPUS_GEMMA4,
    "gemma4-pilot-v1": CORPUS_GEMMA4,
}
MODEL_PRESETS = (
    MODEL_PRESET,
    "fbt_pisa1_residual1_v1",
    "fbt_pisa1_projected_boundary_v1",
    "fbt_pisa1_hc4_v1",
    "fbt_pisa1_mhc4_v1",
)
MAGIC = b"BAUXPTN1"
VOCAB = 8192
GEMMA4_VOCAB = 256015
CONTEXT = 4096
SEED = 0x454E4E58
SPECIALS = ("<|unknown|>", "<|endoftext|>", "<|endrepository|>")
QUOTAS = {
    "implementation": 2867,
    "tests": 615,
    "documentation": 410,
    "configuration": 204,
}
MAX_FILE_CHARS = 262_144
MAX_REPOSITORY_CHARS = 2_097_152
CHARS_PER_TOKEN_RESERVE = 6
SHUFFLE_REPOSITORIES = 1
PARQUET_BATCH = 1
COLLECTOR_VERSION = 2
SPLIT_POLICY = "repository_sha256_v2"
PRESET_SEQUENCES = {"train": 20, "validation": 16, "test": 16}
CORPUS_SEQUENCES = {
    CORPUS_PRESET: PRESET_SEQUENCES,
    CORPUS_800K: {"train": 200, "validation": 16, "test": 16},
    CORPUS_GEMMA4: PRESET_SEQUENCES,
}
LOG = logging.getLogger("baux.pretrain")


@contextmanager
def phase(name: str):
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(5):
            LOG.info("%s | working | elapsed %.1fs", name, time.monotonic() - started)

    LOG.info("%s | start", name)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException as error:
        LOG.error(
            "%s | failed | elapsed %.1fs | %s",
            name,
            time.monotonic() - started,
            error,
        )
        raise
    else:
        LOG.info("%s | complete | elapsed %.1fs", name, time.monotonic() - started)
    finally:
        stopped.set()
        thread.join()


@dataclass(frozen=True)
class Document:
    repository: str
    commit: str
    content_id: str
    path: str
    bucket: str
    text: str


def repository_split(repository: str, commit: str) -> str:
    del commit
    digest = hashlib.sha256(repository.encode()).digest()
    partition = int.from_bytes(digest[:8], "little") % 20
    if partition < 18:
        return "train"
    return "validation" if partition == 18 else "test"


def classify(path: str, language: str) -> str | None:
    lowered = "/" + path.lower().lstrip("/")
    basename = lowered.rsplit("/", 1)[-1]
    if language == "Python":
        if (
            "/test/" in lowered
            or "/tests/" in lowered
            or basename.startswith("test_")
            or basename.endswith("_test.py")
        ):
            return "tests"
        return "implementation"
    if language in {"Markdown", "reStructuredText"} or basename.startswith(
        ("readme", "contributing", "architecture")
    ):
        return "documentation"
    if language in {"Dockerfile", "INI", "JSON", "Makefile", "Shell", "TOML", "YAML"}:
        return "configuration"
    return None


def required_characters(sequences: dict[str, int]) -> dict[str, dict[str, int]]:
    result = {
        split: {
            bucket: math.ceil(count * tokens * CHARS_PER_TOKEN_RESERVE)
            for bucket, tokens in QUOTAS.items()
        }
        for split, count in sequences.items()
    }
    minimum_train = 8_000_000
    if sum(result["train"].values()) < minimum_train:
        for bucket, quota in QUOTAS.items():
            result["train"][bucket] = max(
                result["train"][bucket],
                math.ceil(minimum_train * quota / CONTEXT),
            )
    return result


def train_tokenizer(documents, buckets=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=SPECIALS[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        min_frequency=2,
        show_progress=True,
        special_tokens=list(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        (
            document.text
            for bucket in (buckets or QUOTAS)
            for document in documents["train"][bucket]
        ),
        trainer=trainer,
    )
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCAB:
        raise ValueError(
            "training corpus did not produce the required 8192-token vocabulary"
        )
    if tuple(tokenizer.token_to_id(token) for token in SPECIALS) != (0, 1, 2):
        raise ValueError("pretraining special-token IDs changed")
    return tokenizer


def write_values(stream, values, vocab: int = VOCAB) -> int:
    if any(not 0 <= token < vocab for token in values):
        raise ValueError("tokenizer emitted an out-of-range token")
    encoded = array("H" if vocab <= 65536 else "I", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    encoded.tofile(stream)
    return len(encoded)


def _lookupid(tokenizer, candidates) -> int | None:
    for token in candidates:
        try:
            token_id = tokenizer.token_to_id(token)
        except (KeyError, ValueError):
            continue
        if token_id is not None:
            return token_id
    return None


def _spoolbucket(
    path: Path, docs, tokenizer, delimiters: tuple[int, int], vocab: int
) -> tuple[int, list[dict]]:
    eot, eor = delimiters
    count, records, prev_repo = 0, [], None
    with path.open("xb") as stream:
        for doc in docs:
            if prev_repo not in (None, doc.repository):
                count += write_values(stream, [eor], vocab=vocab)
            encoding = tokenizer.encode(doc.text, add_special_tokens=False)
            records.append(
                {
                    "offset": count,
                    "length": len(encoding.ids),
                    "repository": doc.repository,
                    "commit": doc.commit,
                    "path": doc.path,
                    "content_id": doc.content_id,
                }
            )
            count += write_values(stream, encoding.ids, vocab=vocab)
            count += write_values(stream, [eot], vocab=vocab)
            prev_repo = doc.repository
    return count, records


def spool_tokens(
    tokenizer, documents, directory: Path, buckets=None, vocab: int = VOCAB
):
    eot = _lookupid(tokenizer, (SPECIALS[1], "<eos>"))
    eor = _lookupid(tokenizer, (SPECIALS[2], "<|endrepository|>"))
    if eot is None or eor is None:
        raise ValueError("tokenizer missing required document/repository delimiters")
    paths = {split: {} for split in documents}
    counts = {split: {} for split in documents}
    document_index = {}
    directory.mkdir()
    extension = "u16" if vocab <= 65536 else "u32"
    for split, split_docs in documents.items():
        for bucket in buckets or QUOTAS:
            path = directory / f"{split}-{bucket}.{extension}"
            count, records = _spoolbucket(
                path, split_docs[bucket], tokenizer, (eot, eor), vocab
            )
            paths[split][bucket] = path
            counts[split][bucket] = count
            document_index[path.name] = records
            LOG.info("tokenize | %s/%s | %s tokens", split, bucket, f"{count:,}")
    (directory / "document-index.json").write_text(
        json.dumps(document_index, separators=(",", ":")) + "\n"
    )
    return paths, counts


def pack(
    pools: dict[str, list[int]], sequences: int, seed: int
) -> tuple[list[int], dict]:
    needed = {bucket: sequences * quota for bucket, quota in QUOTAS.items()}
    short = {
        bucket: needed[bucket] - len(pools[bucket])
        for bucket in QUOTAS
        if len(pools[bucket]) < needed[bucket]
    }
    if short:
        raise ValueError(f"token pools are smaller than the requested mixture: {short}")
    offsets = {bucket: 0 for bucket in QUOTAS}
    packed = []
    for sequence in range(sequences):
        order = list(QUOTAS)
        random.Random(seed + sequence).shuffle(order)
        for bucket in order:
            start = offsets[bucket]
            end = start + QUOTAS[bucket]
            packed.extend(pools[bucket][start:end])
            offsets[bucket] = end
    assert len(packed) == sequences * CONTEXT
    return packed, {bucket: offsets[bucket] for bucket in QUOTAS}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_stream(
    path: Path, tokens: list[int], sequences: int, vocab: int = VOCAB
) -> str:
    header = MAGIC + vocab.to_bytes(4, "little") + CONTEXT.to_bytes(4, "little")
    header += sequences.to_bytes(4, "little") + (0).to_bytes(4, "little")
    with path.open("xb") as stream:
        stream.write(header)
        write_values(stream, tokens, vocab=vocab)
    return file_digest(path)


def _packseq(output, streams, offsets, quotas, stride: int, seq_seed: int) -> None:
    order = list(quotas)
    random.Random(seq_seed).shuffle(order)
    for bucket in order:
        size = quotas[bucket] * stride
        chunk = streams[bucket].read(size)
        if len(chunk) != size:
            raise ValueError(f"token pool ended early: {bucket}")
        output.write(chunk)
        offsets[bucket] += quotas[bucket]


def _writepacked(
    path: Path,
    header: bytes,
    streams: dict,
    offsets: dict,
    quotas: dict,
    spec: tuple[int, int, int],
) -> None:
    sequences, seed, stride = spec
    with path.open("xb") as output:
        output.write(header)
        for sequence in range(sequences):
            _packseq(output, streams, offsets, quotas, stride, seed + sequence)


def pack_stream(
    path: Path,
    paths: dict[str, Path],
    counts: dict[str, int],
    sequences: int,
    seed: int,
    quotas=None,
    *,
    vocab: int = VOCAB,
) -> tuple[str, dict[str, int]]:
    quotas = quotas or QUOTAS
    needed = {bucket: sequences * quota for bucket, quota in quotas.items()}
    short = {
        bucket: needed[bucket] - counts[bucket]
        for bucket in quotas
        if counts[bucket] < needed[bucket]
    }
    if short:
        raise ValueError(f"token pools are smaller than the requested mixture: {short}")
    stride = 2 if vocab <= 65536 else 4
    header = MAGIC + vocab.to_bytes(4, "little") + CONTEXT.to_bytes(4, "little")
    header += sequences.to_bytes(4, "little") + (0).to_bytes(4, "little")
    streams = {bucket: paths[bucket].open("rb") for bucket in quotas}
    offsets = {bucket: 0 for bucket in quotas}
    try:
        _writepacked(path, header, streams, offsets, quotas, (sequences, seed, stride))
    finally:
        for stream in streams.values():
            stream.close()
    return file_digest(path), offsets


def canonical_id(namespace: str, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(namespace.encode() + b"\0" + payload).hexdigest()[:20]
