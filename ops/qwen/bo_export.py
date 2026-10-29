"""Export and paired-comparison helpers for Qwen Metal BO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
from pathlib import Path

_FILE_CHUNK_BYTES = 16 * 1024 * 1024


def _paired(candidate: list[float], incumbent: list[float], population: int) -> dict:
    if len(candidate) != len(incumbent) or len(candidate) < 2:
        raise ValueError("paired losses must have equal size of at least two")
    count = len(candidate)
    candidate_mean = sum(candidate) / count
    incumbent_mean = sum(incumbent) / count
    differences = [old - new for new, old in zip(candidate, incumbent, strict=True)]
    improvement = sum(differences) / count
    correction = (1.0 - count / population) / count

    def variance(values: list[float]) -> float:
        mean = sum(values) / len(values)
        return (
            correction
            * sum((value - mean) ** 2 for value in values)
            / (len(values) - 1)
        )

    improvement_variance = variance(differences)
    improvement_se = math.sqrt(improvement_variance)
    threshold = 2.0 * improvement_se
    return {
        "candidate_loss": candidate_mean,
        "incumbent_loss": incumbent_mean,
        "candidate_variance": variance(candidate),
        "incumbent_variance": variance(incumbent),
        "improvement": improvement,
        "improvement_variance": improvement_variance,
        "improvement_se": improvement_se,
        "threshold": threshold,
        "accepted": improvement > threshold and candidate_mean < incumbent_mean,
        "deteriorated": improvement < -threshold,
    }


def _ckptheader(path: Path) -> dict:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("Qwen safetensors header is truncated")
        size = struct.unpack("<Q", raw_size)[0]
        if size > 64 * 1024 * 1024:
            raise ValueError("Qwen safetensors header is unreasonably large")
        try:
            return json.loads(stream.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Qwen safetensors header is invalid") from error


def _sha256file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _writestream(
    temporary_path: Path,
    weights_path: Path,
    records: list,
    encoded: bytes,
    padding: bytes,
    best_bytes: memoryview,
) -> None:
    with temporary_path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded) + len(padding)))
        stream.write(encoded)
        stream.write(padding)
        offset = 0
        for _, record in records:
            length = math.prod(record["shape"])
            start = offset * 2
            end = (offset + length) * 2
            for chunk_start in range(start, end, _FILE_CHUNK_BYTES):
                stream.write(
                    best_bytes[chunk_start : min(chunk_start + _FILE_CHUNK_BYTES, end)]
                )
            offset += length
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, weights_path)


def _writemanifest(source: Path, output: Path) -> None:
    for name in ("config.json", "merges.txt", "tokenizer_config.json", "vocab.json"):
        shutil.copyfile(source / name, output / name)
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "model_id": source_manifest["model_id"],
        "revision": source_manifest["revision"],
        "config": source_manifest["config"],
        "weights_downloaded": True,
        "optimized_from": str(source),
        "files": {
            "model.safetensors": {
                "bytes": (output / "model.safetensors").stat().st_size,
                "sha256": _sha256file(output / "model.safetensors"),
            }
        },
    }
    manifest_path = output / "manifest.json"
    temporary_manifest_path = output / ".manifest.json.tmp"
    try:
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_manifest_path, manifest_path)
    finally:
        temporary_manifest_path.unlink(missing_ok=True)


def _exportbest(source: Path, output: Path, best) -> None:
    """Write the native BF16 vector back as a loadable safetensors checkpoint."""
    import numpy as np

    if not isinstance(best, np.ndarray) or best.dtype != np.uint16 or best.ndim != 1:
        raise ValueError("Metal search returned an invalid BF16 incumbent")
    header = _ckptheader(source / "model.safetensors")
    records = sorted(
        (name, value)
        for name, value in header.items()
        if name not in {"__metadata__", "lm_head.weight"}
    )
    expected = sum(math.prod(record["shape"]) for _, record in records)
    if expected != best.size:
        raise ValueError(f"Best vector has {best.size} values; expected {expected}")

    new_header = {}
    offset = 0
    for name, record in records:
        length = math.prod(record["shape"])
        new_header[name] = {
            "dtype": "BF16",
            "shape": record["shape"],
            "data_offsets": [offset * 2, (offset + length) * 2],
        }
        offset += length
    encoded = json.dumps(new_header, separators=(",", ":"), sort_keys=True).encode()
    padding = b" " * ((8 - (len(encoded) % 8)) % 8)
    best_bytes = memoryview(best).cast("B")
    temporary_weights_path = output / ".model.safetensors.tmp"
    try:
        _writestream(
            temporary_weights_path,
            output / "model.safetensors",
            records,
            encoded,
            padding,
            best_bytes,
        )
    finally:
        temporary_weights_path.unlink(missing_ok=True)
    _writemanifest(source, output)
