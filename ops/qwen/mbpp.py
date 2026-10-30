"""Pinned, bounded access to Qwen's MBPP training split."""

from __future__ import annotations

import hashlib
import json
from urllib.request import ProxyHandler, Request, build_opener

DATASET_ID = "google-research-datasets/mbpp"
DATASET_REVISION = "5a8a3b632e28582ab85087da984bef822e34e415"
DATASET_SOURCE_REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
DATASET_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
DATASET_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{DATASET_SOURCE_REVISION}/mbpp/mbpp.jsonl"
)
DATASET_INFO_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    f"{DATASET_REVISION}/dataset_infos.json"
)
TRAIN_IDS = range(601, 975)
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _responsebody(response) -> bytes:
    chunks = []
    size = 0
    while chunk := response.read(65536):
        size += len(chunk)
        if size > MAX_DOWNLOAD_BYTES:
            raise ValueError("Public artifact exceeds download size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def read_public(url: str) -> bytes:
    """Download a bounded public artifact without ambient proxy credentials."""
    request = Request(url, headers={"User-Agent": "baux-mbpp/1"})
    with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
        return _responsebody(response)


def load_train() -> list[dict]:
    """Load the pinned 374-row MBPP training split."""
    info = json.loads(read_public(DATASET_INFO_URL))["full"]
    checksum = next(iter(info["download_checksums"].values()))["checksum"]
    if checksum != DATASET_SHA256 or info["splits"]["train"]["num_examples"] != 374:
        raise ValueError("Pinned HF MBPP metadata does not match the TRAIN contract")
    data = read_public(DATASET_URL)
    if _sha256(data) != DATASET_SHA256:
        raise ValueError("MBPP source SHA256 mismatch")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    train = [row for row in rows if row["task_id"] in TRAIN_IDS]
    if sorted(row["task_id"] for row in train) != list(TRAIN_IDS):
        raise ValueError("MBPP TRAIN IDs differ from the pinned split")
    return train
