"""Read the pinned corpus through OpenDAL and Rust Arrow/Parquet."""

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from ops.pretrain_base import (
    LOG,
    MAX_FILE_CHARS,
    MAX_REPOSITORY_CHARS,
    SEED,
    SOURCE_ID,
    SOURCE_REVISION,
    Document,
    phase,
)


def source_paths(files, seed):
    import numpy as np
    from huggingface_hub import HfFileSystem, hf_hub_url

    # datasets 4.1.1 shuffles its shard list with this same generator. With a
    # one-repository buffer, the records within each shard keep their order.
    files = list(files)
    np.random.default_rng(seed).shuffle(files)
    filesystem = HfFileSystem()
    paths = []
    for file in files:
        source = filesystem.resolve_path(file)
        url = hf_hub_url(
            source.repo_id,
            source.path_in_repo,
            repo_type=source.repo_type,
            revision=source.revision,
        )
        paths.append(urlsplit(url).path)
    return paths


def _terminate(process) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def _readerr(stderr) -> str:
    last_line = ""
    for line in stderr:
        text = line.rstrip()
        last_line = text or last_line
        LOG.info("%s", text)
    return last_line


def _drainerr(process) -> str:
    assert process.stderr is not None
    try:
        return _readerr(process.stderr)
    except BaseException:
        _terminate(process)
        raise


def read_native(job):
    root = Path(__file__).resolve().parents[1]
    cmd = [
        "./tools/buck2w",
        "--isolation-dir",
        "dev",
        "run",
        "//rust/crates/corpus:corpus",
    ]
    with (
        tempfile.TemporaryFile(mode="w+") as request,
        tempfile.TemporaryFile(mode="w+") as result,
    ):
        json.dump(job, request)
        request.seek(0)
        with subprocess.Popen(
            cmd,
            cwd=root,
            stdin=request,
            stdout=result,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            last_line = _drainerr(process)
            if process.wait():
                raise ValueError(f"native corpus reader failed: {last_line}")
        result.seek(0)
        return json.load(result)


def collect_native(targets):
    from datasets import load_dataset_builder

    with phase("open source"):
        builder = load_dataset_builder(SOURCE_ID, revision=SOURCE_REVISION)
        assert builder.config.data_files is not None
        paths = source_paths(builder.config.data_files["train"], SEED)
    selected = read_native(
        {
            "root": "https://huggingface.co",
            "paths": paths,
            "targets": targets,
            "max_file": MAX_FILE_CHARS,
            "max_repository": MAX_REPOSITORY_CHARS,
        }
    )
    documents = {
        split: {
            bucket: [Document(**document) for document in rows]
            for bucket, rows in buckets.items()
        }
        for split, buckets in selected["documents"].items()
    }
    return documents, selected["characters"]
