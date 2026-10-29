import pytest

pytest.importorskip("tokenizers")

from ops import gemma4
from ops import pretrain


def test_vocab():
    tok = gemma4.load_gemma4()
    assert tok.get_vocab_size(with_added_tokens=True) == gemma4.VOCAB_SIZE
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<eos>") == 1
    assert tok.token_to_id("<bos>") == 2
    assert tok.token_to_id("<|endrepository|>") == 256014
    assert tok.token_to_id("<|turn>") == 256000


def test_recipe():
    recipe = pretrain.corpus_recipe(preset=pretrain.CORPUS_GEMMA4)
    assert recipe["preset"] == pretrain.CORPUS_GEMMA4
    assert recipe["tokenizer"]["kind"] == "gemma4"
    assert recipe["tokenizer"]["vocabulary"] == gemma4.VOCAB_SIZE

    stage = pretrain.token_stage(
        {"train": {"implementation": 100}}, preset=pretrain.CORPUS_GEMMA4
    )
    assert stage["tokenizer"]["kind"] == "gemma4"
    assert stage["tokenizer"]["vocabulary"] == gemma4.VOCAB_SIZE
    assert "<|endrepository|>" in stage["tokenizer"]["special_tokens"]


def test_spool(tmp_path):
    tokenizer = gemma4.load_gemma4()
    documents = {"train": {bucket: [] for bucket in pretrain.QUOTAS}}
    documents["train"]["implementation"] = [
        pretrain.Document(
            "repo1", "c1", "id1", "main.py", "implementation", "def hello(): pass"
        ),
        pretrain.Document(
            "repo2", "c2", "id2", "utils.py", "implementation", "import sys"
        ),
    ]
    for bucket in ("tests", "documentation", "configuration"):
        documents["train"][bucket] = [
            pretrain.Document(
                "repo1", "c1", f"id_{bucket}", "file.txt", bucket, "content"
            )
        ]

    pools = tmp_path / "pools"
    paths, counts = pretrain.spool_tokens(
        tokenizer, documents, pools, vocab=gemma4.VOCAB_SIZE
    )
    impl_path = paths["train"]["implementation"]
    assert impl_path.name.endswith(".u32")
    data = impl_path.read_bytes()
    assert len(data) == counts["train"]["implementation"] * 4

    tokens = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
    assert all(0 <= t < gemma4.VOCAB_SIZE for t in tokens)
    # Repository delimiter should be present between repo1 and repo2
    eor = tokenizer.token_to_id("<|endrepository|>")
    assert eor in tokens


def test_pack(tmp_path):
    sequences = 2
    paths = {}
    counts = {}
    pools = tmp_path / "pools"
    pools.mkdir()
    for bucket, quota in pretrain.QUOTAS.items():
        needed = sequences * quota
        path = pools / f"train-{bucket}.u32"
        values = list(range(100, 100 + needed))
        with path.open("xb") as stream:
            counts[bucket] = pretrain.write_values(
                stream, values, vocab=gemma4.VOCAB_SIZE
            )
        paths[bucket] = path

    out_file = tmp_path / "train.bauxptn"
    digest, offsets = pretrain.pack_stream(
        out_file,
        paths,
        counts,
        sequences,
        pretrain.SEED,
        vocab=gemma4.VOCAB_SIZE,
    )
    assert len(digest) == 64
    assert out_file.stat().st_size == 24 + sequences * pretrain.CONTEXT * 4

    raw = out_file.read_bytes()
    assert raw[:8] == pretrain.MAGIC
    assert int.from_bytes(raw[8:12], "little") == gemma4.VOCAB_SIZE
    assert int.from_bytes(raw[12:16], "little") == pretrain.CONTEXT
    assert int.from_bytes(raw[16:20], "little") == sequences
    assert int.from_bytes(raw[20:24], "little") == 0
