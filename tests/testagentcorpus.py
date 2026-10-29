"""Test agentic coding workflow corpus generation and packaging."""

from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

from ops.agent_corpus import build_episodes, train_tokenizer, pack_stream


def test_episodes():
    corpus = build_episodes(episodes=5, seed=123)
    assert len(corpus) == 5
    for ep in corpus:
        assert "<|turn>system\n" in ep
        assert "<|turn>user\n" in ep
        assert "<|turn>model\n" in ep
        assert "<|channel>thought\n" in ep
        assert "<|tool_call>" in ep
        assert "<|tool_response>" in ep
        assert "<turn|>" in ep


def test_packaging(tmp_path: Path):
    corpus = build_episodes(episodes=10, seed=456)
    tok_path = tmp_path / "tokenizer.json"
    tok = train_tokenizer(corpus, tok_path, vocab_size=256)
    assert tok_path.is_file()

    out_file = tmp_path / "train.bauxptn"
    count, digest = pack_stream(
        tok, corpus, out_file, sequences=4, context=64, vocab=256
    )
    assert count == 256
    assert out_file.stat().st_size == 24 + 256 * 2
    assert len(digest) == 64
