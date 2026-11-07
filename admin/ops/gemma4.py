"""Canonical Gemma 4 tokenizer loader and exporter."""

from __future__ import annotations

from pathlib import Path

CACHE_PATH = Path(".cache/baux/gemma4/tokenizer.json")
VOCAB_SIZE = 256015
SPECIAL_TAGS = (
    "<|turn>",
    "<turn|>",
    "<|think|>",
    "<|channel>",
    "<channel|>",
    "<|tool>",
    "<tool|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|image|>",
    "<|audio|>",
    "<|endrepository|>",
)


def create_gemma4(target: Path = CACHE_PATH) -> Path:
    from tokenizers import AddedToken, Tokenizer

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_pretrained("unsloth/gemma-2-9b")
    tokens = [AddedToken(tag, special=True) for tag in SPECIAL_TAGS]
    tokenizer.add_special_tokens(tokens)
    tokenizer.save(str(target))
    return target


def load_gemma4(target: Path | None = None):
    from tokenizers import Tokenizer

    path = Path(target or CACHE_PATH)
    if not path.is_file():
        create_gemma4(path)
    return Tokenizer.from_file(str(path))
