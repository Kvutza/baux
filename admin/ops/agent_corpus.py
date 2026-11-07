"""Agentic coding workflow corpus generator and Gemma 4 packager.

Generates multi-turn SWE-bench style execution trajectories conforming to
baux.agent.v1 and packs them into a contiguous 2.1M token .bauxptn stream.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL_TAGS = [
    "<|unknown|>",
    "<|endoftext|>",
    "<|endrepository|>",
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
]

TOPICS = [
    ("cache_eviction", "LRU cache evicted wrong key on concurrent write"),
    ("deadlock", "Worker thread deadlock under channel contention"),
    ("off_by_one", "Window aggregation off-by-one boundary index"),
    ("null_deref", "Unchecked optional unwrapped on empty HTTP stream"),
    ("json_escape", "Unescaped control characters in parser payload"),
    ("retry_budget", "Exponential backoff failed to clamp max attempts"),
    ("memory_leak", "Circular closure references in task queue observer"),
    ("token_bucket", "Rate limiter bucket drain underflowed integer"),
    ("schema_drift", "Protobuf field migration missed legacy fallback"),
    ("file_descriptor", "Dangling socket handle leaked during teardown"),
]

ROLES = ["system", "user", "model", "tool"]


def make_turn(role: str, body: str) -> str:
    """Format a single dialogue turn with control delimiters."""
    return f"<|turn>{role}\n{body}<turn|>\n"


def make_thought(reasoning: str) -> str:
    """Format private thought channel body."""
    return f"<|channel>thought\n{reasoning}\n<channel|>\n"


def make_call(name: str, args: dict[str, str]) -> str:
    """Format surgical tool call with string delimiters."""
    items = [f'{k}:<|"|>{v}<|"|>' for k, v in args.items()]
    payload = ",".join(items)
    return f"<|tool_call>call:{name}{{{payload}}}<tool_call|>\n"


def make_response(name: str, body: str) -> str:
    """Format tool execution response body."""
    return f"<|tool_response>response:{name}{{{body}}}<tool_response|>\n"


def _middleturns(file_path: str, test_path: str, summary: str) -> list[str]:
    t1 = make_thought(
        f"1. Analyze issue: {summary}.\n"
        f"2. Inspect {test_path} to observe reproduction assertion failure.\n"
        "3. Run pytest via bash tool."
    )
    c1 = make_call("bash", {"command": f"pytest -v {test_path}"})
    r1 = make_response(
        "bash",
        f"FAILED {test_path}::test_reproduce - AssertionError: {summary}\n"
        "1 failed, 12 passed in 0.18s",
    )
    t2 = make_thought(
        f"Test failed as expected. Need to read {file_path} around line 40 "
        "to identify where logic error occurs."
    )
    c2 = make_call("read", {"path": file_path})
    code_body = (
        "def process_event(event):\n"
        "    if not event.valid:\n"
        "        return None\n"
        "    target = event.index + 1\n"
        "    return store.fetch(target)\n"
    )
    r2 = make_response("read", code_body)
    t3 = make_thought(
        "Root cause identified: event.index + 1 exceeds valid range. "
        "Should be event.index. Applying patch with edit tool."
    )
    c3 = make_call(
        "edit",
        {
            "path": file_path,
            "old_str": "target = event.index + 1",
            "new_str": "target = event.index",
        },
    )
    r3 = make_response("edit", f"Applied 1 hunk to {file_path}")
    return [
        make_turn("model", t1 + c1),
        make_turn("tool", r1),
        make_turn("model", t2 + c2),
        make_turn("tool", r2),
        make_turn("model", t3 + c3),
        make_turn("tool", r3),
    ]


def sample_episode(rng: random.Random, ep_id: int) -> str:
    """Construct one multi-turn SWE problem-solving transcript."""
    slug, summary = rng.choice(TOPICS)
    file_path = f"src/engine/{slug}.py"
    test_path = f"tests/test_{slug}.py"

    sys_text = (
        "<|think|>You are an expert autonomous SWE coding agent.\n"
        "Solve the reported software issue by reading files, executing bash "
        "tests, applying surgical edits, and verifying no regressions occur."
    )
    user_text = (
        f"Issue #{ep_id + 100}: {summary}\n"
        f"Reproduction command: pytest -v {test_path}\n"
        f"Please locate the root cause in {file_path} and fix it."
    )
    t4 = make_thought(
        "Patch applied. Re-running pytest test suite to confirm resolution."
    )
    c4 = make_call("bash", {"command": f"pytest -v {test_path}"})
    r4 = make_response("bash", "13 passed in 0.22s. All tests passing.")
    t5 = make_thought("Fix verified. Responding with root cause and fix summary.")
    resolution = (
        f"Resolved issue #{ep_id + 100}: {summary}.\n"
        f"Corrected indexing boundary in {file_path}. All 13 tests passing."
    )

    chunks = [
        make_turn("system", sys_text),
        make_turn("user", user_text),
        *_middleturns(file_path, test_path, summary),
        make_turn("model", t4 + c4),
        make_turn("tool", r4),
        make_turn("model", t5 + resolution),
    ]
    return "".join(chunks)


def build_episodes(episodes: int = 1200, seed: int = 42) -> list[str]:
    """Generate collection of agentic workflow episodes."""
    rng = random.Random(seed)
    return [sample_episode(rng, i) for i in range(episodes)]


def train_tokenizer(corpus: list[str], path: Path, vocab_size: int = 8192):
    """Train BPE tokenizer on agentic coding workflows."""
    tok = Tokenizer(models.BPE(unk_token="<|unknown|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TAGS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    with path.open("r") as f:
        data = json.load(f)
    vocab = data["model"]["vocab"]
    for i in range(len(vocab), vocab_size):
        vocab[f"<|reserved_{i}|>"] = i
    with path.open("w") as f:
        json.dump(data, f)
    return Tokenizer.from_file(str(path))


def pack_stream(
    tok: Tokenizer,
    corpus: list[str],
    output: Path,
    sequences: int = 512,
    context: int = 4096,
    vocab: int = 8192,
):
    """Encode agentic corpus and pack into contiguous .bauxptn binary."""
    all_tokens: list[int] = []
    eor_id = tok.token_to_id("<|endrepository|>") or 2

    for ep in corpus:
        encoded = tok.encode(ep).ids
        all_tokens.extend(encoded)
        all_tokens.append(eor_id)

    total_needed = sequences * context
    while len(all_tokens) < total_needed:
        all_tokens.extend(all_tokens[: total_needed - len(all_tokens)])
    all_tokens = all_tokens[:total_needed]

    output.parent.mkdir(parents=True, exist_ok=True)
    magic = b"BAUXPTN1"
    header = (
        magic
        + vocab.to_bytes(4, "little")
        + context.to_bytes(4, "little")
        + sequences.to_bytes(4, "little")
        + (0).to_bytes(4, "little")
    )

    with output.open("wb") as f:
        f.write(header)
        for token in all_tokens:
            f.write((token % vocab).to_bytes(2, "little"))

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return total_needed, digest
