from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from baux.turbo.optimizer_fixtures import (
    FIXTURE_GENERATOR_ENTRIES,
    build_fixture,
    output_path,
)


def serializetoml(payload: dict) -> str:
    lines = [
        f'upstream_enn_revision = "{payload["upstream_enn_revision"]}"',
        f'rng_semantics = "{payload["rng_semantics"]}"',
        f"seed = {payload['seed']}",
        f"num_cycles = {payload['num_cycles']}",
        f"num_arms = {payload['num_arms']}",
        f'objective = "{payload["objective"]}"',
        f"bounds = {payload['bounds']!r}",
        "",
    ]
    for step in payload["steps"]:
        lines.extend(["[[steps]]", "ask = ["])
        lines.extend(f"  {pt!r}," for pt in step["ask"])
        lines.extend(["]", "tell_y = ["])
        lines.extend(f"  {val!r}," for val in step["tell_y"])
        lines.extend(["]", f"tr_length = {step['tr_length']}", ""])
    return "\n".join(lines)


def main() -> None:
    root = ROOT
    for entry in FIXTURE_GENERATOR_ENTRIES:
        for seed in (0, 1, 2):
            payload = build_fixture(entry, seed)
            path = output_path(entry, seed, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serializetoml(payload), encoding="utf-8")
            print("wrote", path)


if __name__ == "__main__":
    main()
