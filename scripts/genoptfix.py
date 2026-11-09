from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from baux.turbo.optimizer_fixtures import (  # noqa: E402
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
        lines.append("[[steps]]")
        lines.append("ask = [")
        for pt in step["ask"]:
            lines.append(f"  {pt!r},")
        lines.append("]")
        lines.append("tell_y = [")
        for val in step["tell_y"]:
            lines.append(f"  {val!r},")
        lines.append("]")
        lines.append(f"tr_length = {step['tr_length']}")
        lines.append("")
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
