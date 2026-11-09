#!/bin/sh
set -eu

ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v sccache >/dev/null 2>&1; then
    echo "error: missing required command: sccache" >&2
    echo "Install sccache (e.g. brew install sccache) and ensure it is on PATH." >&2
    exit 1
fi

sccache --zero-stats >/dev/null 2>&1 || true

echo "Testing build through ./baux build..."
./baux build

echo "sccache statistics:"
sccache --show-stats
