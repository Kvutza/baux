#!/bin/sh
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec python3 "$REPO_ROOT/tools/enn_coverage.py" \
  --adversarial \
  --manifest "$REPO_ROOT/research/upstream/coverage_manifest.json" \
  --enn-root "$REPO_ROOT/.cache/baux/upstream/enn"
