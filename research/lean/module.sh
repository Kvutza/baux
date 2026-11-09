#!/bin/sh
# Focused verification in the supplied workspace, using its pinned Lake project.
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: module.sh WORKSPACE BauxResearch.MODULE" >&2
    exit 2
fi
cd "$1/research/lean"
module=$2
case "$module" in
    BauxResearch.*) ;;
    *) echo "expected a BauxResearch module" >&2; exit 2 ;;
esac
case "$module" in
    *[!A-Za-z0-9_.]*) echo "invalid module name" >&2; exit 2 ;;
esac
source=$(printf '%s' "$module" | tr . /).lean
test -f "$source"
# Restore only the direct Mathlib imports and their dependency closures.
imports=$(awk '
    $1 == "import" && $2 ~ /^Mathlib\./ { print $2 }
    $1 == "public" && $2 == "import" && $3 ~ /^Mathlib\./ { print $3 }
' "$source")
if [ -n "$imports" ]; then
    # Module names deliberately expand to individual arguments, without globbing.
    set -f
    lake exe cache get "$imports"
fi
lake build "$module"
# Replay the source, including its explicit axiom reports, even on a cache hit.
lake env lean "$source"
