#!/bin/sh
set -eu
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-compiler.XXXXXX")
trap 'test_status=$?; rm -rf "$test_dir"; exit "$test_status"' EXIT HUP INT TERM
"$BAUX_TEST_PYTHON" -m zipfile -e "$BAUX_TEST_WHEEL" "$test_dir"
export PYTHONPATH="$test_dir"
export PYTHONFAULTHANDLER=1
"$BAUX_TEST_PYTHON" -m pytest -q -p no:cacheprovider -c /dev/null \
    --confcutdir=tests/compiler tests/compiler/test*.py "$@"
