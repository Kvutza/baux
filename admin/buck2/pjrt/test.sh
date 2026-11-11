#!/bin/sh
set -eu
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-pjrt.XXXXXX")
trap 'status=$?; rm -rf "$test_dir"; exit "$status"' EXIT HUP INT TERM
cp "$BAUX_PJRT_PROBE" "$BAUX_PJRT_LIBRARY" "$test_dir/"
"$test_dir/probe"
