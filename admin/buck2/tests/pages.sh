#!/bin/sh
set -eu

if [ "${BAUX_TEST_WHEEL:-}" != "" ] && [ -x "${BAUX_TEST_PYTHON:-}" ]; then
    test_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-pages.XXXXXX")
    trap 'rm -rf "$test_dir"' EXIT HUP INT TERM
    "$BAUX_TEST_PYTHON" -m zipfile -e "$BAUX_TEST_WHEEL" "$test_dir"
    export PYTHONPATH="$test_dir:$PWD"
    "$BAUX_TEST_PYTHON" -m pytest -q tests/test_pages.py "$@"
else
    py_bin=".pixi/envs/${BAUX_PY_ENV:-baux-py314}/bin/python"

    if [ -x "$py_bin" ]; then
        exec "$py_bin" -m pytest -q tests/test_pages.py "$@"
    else
        first_wheel=""
        for w in dist/baux-*.whl; do
            if [ -f "$w" ]; then
                first_wheel="$w"
                break
            fi
        done
        if [ -n "$first_wheel" ]; then
            exec uv run --with "$first_wheel" --no-project pytest -q tests/test_pages.py "$@"
        else
            exec python3 -m pytest -q tests/test_pages.py "$@"
        fi
    fi
fi
