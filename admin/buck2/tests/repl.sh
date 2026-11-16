#!/bin/sh
set -eu

py_bin=".pixi/envs/${BAUX_PY_ENV:-baux-py314}/bin/python"

if [ -x "$py_bin" ]; then
    exec "$py_bin" -m pytest -q tests/test_repl.py "$@"
else
    first_wheel=""
    for w in dist/baux-*.whl; do
        if [ -f "$w" ]; then
            first_wheel="$w"
            break
        fi
    done
    if [ -n "$first_wheel" ]; then
        exec uv run --with "$first_wheel" --no-project pytest -q tests/test_repl.py "$@"
    else
        exec python3 -m pytest -q tests/test_repl.py "$@"
    fi
fi
