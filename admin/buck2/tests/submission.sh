#!/bin/sh
set -eu
# Use the repository's pinned verification environment, as wheel checks do.
export PYTHONPATH="$PWD/python:$PWD/admin"
exec "$BAUX_TEST_PYTHON" -m pytest -q -p no:cacheprovider -c /dev/null \
    --confcutdir=tests/python_api tests/python_api/testsubmission.py
