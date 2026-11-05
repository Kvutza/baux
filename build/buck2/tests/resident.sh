#!/bin/sh
set -eu
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-regional.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT HUP INT TERM
"$BAUX_TEST_PYTHON" -m zipfile -e "$BAUX_TEST_WHEEL" "$test_dir"
export PYTHONPATH="$test_dir"
exec_python="$BAUX_TEST_PYTHON"
"$exec_python" -m pytest -q -p no:cacheprovider -c /dev/null \
    --confcutdir=tests/python_api tests/python_api/testresidentregion.py \
    tests/python_api/testsubmission.py tests/python_api/testcreateoptct.py \
    tests/python_api/testrestartcenter.py \
    tests/python_api/test_adapter.py::test_regionalprovider \
    tests/python_api/test_adapter.py::test_missingfactory \
    tests/python_api/test_adapter.py::test_regionmodels \
    tests/python_api/test_adapter.py::test_reusedprovider \
    tests/python_api/test_adapter.py::test_external \
    tests/testcfg_optimizer.py::test_enncosinerejected
