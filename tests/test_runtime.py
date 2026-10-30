from __future__ import annotations

import pytest

import baux


def test_capabilities():
    rows = baux.capabilities()
    assert [row.name for row in rows] == ["cpu", "cuda", "metal", "opencl"]
    assert rows[0].available
    assert "posterior" in rows[0].direct_operations
    for row in rows:
        if row.compiled and row.name in ("metal", "cuda"):
            assert "device_posterior" in row.direct_operations
            assert "device_posterior" not in row.host_operations
        else:
            assert "device_posterior" in row.missing_operations
    for row in rows:
        if not row.compiled:
            assert row.available is False
            assert not row.direct_operations
            assert not row.fallback_operations
            assert row.reason
        elif row.name != "cpu":
            assert row.available is None
        if row.name == "metal" and row.compiled:
            assert "posterior_draw" in row.fallback_operations
        if row.name == "cuda" and row.compiled:
            assert "posterior_draw" in row.direct_operations
            assert "posterior_draw" in row.host_operations
    assert all(isinstance(row.array_frameworks, tuple) for row in rows)


def test_immutable():
    row = baux.capabilities()[0]
    with pytest.raises(AttributeError):
        row.available = False


def test_targets():
    from baux import baux_rust

    for public, target_row in zip(
        baux.capabilities(), baux_rust.capabilities(), strict=True
    ):
        actual = dict.fromkeys(public.direct_operations, "direct")
        actual.update(dict.fromkeys(public.fallback_operations, "fallback"))
        actual.update(dict.fromkeys(public.missing_operations, "missing"))
        assert actual == target_row["operations"]
        assert public.compiled == target_row["compiled"]


def test_imports():
    import os
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, baux; baux.capabilities(); "
                "assert not {'torch', 'jax', 'pyopencl'} & sys.modules.keys()"
            ),
        ],
        env=os.environ.copy(),
        check=True,
    )
