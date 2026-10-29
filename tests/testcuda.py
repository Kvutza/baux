from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from baux import cuda


class Backend:
    def __init__(self) -> None:
        self.checked = False

    @staticmethod
    def target() -> str:
        return "sm_75"

    def check(self) -> None:
        self.checked = True

    @staticmethod
    def trace(output) -> None:
        output.write_text(
            json.dumps({"format": "baux.initialized-mhc4.v1"}), encoding="utf-8"
        )


def test_api(monkeypatch, tmp_path):
    backend = Backend()
    monkeypatch.setattr(cuda, "_cuda", backend)
    assert cuda.target() == "sm_75"
    cuda.check()
    assert backend.checked
    output = cuda.trace(tmp_path / "trace.json")
    assert cuda.read_trace(output)["format"] == "baux.initialized-mhc4.v1"


def test_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda, "_cuda", None)
    with pytest.raises(RuntimeError, match="CUDA wheel"):
        cuda.check()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"format":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="initialized trace"):
        cuda.read_trace(invalid)


def test_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda, "_cuda", Backend())
    runner = CliRunner()
    assert runner.invoke(cuda.main, ["target"]).output == "sm_75\n"
    output = tmp_path / "trace.json"
    result = runner.invoke(cuda.main, ["trace", str(output)])
    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_run(monkeypatch, tmp_path):
    package = tmp_path / "site" / "baux"
    binaries = package / "bin"
    recipes = package / "data" / "recipes"
    binaries.mkdir(parents=True)
    recipes.mkdir(parents=True)
    for name in ("baux-runner", "baux-corpus", "baux-cuda"):
        (binaries / name).write_bytes(name.encode())
    (package / "source.sha256").write_text("a" * 64, encoding="utf-8")
    (recipes / "test.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text("version = 4\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cuda, "__file__", str(package / "cuda.py"))
    monkeypatch.setattr(cuda, "version", lambda _: "0.2.3+cuda75")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        cuda.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    output = cuda.run(config, tmp_path / "result")

    assert output == (tmp_path / "result").resolve()
    command = calls[0][0][0]
    environment = calls[0][1]["env"]
    assert command[1:3] == ["cuda-tune-t4", str(config.resolve())]
    assert Path(environment["BAUX_CUDA_EXECUTABLE"]).is_file()
    assert environment["BAUX_SOURCE_DIGEST"] == "a" * 64
    assert (
        Path(environment["BUILD_WORKSPACE_DIRECTORY"]) / "data/recipes/test.json"
    ).is_file()
