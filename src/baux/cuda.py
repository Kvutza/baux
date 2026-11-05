"""CUDA execution shipped inside the Baux Python wheel."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click

from ._rust import cuda as _cuda


def _backend() -> Any:
    if _cuda is None:
        raise RuntimeError("CUDA support requires the Baux CUDA wheel on Linux x86-64")
    return _cuda


def target() -> str:
    """Return the PTX target selected for the current NVIDIA device."""
    return str(_backend().target())


def check() -> None:
    """Run the production CUDA component and recurrent-model checks."""
    _backend().check()


def trace(output: str | Path) -> Path:
    """Write the initialized production-model trace used for backend parity."""
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _backend().trace(path)
    return path


def read_trace(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate an initialized parity trace."""
    source = Path(path)
    report = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("format") != "baux.initialized-mhc4.v1"
    ):
        raise ValueError(f"{source} is not an Baux initialized trace")
    return report


def _stageworkspace(package: Path) -> tuple[Path, Path, dict[str, str]]:
    packaged = package / "bin"
    digest = package / "source.sha256"
    names = ("baux-runner", "baux-corpus", "baux-cuda")
    missing = [packaged / name for name in names if not (packaged / name).is_file()]
    if not digest.is_file():
        missing.append(digest)
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise RuntimeError(f"CUDA wheel is incomplete; missing: {missing_names}")

    workspace = Path.home() / ".cache" / "baux" / "wheel" / version("baux")
    binaries = workspace / "bin"
    binaries.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest = binaries / name
        shutil.copy2(packaged / name, dest)
        dest.chmod(0o755)
    recipes = workspace / "data" / "recipes"
    if not recipes.is_dir():
        recipes.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package / "data" / "recipes", recipes)

    environment = os.environ.copy()
    environment.update(
        {
            "BUILD_WORKSPACE_DIRECTORY": str(workspace),
            "BAUX_CORPUS_EXECUTABLE": str(binaries / "baux-corpus"),
            "BAUX_CUDA_EXECUTABLE": str(binaries / "baux-cuda"),
            "BAUX_SOURCE_DIGEST": digest.read_text(encoding="utf-8").strip(),
        }
    )
    return workspace, binaries / "baux-runner", environment


def run(config: str | Path, output: str | Path) -> Path:
    """Run a complete CUDA pre-training experiment from a TOML specification."""
    source = Path(config).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    workspace, runner, environment = _stageworkspace(Path(__file__).resolve().parent)
    subprocess.run(
        [str(runner), "cuda-tune-t4", str(source), "--out", str(destination)],
        cwd=workspace,
        env=environment,
        check=True,
    )
    return destination


@click.group(help=__doc__)
def main() -> None:
    pass


@main.command("target")
def target_cmd() -> None:
    """Print the selected device target."""
    click.echo(target())


@main.command("check")
def check_cmd() -> None:
    """Run the production CUDA model checks."""
    try:
        check()
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    click.echo("ok")


@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def trace_cmd(output: Path) -> None:
    """Write an initialized-model parity trace."""
    try:
        click.echo(trace(output))
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error


main.command("trace")(trace_cmd)


if __name__ == "__main__":
    main()
