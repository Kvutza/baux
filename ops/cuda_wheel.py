"""Package a CUDA-enabled Baux extension as a CPython Linux wheel."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import shutil
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

import click


def _digest(path: Path) -> str:
    value = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
    return "sha256=" + value.rstrip(b"=").decode("ascii")


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _validateenv(python_abi: str, artifacts: tuple[Path, ...]) -> str:
    actual_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_abi != actual_abi or not 12 <= sys.version_info.minor <= 15:
        raise RuntimeError(
            f"CUDA wheel requested {python_abi}, running interpreter is {actual_abi}"
        )
    if sys.platform != "linux":
        raise RuntimeError("CUDA wheels are built on Linux")
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix or f"cpython-{actual_abi[2:]}" not in suffix:
        raise RuntimeError(f"unexpected extension suffix: {suffix!r}")
    return suffix


def _writedistinfo(
    stage: Path, root: Path, version: str, python_requires: str, tag: str
) -> None:
    dist = stage / f"baux-{version}.dist-info"
    _write(
        dist / "METADATA",
        "\n".join(
            [
                "Metadata-Version: 2.3",
                "Name: baux",
                f"Version: {version}",
                "Summary: Epistemic Nearest Neighbors with CUDA-Oxide sm_75 support",
                f"Requires-Python: {python_requires}",
                "Requires-Dist: numpy>=2.1",
                "Requires-Dist: scipy>=1.11",
                "Requires-Dist: click==8.5.0",
                "Requires-Dist: botorch>=0.18.1,<0.19",
                "Requires-Dist: optuna>=5.0,<5.1",
                "Requires-Dist: ax-platform>=1.3.1,<1.4",
                "",
            ]
        ),
    )
    _write(
        dist / "WHEEL",
        "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: baux.cuda_wheel",
                "Root-Is-Purelib: false",
                f"Tag: {tag}",
                "",
            ]
        ),
    )
    _write(dist / "top_level.txt", "baux\n")
    _write(
        dist / "entry_points.txt",
        "[console_scripts]\nbaux-cuda = baux.cuda:main\n",
    )
    license_dir = dist / "licenses"
    license_dir.mkdir(parents=True)
    shutil.copy2(root / "LICENSE", license_dir / "LICENSE")
    shutil.copy2(root / "NOTICE", license_dir / "NOTICE")


def _archivestage(stage: Path, version: str, wheel: Path) -> None:
    record = stage / f"baux-{version}.dist-info" / "RECORD"
    rows = [
        (
            path.relative_to(stage).as_posix(),
            _digest(path),
            str(path.stat().st_size),
        )
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path != record
    ]
    rows.append((record.relative_to(stage).as_posix(), "", ""))
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    _write(record, stream.getvalue())
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())


def build_wheel(
    root: Path,
    extension: Path,
    runner: Path,
    corpus: Path,
    cuda: Path,
    *extra: object,
    digest: Path | None = None,
    output: Path | None = None,
    version: str = "",
    python_abi: str = "",
    python_requires: str = "",
) -> Path:
    if extra:
        digest, output, version, python_abi, python_requires = extra  # type: ignore[assignment]
    assert digest is not None and output is not None
    suffix = _validateenv(python_abi, (extension, runner, corpus, cuda, digest))
    full_version = version + "+cuda75"
    tag = f"{python_abi}-{python_abi}-manylinux_2_28_x86_64"
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"baux-{full_version}-{tag}.whl"
    with tempfile.TemporaryDirectory(prefix="baux-cuda-wheel-") as directory:
        stage = Path(directory)
        package = stage / "baux"
        shutil.copytree(
            root / "src/baux",
            package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"),
        )
        shutil.copy2(extension, package / f"baux_rust{suffix}")
        binary_dir = package / "bin"
        binary_dir.mkdir()
        for source in (runner, corpus, cuda):
            shutil.copy2(source, binary_dir / source.name)
        shutil.copytree(root / "data" / "recipes", package / "data" / "recipes")
        shutil.copy2(digest, package / "source.sha256")
        _writedistinfo(stage, root, full_version, python_requires, tag)
        _archivestage(stage, full_version, wheel)
    return wheel


def main(**kwargs: object) -> None:
    try:
        wheel = build_wheel(
            kwargs["root"],  # type: ignore[arg-type]
            kwargs["extension"],  # type: ignore[arg-type]
            kwargs["runner"],  # type: ignore[arg-type]
            kwargs["corpus"],  # type: ignore[arg-type]
            kwargs["cuda"],  # type: ignore[arg-type]
            digest=kwargs["digest"],  # type: ignore[arg-type]
            output=kwargs["output"],  # type: ignore[arg-type]
            version=str(kwargs["version"]),
            python_abi=str(kwargs["python_abi"]),
            python_requires=str(kwargs["python_requires"]),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(wheel)


for _decorator in (
    click.option("--python-requires", required=True),
    click.option("--python-abi", required=True),
    click.option("--version", required=True),
    click.option(
        "--root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        default=Path(__file__).resolve().parents[1],
        show_default=True,
    ),
    click.argument("output", type=click.Path(file_okay=False, path_type=Path)),
    click.argument(
        "digest", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    click.argument(
        "cuda", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    click.argument(
        "corpus", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    click.argument(
        "runner", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    click.argument(
        "extension", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    ),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
