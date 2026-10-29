"""Resolve release tags and verify source and wheel versions before publishing."""

from __future__ import annotations

import configparser
from email.parser import Parser
import subprocess
import tomllib
import zipfile
from pathlib import Path

import click


PYTHON_ABIS = ("cp314", "cp315")
PLATFORM_TAGS = (
    "macosx_11_0_arm64",
    "manylinux_2_28_aarch64",
    "manylinux_2_28_x86_64",
)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def resolve_release(root: Path, tag: str) -> str:
    if not tag.startswith("v") or len(tag) == 1:
        raise ValueError("release tag must be v followed by the package version")
    ref = f"refs/tags/{tag}"
    git(root, "check-ref-format", ref)
    commit = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    workspace = tomllib.loads(git(root, "show", f"{commit}:Cargo.toml"))
    versions = {workspace["workspace"]["package"]["version"]}
    for path in ("backends/cuda/Cargo.toml", "backends/cuda/device/Cargo.toml"):
        package = tomllib.loads(git(root, "show", f"{commit}:{path}"))
        versions.add(package["package"]["version"])
    buck = configparser.ConfigParser()
    buck.read_string(git(root, "show", f"{commit}:.buckconfig"))
    versions.add(buck["baux"]["release_version"])
    if versions != {tag[1:]}:
        raise ValueError(
            f"tag {tag} disagrees with package versions: {sorted(versions)}"
        )
    return commit


def verify_checkout(root: Path, tag: str, commit: str) -> None:
    if resolve_release(root, tag) != commit:
        raise ValueError("release tag moved after resolution")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError("checkout is not the resolved release commit")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("release checkout contains modified tracked files")


def wheel_names(version: str, platform_tags: tuple[str, ...]) -> set[str]:
    return {
        f"baux-{version}-{abi}-{abi}-{platform_tag}.whl"
        for abi in PYTHON_ABIS
        for platform_tag in platform_tags
    }


def verify_wheels(
    directory: Path, version: str, platform_tags: tuple[str, ...] = PLATFORM_TAGS
) -> None:
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise ValueError("no release wheels found")
    actual = {wheel.name for wheel in wheels}
    expected = wheel_names(version, platform_tags)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "release wheels do not match expected set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata_paths = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"expected one wheel metadata document: {wheel.name}")
            metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if metadata.get_all("Name") != ["baux"] or metadata.get_all("Version") != [
            version
        ]:
            raise ValueError(f"wheel metadata disagrees with release: {wheel.name}")


def verify_release(
    root: Path,
    tag: str,
    commit: str | None,
    platform_tags: tuple[str, ...],
    wheels: Path | None,
) -> str:
    """Resolve the source identity and check every requested release artifact."""
    resolved = resolve_release(root, tag)
    if commit:
        verify_checkout(root, tag, commit)
    if wheels:
        verify_wheels(wheels, tag[1:], platform_tags or PLATFORM_TAGS)
    return resolved


def main(
    tag: str,
    commit: str | None,
    platform_tags: tuple[str, ...],
    wheels: Path | None,
) -> None:
    root = Path.cwd()
    try:
        resolved = verify_release(root, tag, commit, platform_tags, wheels)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"tag={tag}\ncommit={resolved}")


for _decorator in (
    click.option(
        "--wheels", type=click.Path(exists=True, file_okay=False, path_type=Path)
    ),
    click.option(
        "--platform-tag",
        "platform_tags",
        multiple=True,
        type=click.Choice(PLATFORM_TAGS),
    ),
    click.option("--commit"),
    click.argument("tag"),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
