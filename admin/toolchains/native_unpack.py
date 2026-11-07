"""Unpack pinned LLVM archives or merge the data files of Linux SDK packages."""

from pathlib import Path
import tarfile
import zipfile

import click


def extract(stream, output, package=None):
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if package and not relocate(member, package):
                continue
            archive.extract(member, output, filter="data")


def relocate(member, package):
    name = member.name.removeprefix("./")
    if name.startswith("info/licenses/"):
        member.name = "share/licenses/" + package + "/" + name[14:]
        return True
    # Buck supplies the sysroot explicitly; conda binaries need no relocation.
    return name.split("/", 1)[0] not in {"info", "bin"}


def extract_conda(path, output, package):
    with zipfile.ZipFile(path) as archive:
        payloads = [
            name
            for name in archive.namelist()
            if name.startswith("pkg-") and name.endswith(".tar.zst")
        ]
        if len(payloads) != 1:
            raise ValueError(f"Expected one package payload in {path}")
        extract_payload(archive, payloads[0], output, package)
        for name in archive.namelist():
            if name.startswith("info-") and name.endswith(".tar.zst"):
                extract_payload(archive, name, output, package)


def extract_payload(archive, name, output, package):
    with archive.open(name) as stream:
        extract(stream, output, package)


def extract_file(path, output, package):
    with path.open("rb") as stream:
        extract(stream, output, package)


def main(sdk: bool, output: Path, archives: tuple[Path, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in archives:
        package = path.name if sdk else None
        if path.suffix == ".conda":
            extract_conda(path, output, package)
        else:
            extract_file(path, output, package)


for _decorator in (
    click.argument(
        "archives",
        nargs=-1,
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    ),
    click.argument("output", type=click.Path(file_okay=False, path_type=Path)),
    click.option("--sdk", is_flag=True, help="Merge SDK packages by package name."),
    click.command(help=__doc__),
):
    main = _decorator(main)


if __name__ == "__main__":
    main()
