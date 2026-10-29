import configparser
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def read_version(path: Path, key: tuple[str, ...]) -> str:
    with path.open("rb") as file:
        data = tomllib.load(file)
    for part in key:
        data = data[part]
    return data


def test_version():
    config = configparser.ConfigParser()
    config.read(ROOT / ".buckconfig")
    versions = {
        read_version(ROOT / "Cargo.toml", ("workspace", "package", "version")),
        read_version(ROOT / "backends/cuda/Cargo.toml", ("package", "version")),
        read_version(ROOT / "backends/cuda/device/Cargo.toml", ("package", "version")),
        config["baux"]["release_version"],
    }
    assert len(versions) == 1, versions
