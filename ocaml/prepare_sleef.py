"""Build the pinned vector math dependency outside the source tree."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

VERSION = "3.9.0"
SHA256 = "af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908"
root = Path(sys.argv[1]).resolve()
install = root / "install"
if (install / "lib/libsleef.a").exists():
    sys.exit(0)
root.mkdir(parents=True, exist_ok=True)
archive = root / "source.tar.gz"
if not archive.exists():
    archive.write_bytes(
        urllib.request.urlopen(
            f"https://codeload.github.com/shibatch/sleef/tar.gz/refs/tags/{VERSION}",
            timeout=60,
        ).read()
    )
if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
    raise RuntimeError("SLEEF source checksum mismatch")
with tarfile.open(archive) as source:
    source.extractall(root, filter="data")
subprocess.run(
    [
        "cmake",
        "-S",
        str(root / f"sleef-{VERSION}"),
        "-B",
        str(root / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install}",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DSLEEF_BUILD_TESTS=OFF",
        "-DSLEEF_BUILD_DFT=OFF",
        "-DSLEEF_BUILD_QUAD=OFF",
        "-DSLEEF_ENABLE_TLFLOAT=OFF",
    ],
    check=True,
)
subprocess.run(["cmake", "--build", str(root / "build"), "--parallel", "4"], check=True)
subprocess.run(["cmake", "--install", str(root / "build")], check=True)
