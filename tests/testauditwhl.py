import subprocess
import sys
import types
import zipfile

from build.bazel.audit_wheel import audit_macos, audit_wheel, wheel_paths


def test_audit(tmp_path, monkeypatch):
    wheel = tmp_path / "baux-0.0.0-cp314-cp314-macosx_11_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("baux/__init__.py", "")
        archive.writestr("baux/baux_rust.so", "")
        archive.writestr("baux-0.0.0.dist-info/licenses/LICENSE", "license")
        archive.writestr("baux-0.0.0.dist-info/licenses/NOTICE", "notice")

    def check_output(command, text):
        assert text
        if command[1] == "-l":
            return "cmd LC_BUILD_VERSION\nminos 11.0\n"
        return "baux_rust.so\n\t/usr/lib/libSystem.B.dylib\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)
    rust = types.ModuleType("baux.baux_rust")
    rust.search = types.SimpleNamespace(TrialEngine=object)
    package = types.ModuleType("baux")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "baux", package)
    monkeypatch.setitem(sys.modules, "baux.baux_rust", rust)

    root = tmp_path / "contents"
    package_dir = root / "baux"
    dist_dir = root / "baux-0.0.0.dist-info"
    package_dir.mkdir(parents=True)
    dist_dir.mkdir()
    extension = package_dir / "baux_rust.cpython-314-darwin.so"
    extension.touch()
    assert wheel_paths(str(root)) == (str(dist_dir), str(extension))
    audit_macos(str(extension))
    audit_wheel(str(wheel))
