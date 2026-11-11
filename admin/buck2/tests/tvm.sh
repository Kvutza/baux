#!/bin/sh
set -eu
# Isolated stdlib only: no TVM Python package, site-packages or PYTHONPATH.
exec "$BAUX_TEST_PYTHON" -I -S - "$BAUX_TEST_WHEEL" "$BAUX_TVM" <<'PY'
import ctypes as c
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="baux-tvm-package-") as directory:
    root = Path(directory)
    with zipfile.ZipFile(sys.argv[1]) as wheel:
        for name in ("TVM-LICENSE", "TVM-FFI-LICENSE", "TVM-FFI-NOTICE"):
            matches = [path for path in wheel.namelist() if path.endswith("/licenses/" + name)]
            assert len(matches) == 1 and len(wheel.read(matches[0])) > 100, name
        wheel.extractall(root)
    libraries = root / "baux" / ".dylibs"
    library = c.CDLL(str(libraries / "libbaux_tvm.dylib"))
    artifact = root / "round.so"
    shutil.copyfile(sys.argv[2], artifact)
    size, pointer, text = c.c_size_t, c.c_void_p, c.c_char_p
    shape = (size * 1)(4)
    error = c.create_string_buffer(2048)

    def bind(name, result, *arguments):
        function = getattr(library, "bauxtvm_" + name)
        function.restype, function.argtypes = result, arguments
        return function

    load = bind("load", c.c_int, text, text, text, c.POINTER(pointer), pointer, size)
    put = bind("put", c.c_int, pointer, c.POINTER(size), size, c.c_uint32,
               pointer, size, c.POINTER(pointer), pointer, size)
    call = bind("call", c.c_int, pointer, c.POINTER(pointer), size,
                c.POINTER(pointer), size, pointer, size)
    read = bind("read", c.c_int, pointer, pointer, pointer, size, pointer, size)
    release = bind("release", None, pointer)
    close = bind("close", None, pointer)
    session, tensor = pointer(), pointer()
    assert load(bytes(artifact), b"main", b"cpu", c.byref(session), error, len(error)) == 0, error.value
    try:
        values = (c.c_float * 4)(1, 2, 3, 4)
        assert put(session, shape, 1, 7, values, c.sizeof(values), c.byref(tensor), error, len(error)) == 0, error.value
        outputs = (pointer * 2)()
        try:
            assert call(session, (pointer * 2)(tensor, tensor), 2, outputs, 2, error, len(error)) == 0, error.value
            for output, expected in zip(outputs, ([2, 4, 6, 8], [4, 16, 36, 64]), strict=True):
                assert read(session, output, values, c.sizeof(values), error, len(error)) == 0, error.value
                assert list(values) == expected, list(values)
        finally:
            for output in outputs:
                release(output)
            release(tensor)
    finally:
        close(session)

    # Confirm dyld resolved each dependency from the relocated wheel itself.
    dyld = c.CDLL(None)
    dyld._dyld_image_count.restype = c.c_uint32
    dyld._dyld_get_image_name.argtypes = [c.c_uint32]
    dyld._dyld_get_image_name.restype = text
    images = [Path(dyld._dyld_get_image_name(i).decode()).resolve()
              for i in range(dyld._dyld_image_count())]
    for name in ("libbaux_tvm.dylib", "libtvm_runtime.dylib", "libtvm_ffi.dylib"):
        assert [path for path in images if path.name == name] == [(libraries / name).resolve()], name
    print("Relocated wheel: native TVM execution, bundled dependencies and upstream licenses passed")
PY
