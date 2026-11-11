#!/bin/sh
set -eu
# Isolated stdlib only: no compiler Python packages, site-packages or PYTHONPATH.
exec "$BAUX_TEST_PYTHON" -I -S - "$BAUX_TEST_WHEEL" "$BAUX_STABLEHLO" <<'PY'
import ctypes as c
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="baux-native-package-") as directory:
    root = Path(directory)
    with zipfile.ZipFile(sys.argv[1]) as wheel:
        for name in ("TVM-LICENSE", "TVM-FFI-LICENSE", "TVM-FFI-NOTICE", "PJRT-LICENSE"):
            matches = [path for path in wheel.namelist() if path.endswith("/licenses/" + name)]
            assert len(matches) == 1 and len(wheel.read(matches[0])) > 100, name
        wheel.extractall(root)
    libraries = root / "baux" / ".dylibs"
    size, pointer, text = c.c_size_t, c.c_void_p, c.c_char_p
    shape = (size * 1)(4)
    error = c.create_string_buffer(2048)

    def bind(name, result, *arguments):
        function = getattr(library, "baux" + runtime + "_" + name)
        function.restype, function.argtypes = result, arguments
        return function

    for runtime in ("pjrt",):
        library = c.CDLL(str(libraries / "libbaux_pjrt.dylib"))
        put = bind("put", c.c_int, pointer, c.POINTER(size), size, c.c_uint32,
                   pointer, size, c.POINTER(pointer), pointer, size)
        call = bind("call", c.c_int, pointer, c.POINTER(pointer), size,
                    c.POINTER(pointer), size, pointer, size)
        read = bind("read", c.c_int, pointer, pointer, pointer, size, pointer, size)
        release = bind("release", None, pointer)
        close = bind("close", None, pointer)
        session, tensor = pointer(), pointer()
        compile = bind("compile", c.c_int, text, size, text, text, size, size, c.POINTER(pointer), pointer, size)
        source = Path(sys.argv[2]).read_bytes()
        # One replica/partition and two f32[4] argument ShapeProtos.
        options = bytes([0x1a, 4, 0x20, 1, 0x28, 1] + [0x0a, 4, 0x10, 11, 0x18, 4] * 2)
        status = compile(source, len(source), b"cpu", options, len(options), 2, c.byref(session), error, len(error))
        assert status == 0, error.value
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
    for name in ("libbaux_tvm.dylib", "libtvm_runtime.dylib", "libtvm_runtime_metal.dylib", "libtvm_ffi.dylib", "libtvm_compiler.dylib", "libbaux_pjrt.dylib", "libpjrt_cpu.dylib"):
        assert (libraries / name).is_file(), name
    # Loading the Metal plugin must resolve entirely within the relocated wheel.
    metal = c.CDLL(str(libraries / "libtvm_runtime_metal.dylib"))
    images = [Path(dyld._dyld_get_image_name(i).decode()).resolve()
              for i in range(dyld._dyld_image_count())]
    for name in ("libbaux_pjrt.dylib", "libpjrt_cpu.dylib", "libtvm_runtime_metal.dylib", "libtvm_runtime.dylib", "libtvm_ffi.dylib"):
        assert [path for path in images if path.name == name] == [(libraries / name).resolve()], name
    print("Relocated wheel: native XLA execution, TVM compiler payload, bundled dependencies and upstream licenses passed")
PY
