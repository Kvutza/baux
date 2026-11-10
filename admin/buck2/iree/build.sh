#!/bin/sh
set -eu
source_dir=$(cd "$1" && pwd)
flatcc_dir=$(cd "$2" && pwd)
bridge_dir=$(cd "$3" && pwd)
output_dir=$(mkdir -p "$4" && cd "$4" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-iree.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM
cp -R "$source_dir" "$build_dir/source"
chmod -R u+w "$build_dir/source"
cp -R "$flatcc_dir/." "$build_dir/source/third_party/flatcc/"
cmake -S "$bridge_dir" -B "$build_dir/build" -G Ninja \
    -DIREE_SOURCE="$build_dir/source" -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_BUILD_BENCHMARKS=OFF \
    -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_BINDINGS_TFLITE=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_METAL=ON \
    -DIREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE=OFF \
    -DIREE_ENABLE_WERROR_FLAG=OFF -DIREE_ENABLE_CCACHE=OFF
cmake --build "$build_dir/build" --target baux_iree --parallel 8
cp "$build_dir/build/libbaux_iree.dylib" "$output_dir/"
