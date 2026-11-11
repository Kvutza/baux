#!/bin/sh
set -eu
tvm_wheel=$1
ffi_wheel=$2
bridge=$3
out=$(mkdir -p "$4" && cd "$4" && pwd)
build=$(mktemp -d "${TMPDIR:-/tmp}/baux-tvm.XXXXXX")
trap 'status=$?; rm -rf "$build"; exit "$status"' EXIT HUP INT TERM
unzip -q "$tvm_wheel" 'tvm/lib/libtvm_runtime.dylib' -d "$build"
unzip -q "$ffi_wheel" 'tvm_ffi/include/*' 'tvm_ffi/lib/libtvm_ffi.dylib' -d "$build"
cp "$build/tvm/lib/libtvm_runtime.dylib" "$out/"
cp "$build/tvm_ffi/lib/libtvm_ffi.dylib" "$out/"
unzip -p "$tvm_wheel" 'apache_tvm-0.27.0.post1.dist-info/licenses/LICENSE' > "$out/TVM-LICENSE"
unzip -p "$ffi_wheel" 'apache_tvm_ffi-0.1.15.post2.dist-info/licenses/LICENSE' > "$out/TVM-FFI-LICENSE"
unzip -p "$ffi_wheel" 'apache_tvm_ffi-0.1.15.post2.dist-info/licenses/NOTICE' > "$out/TVM-FFI-NOTICE"
clang++ -std=c++17 -O2 -fPIC -dynamiclib "$bridge" \
    -I"$build/tvm_ffi/include" \
    -L"$out" -Wl,-needed_library,"$out/libtvm_runtime.dylib" -ltvm_ffi \
    -Wl,-install_name,@rpath/libbaux_tvm.dylib -Wl,-rpath,@loader_path \
    -o "$out/libbaux_tvm.dylib"
