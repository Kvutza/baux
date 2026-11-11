#!/bin/sh
set -eu
tvm=$1
ffi=$2
bridge=$3
out=$(mkdir -p "$4" && cd "$4" && pwd)
cp "$tvm/apache_tvm-0.27.0.post1.dist-info/licenses/LICENSE" "$out/TVM-LICENSE"
cp "$ffi/apache_tvm_ffi-0.1.15.post2.dist-info/licenses/LICENSE" "$out/TVM-FFI-LICENSE"
cp "$ffi/apache_tvm_ffi-0.1.15.post2.dist-info/licenses/NOTICE" "$out/TVM-FFI-NOTICE"
clang++ -std=c++17 -O2 -fPIC -dynamiclib "$bridge" \
    -I"$ffi/tvm_ffi/include" \
    -L"$tvm/tvm/lib" -L"$ffi/tvm_ffi/lib" \
    -Wl,-needed_library,"$tvm/tvm/lib/libtvm_runtime.dylib" -ltvm_ffi \
    -Wl,-install_name,@rpath/libbaux_tvm.dylib -Wl,-rpath,@loader_path \
    -o "$out/libbaux_tvm.dylib"
