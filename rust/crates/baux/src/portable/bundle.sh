#!/bin/sh
set -eu
mkdir -p "$4"
native_name=$(basename "$1")
cp "$1" "$4/$native_name"
ln -s "$native_name" "$4/libbaux_program.dylib"
cp "$2" "$4/libbaux_iree.dylib"
cp "$3" "$4/portable.h"
cp "$5" "$4/libbaux_tvm.dylib"
cp "$6" "$4/libtvm_runtime.dylib"
cp "$7" "$4/libtvm_ffi.dylib"
cp "${8}" "$4/TVM-LICENSE"
cp "${9}" "$4/TVM-FFI-LICENSE"
cp "${10}" "$4/TVM-FFI-NOTICE"
cp "${11}" "$4/libbaux_pjrt.dylib"
cp "${12}" "$4/libpjrt_cpu.dylib"
cp "${13}" "$4/PJRT-LICENSE"
