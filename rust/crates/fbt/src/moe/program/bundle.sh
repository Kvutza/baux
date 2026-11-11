#!/bin/sh
set -eu
mkdir -p "$5"
native_name=$(basename "$1")
cp "$1" "$5/$native_name"
ln -s "$native_name" "$5/libbaux_program.dylib"
cp "$2" "$5/libbaux_iree.dylib"
cp "$3" "$5/portable.h"
cp "$4" "$5/model.h"
cp "$6" "$5/libbaux_tvm.dylib"
cp "$7" "$5/libtvm_runtime.dylib"
cp "$8" "$5/libtvm_ffi.dylib"
cp "${9}" "$5/TVM-LICENSE"
cp "${10}" "$5/TVM-FFI-LICENSE"
cp "${11}" "$5/TVM-FFI-NOTICE"
cp "${12}" "$5/libbaux_pjrt.dylib"
cp "${13}" "$5/libpjrt_cpu.dylib"
cp "${14}" "$5/PJRT-LICENSE"
