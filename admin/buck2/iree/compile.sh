#!/bin/sh
set -eu
compiler_dir=$(cd "$(dirname "$4")" && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/iree-compile.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM
cp "$4" "$tmp_dir/iree-compile"
chmod +x "$tmp_dir/iree-compile"
ln -s "$compiler_dir/libIREECompiler.dylib" "$tmp_dir/libIREECompiler.dylib"
"$tmp_dir/iree-compile" "$1" -o "$2" --iree-input-type=stablehlo \
  --iree-hal-target-backends="$3" --iree-metal-compile-to-metallib=false \
  --iree-input-demote-f64-to-f32=false --iree-input-demote-f32-to-f16=false \
  --iree-input-demote-i64-to-i32=false --iree-input-promote-bf16-to-f32=false \
  --iree-input-promote-f16-to-f32=false
