#!/bin/sh
set -eu
project=$PWD
while [ ! -f "$project/.buckconfig" ]; do
  [ "$project" != / ] || { echo 'repository root not found' >&2; exit 2; }
  project=$(dirname "$project")
done
exec "$project/.venv/bin/iree-compile" "$1" -o "$2" --iree-input-type=stablehlo \
  --iree-hal-target-backends="$3" --iree-metal-compile-to-metallib=false \
  --iree-input-demote-f64-to-f32=false --iree-input-demote-f32-to-f16=false \
  --iree-input-demote-i64-to-i32=false --iree-input-promote-bf16-to-f32=false \
  --iree-input-promote-f16-to-f32=false
