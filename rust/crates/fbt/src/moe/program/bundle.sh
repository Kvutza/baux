#!/bin/sh
set -eu
mkdir -p "$5"
native_name=$(basename "$1")
cp "$1" "$5/$native_name"
ln -s "$native_name" "$5/libbaux_program.dylib"
cp "$2" "$5/libbaux_iree.dylib"
cp "$3" "$5/portable.h"
cp "$4" "$5/model.h"
destination=$5
shift 5
for dependency in "$@"; do
    cp "$dependency" "$destination/"
done
