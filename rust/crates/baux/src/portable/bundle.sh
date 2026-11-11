#!/bin/sh
set -eu
mkdir -p "$4"
native_name=$(basename "$1")
cp "$1" "$4/$native_name"
ln -s "$native_name" "$4/libbaux_program.dylib"
cp "$2" "$4/libbaux_iree.dylib"
cp "$3" "$4/portable.h"
destination=$4
shift 4
for dependency in "$@"; do
    cp "$dependency" "$destination/"
done
