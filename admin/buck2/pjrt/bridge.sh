#!/bin/sh
set -eu
plugin=$1
header=$2
source=$3
out=$4
clang++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic -fPIC -dynamiclib "$source/bridge.cc" "$source/buffer.cc" \
    -I"$(dirname "$header")" "$plugin" -Wl,-install_name,@rpath/libbaux_pjrt.dylib \
    -Wl,-rpath,@loader_path -o "$out"
