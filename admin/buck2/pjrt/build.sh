#!/bin/sh
set -eu
wheel=$1
header=$2
probe=$3
out=$(mkdir -p "$4" && cd "$4" && pwd)
unzip -p "$wheel" cyxla_pjrt_cpu/libpjrt_cpu.dylib > "$out/libpjrt_cpu.dylib"
unzip -p "$wheel" cyxla_pjrt_cpu-0.12.0.1.dist-info/licenses/LICENSE > "$out/PJRT-LICENSE"
clang++ -std=c++17 -O2 "$probe" -I"$(dirname "$header")" \
    "$out/libpjrt_cpu.dylib" -Wl,-rpath,@loader_path -o "$out/probe"
