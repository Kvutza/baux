#!/bin/sh
set -eu
VERSION="3.9.0"
SHA256="af60856abac08a3b5e72a8d156dd71fec1f7ac23de8ee67793f45f9edcdf0908"
target="$1"
install="$target/install"
if [ -f "$install/lib/libsleef.a" ]; then
    exit 0
fi
mkdir -p "$target"
archive="$target/source.tar.gz"
if [ ! -f "$archive" ]; then
    curl -sSL -o "$archive" "https://codeload.github.com/shibatch/sleef/tar.gz/refs/tags/$VERSION"
fi
actual=$(shasum -a 256 "$archive" | awk '{print $1}')
if [ "$actual" != "$SHA256" ]; then
    echo "SLEEF checksum mismatch: got $actual, expected $SHA256" >&2
    exit 1
fi
tar -xzf "$archive" -C "$target"
cmake -S "$target/sleef-$VERSION" -B "$target/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install" \
    -DBUILD_SHARED_LIBS=OFF \
    -DSLEEF_BUILD_TESTS=OFF \
    -DSLEEF_BUILD_DFT=OFF \
    -DSLEEF_BUILD_QUAD=OFF \
    -DSLEEF_ENABLE_TLFLOAT=OFF
cmake --build "$target/build" --parallel 4
cmake --install "$target/build"
