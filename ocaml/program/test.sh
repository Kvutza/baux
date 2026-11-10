#!/bin/sh
set -eu
root=$(pwd)
opam exec --switch="${BAUX_OCAML_SWITCH:-5.2.0+ox}" -- \
  dune exec --profile release --root "$root/ocaml" \
  --build-dir "$root/.cache/baux/ocaml-program" program/testprogram.exe
