#!/bin/sh
set -eu
root=$(pwd)
if [ ! -f "$root/ocaml/dune-project" ]; then
    echo 'Run this target from the BAUX repository root.' >&2
    exit 2
fi
switch=${BAUX_OCAML_SWITCH:-5.2.0+ox}
build="$root/.cache/baux/ocaml-build"
export BAUX_SLEEF_PREFIX="$root/.cache/baux/sleef-3.9.0/install"
case "${1:-test}" in
    deps)
        exec "$root/ocaml/prepare_sleef.sh" "$root/.cache/baux/sleef-3.9.0"
        ;;
    build)
        exec opam exec --switch="$switch" -- dune build --profile release --root "$root/ocaml" --build-dir "$build" @all
        ;;
    hardware-build)
        exec opam exec --switch="$switch" -- dune build --profile release --root "$root/ocaml" --build-dir "$build" bin/hardware_bench.exe
        ;;
    test-flow)
        exec opam exec --switch="$switch" -- dune exec --profile release --root "$root/ocaml" --build-dir "$build" tests/test_flow.exe
        ;;
    test)
        "$root/ocaml/prepare_sleef.sh" "$root/.cache/baux/sleef-3.9.0"
        exec opam exec --switch="$switch" -- dune runtest --profile release --root "$root/ocaml" --build-dir "$build"
        ;;
    fmt)
        exec opam exec --switch="$switch" -- ocamlformat --enable-outside-detected-project --inplace "$root"/ocaml/lib/*.ml "$root"/ocaml/lib/*.mli "$root"/ocaml/hardware/*.ml "$root"/ocaml/bin/*.ml "$root"/ocaml/tests/test_*.ml
        ;;
    hardware)
        shift
        exec opam exec --switch="$switch" -- dune exec --profile release --root "$root/ocaml" --build-dir "$build" bin/hardware_bench.exe -- "$@"
        ;;
    list|eval|batch|resolve|experiment)
        exec opam exec --switch="$switch" -- dune exec --profile release --root "$root/ocaml" --build-dir "$build" bin/benchmarks.exe -- "$@"
        ;;
    *) echo 'Usage: buck2w run //ocaml:tool -- deps|build|hardware-build|test|test-flow|fmt|list|eval|batch|resolve|experiment|hardware' >&2; exit 2 ;;
esac
