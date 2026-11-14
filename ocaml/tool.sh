#!/bin/sh
set -eu
root=$(pwd)
if [ ! -f "$root/ocaml/dune-project" ]; then
    echo 'Run this target from the BAUX repository root.' >&2
    exit 2
fi
if [ "${1:-}" = --program-libs ]; then
    export BAUX_PROGRAM_LIB="$2"
    shift 2
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
        exec opam exec --switch="$switch" -- ocamlformat --enable-outside-detected-project --inplace "$root"/ocaml/lib/*.ml "$root"/ocaml/lib/*.mli "$root"/ocaml/hardware/*.ml "$root"/ocaml/bin/*.ml "$root"/ocaml/pybridge/*.ml "$root"/ocaml/pybridge/*.mli "$root"/ocaml/tests/test_*.ml
        ;;
    pyext)
        "$root/ocaml/prepare_sleef.sh" "$root/.cache/baux/sleef-3.9.0"
        py_inc=${BAUX_PYTHON_INCLUDE:-$(pixi run -e baux python -c "import sysconfig; print(sysconfig.get_path('include'))")}
        ext_sfx=${BAUX_PYTHON_EXT_SUFFIX:-$(pixi run -e baux python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")}
        out_dir="${2:-$root/.cache/baux/ocaml-pyext}"
        mkdir -p "$out_dir"
        BAUX_BUILD_PYEXT=1 BAUX_PYTHON_INCLUDE="$py_inc" \
            opam exec --switch="$switch" -- dune build --profile release --root "$root/ocaml" --build-dir "$build" pybridge/py_ext.so
        rm -f "$out_dir/_baux_oxcaml$ext_sfx"
        cp "$build/default/pybridge/py_ext.so" "$out_dir/_baux_oxcaml$ext_sfx"
        chmod 755 "$out_dir/_baux_oxcaml$ext_sfx"
        printf '%s\n' "$out_dir/_baux_oxcaml$ext_sfx"
        ;;
    program-fmt)
        exec opam exec --switch="$switch" -- ocamlformat --enable-outside-detected-project --inplace "$root"/ocaml/program/*.ml "$root"/ocaml/program/*.mli
        ;;
    hardware)
        shift
        exec opam exec --switch="$switch" -- dune exec --profile release --root "$root/ocaml" --build-dir "$build" bin/hardware_bench.exe -- "$@"
        ;;
    list|eval|batch|resolve|experiment)
        exec opam exec --switch="$switch" -- dune exec --profile release --root "$root/ocaml" --build-dir "$build" bin/benchmarks.exe -- "$@"
        ;;
    extraction)
        shift
        exec "$root/ocaml/extraction/bin/aeneas" "$@"
        ;;
    extraction-build)
        exec opam exec --switch="$switch" -- dune build --profile release --root "$root/ocaml/extraction/src" main.exe
        ;;
    *) echo 'Usage: buck2w run //ocaml:tool -- deps|build|hardware-build|test|test-flow|fmt|pyext|list|eval|batch|resolve|experiment|hardware|extraction|extraction-build' >&2; exit 2 ;;
esac
