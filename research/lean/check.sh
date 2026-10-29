#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: check.sh PROJECT_DIR OUTPUT [SOURCE ...]" >&2
    exit 2
fi

project_dir=$1
output=$2
shift 2
case "$output" in
    /*) ;;
    *) output="$PWD/$output" ;;
esac

lake_bin=$(command -v lake || true)
if [ -z "$lake_bin" ]; then
    if [ -n "${ELAN_HOME:-}" ]; then
        elan_root=$ELAN_HOME
    elif [ -n "${HOME:-}" ]; then
        elan_root="$HOME/.elan"
    elif command -v getent >/dev/null 2>&1; then
        user_name=$(id -un)
        user_home=$(getent passwd "$user_name" | cut -d: -f6)
        elan_root="$user_home/.elan"
    elif [ -x /usr/bin/dscl ]; then
        user_name=$(id -un)
        user_home=$(/usr/bin/dscl . -read "/Users/$user_name" NFSHomeDirectory | awk '{print $2}')
        elan_root="$user_home/.elan"
    else
        elan_root=/.elan-unavailable
    fi
    lake_bin="$elan_root/bin/lake"
fi
if [ ! -x "$lake_bin" ]; then
    echo "baux: lake is unavailable; install the pinned Lean toolchain with elan" >&2
    exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/baux-lean.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
mkdir -p "$work_dir/project"
if [ "$#" -eq 0 ]; then
    cp -R "$project_dir"/. "$work_dir/project/"
else
    for source in "$@"; do
        case "$source" in
            "$project_dir"/*) relative=${source#"$project_dir"/} ;;
            *) echo "baux: undeclared Lean source path: $source" >&2; exit 1 ;;
        esac
        mkdir -p "$work_dir/project/$(dirname "$relative")"
        cp "$source" "$work_dir/project/$relative"
    done
fi
if ! (
    cd "$work_dir/project"
    "$lake_bin" --quiet --log-level=error exe cache get
    "$lake_bin" --log-level=warning build BauxResearch
    "$lake_bin" env lean --run Main.lean
) >"$output" 2>&1; then
    cat "$output" >&2
    exit 1
fi
