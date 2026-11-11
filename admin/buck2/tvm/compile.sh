#!/bin/sh
set -eu
project=$PWD
while [ ! -f "$project/.buckconfig" ]; do
  [ "$project" != / ] || { echo 'repository root not found' >&2; exit 2; }
  project=$(dirname "$project")
done
python="$project/.pixi/envs/$1/bin/python"
wheel=$2
source=$3
out=$4
package=$(mktemp -d "${TMPDIR:-/tmp}/baux-tvm-compiler.XXXXXX")
trap 'status=$?; rm -rf "$package"; exit "$status"' EXIT HUP INT TERM
"$python" -m zipfile -e "$wheel" "$package"
export PYTHONPATH="$package"
"$python" - "$source" <<'PY' | "$python" -m baux.experimental._tvm "$out"
import json
import sys
from pathlib import Path
specs = [([4], "float32")] * 2
json.dump(dict(source=Path(sys.argv[1]).read_text(), entry="main", inputs=specs,
               outputs=specs, target="llvm"), sys.stdout)
PY
