"""Audit each catalogue kernel's vector specialization in the native executable."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
output = Path(sys.argv[1])
binary = root / ".cache/baux/ocaml-build/default/bin/benchmarks.exe"
obj = (
    root
    / ".cache/baux/ocaml-build/default/lib/.baux_benchmarks.objs/native/baux_benchmarks__Formulas.o"
)
assembly = subprocess.check_output(["otool", "-tvV", str(obj)], text=True)
linked = subprocess.check_output(["nm", "-g", str(binary)], text=True)
(output / "assembly.txt").write_text(assembly)
bodies = re.split(r"(?m)^(_caml[^\n]+):\n", assembly)
symbols = dict(zip(bodies[1::2], bodies[2::2]))
catalogue = (root / "ocaml/lib/problem.ml").read_text()
entries = re.findall(r'entry\s+"([^"]+)"(.*?)(?=;\s*(?:entry|\]))', catalogue, re.S)
rows = []
for name, definition in entries:
    functions = re.findall(r"V\.(\w+)", definition)
    if len(functions) != 1:
        raise RuntimeError(f"Cannot identify SIMD implementation for {name}")
    function = functions[0]
    candidates = [
        (symbol, body)
        for symbol, body in symbols.items()
        if re.search(r"__Formulas__" + re.escape(function) + r"_\d+_\d+_code$", symbol)
    ]
    vector = [
        (symbol, body)
        for symbol, body in candidates
        if re.search(r"\.(?:2d|16b)\b|\bldr\s+q\d+", body)
    ]
    if not vector:
        raise RuntimeError(f"No vector code found for {name}")
    # A closed specialization must not dispatch numerical operators through
    # OCaml generic apply trampolines.
    clean = [
        (symbol, body)
        for symbol, body in vector
        if "caml_apply" not in body and not re.search(r"\bblr\b", body)
    ]
    if not clean:
        raise RuntimeError(f"No specialization without indirect calls for {name}")
    for symbol, _ in clean:
        if symbol not in linked:
            raise RuntimeError(f"Kernel symbol is not linked: {symbol}")
    rows.append(
        {"id": name, "function": function, "symbols": [symbol for symbol, _ in clean]}
    )
if len(rows) != 56:
    raise RuntimeError(f"Expected 56 catalogue audit entries, found {len(rows)}")
math_symbols = [
    f"_Sleef_{op}d2_u10advsimd" for op in ["sin", "cos", "exp", "log", "pow"]
]
for symbol in math_symbols:
    if symbol not in linked:
        raise RuntimeError(f"Missing linked vector math routine {symbol}")
report = {
    "catalogue": rows,
    "vector_math": math_symbols,
    "formulas_object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "scope": "Formula-object instruction evidence with linked-symbol verification; not a formal proof of complete instruction coverage.",
}
(output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Native SIMD audit passed for {len(rows)} catalogue entries.")
