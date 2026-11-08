"""Regenerate frozen fixtures; requires numpy/scipy and network (not test-time).

SciPy implementations are fetched at a fixed revision. SFU-only equations
below are separate scalar Python oracles, not an independent published codebase.
"""

import contextlib
import importlib.util
import pathlib
import sys
import tempfile
import types
import urllib.request

import numpy as np

REVISION = "b7f8d4533c66ebbb33b91b41056f3535a35845a7"
BASE = f"https://raw.githubusercontent.com/scipy/scipy/{REVISION}/benchmarks/benchmarks/go_benchmark_functions/"
MAPPING = {
    "ackley": "Ackley01",
    "bukin6": "Bukin06",
    "cross-in-tray": "CrossInTray",
    "drop-wave": "DropWave",
    "eggholder": "EggHolder",
    "griewank": "Griewank",
    "holder-table": "HolderTable",
    "langermann": "Langermann",
    "levy13": "Levy13",
    "rastrigin": "Rastrigin",
    "schaffer2": "Schaffer02",
    "schaffer4": "Schaffer04",
    "schwefel": "Schwefel26",
    "shubert": "Shubert01",
    "bohachevsky1": "Bohachevsky1",
    "bohachevsky2": "Bohachevsky2",
    "bohachevsky3": "Bohachevsky3",
    "perm-zero": "PermFunction02",
    "sphere": "Sphere",
    "sum-powers": "Sodp",
    "trid": "Trid",
    "matyas": "Matyas",
    "mccormick": "McCormick",
    "power-sum": "PowerSum",
    "zakharov": "Zacharov",
    "three-hump-camel": "ThreeHumpCamel",
    "six-hump-camel": "SixHumpCamel",
    "dixon-price": "DixonPrice",
    "rosenbrock": "Rosenbrock",
    "easom": "Easom",
    "michalewicz": "Michalewicz",
    "beale": "Beale",
    "branin": "Branin01",
    "colville": "Colville",
    "goldstein-price": "GoldsteinPrice",
    "hartmann3": "Hartmann3",
    "hartmann6": "Hartmann6",
    "perm": "PermFunction01",
    "powell": "Powell",
    "shekel": "Shekel10",
    "styblinski-tang": "StyblinskiTang",
}


def load_reference(directory):
    package = types.ModuleType("reference")
    package.__path__ = [str(directory)]
    sys.modules["reference"] = package
    # ASV's optional-import helper is infrastructure, not numerical code.
    common = types.ModuleType("reference.common")
    common.safe_import = contextlib.nullcontext
    sys.modules[common.__name__] = common
    for name in ["__init__.py", "go_benchmark.py"] + [
        f"go_funcs_{letter}.py" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]:
        destination = directory / name
        if not destination.exists():
            destination.write_bytes(urllib.request.urlopen(BASE + name).read())
    spec = importlib.util.spec_from_file_location(
        "reference.functions",
        directory / "__init__.py",
        submodule_search_locations=[str(directory)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference(module, name, x):
    if name == "powell" and len(x) > 4:
        return sum(module.Powell().fun(x[i : i + 4]) for i in range(0, len(x), 4))
    if name in MAPPING:
        obj = getattr(module, MAPPING[name])(len(x))
        if name == "hartmann3":
            obj.p[3, 0] = 0.0381  # SFU coefficient; SciPy uses 0.03815.
        if name == "shekel":
            obj.A[6] = [5, 3, 5, 3]  # SFU row; SciPy has [5, 5, 3, 3].
        result = obj.fun(x)
        if name == "colville":
            # SciPy squares x2 in its first term; SFU squares x1.
            result += 100 * ((x[0] ** 2 - x[1]) ** 2 - (x[0] - x[1] ** 2) ** 2)
        if name == "langermann":
            result = -result  # SFU's displayed positive sum.
        if name == "schwefel":
            result += len(x) * (418.9829 - 418.982887)
        return result
    if name == "hartmann4":
        obj = module.Hartmann6()
        obj.a, obj.p = obj.a[:, :4], obj.p[:, :4]
        return (1.1 + obj.fun(x)) / 0.839
    if name == "hartmann6-rescaled":
        return (reference(module, "hartmann6", x) - 2.58) / 1.94
    if name == "branin-rescaled":
        return (
            reference(module, "branin", np.array([15 * x[0] - 5, 15 * x[1]])) - 54.81
        ) / 51.95
    if name == "branin-modified":
        return reference(module, "branin", x) + 5 * x[0]
    if name == "goldstein-price-rescaled":
        return (np.log(reference(module, "goldstein-price", 4 * x - 2)) - 8.693) / 2.427
    if name == "rosenbrock4-rescaled":
        return (reference(module, "rosenbrock", 15 * x - 5) - 382700) / 375500
    if name == "sphere6-rescaled":
        return (sum(2 ** (i + 1) * v * v for i, v in enumerate(x)) - 1745) / 899
    if name == "gramacy-lee":
        return np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4
    if name == "levy":
        w = 1 + (x - 1) / 4
        return (
            np.sin(np.pi * w[0]) ** 2
            + sum((v - 1) ** 2 * (1 + 10 * np.sin(np.pi * v + 1) ** 2) for v in w[:-1])
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        )
    if name == "sum-squares":
        return sum((i + 1) * v * v for i, v in enumerate(x))
    if name == "rotated-hyperellipsoid":
        return sum(sum(v * v for v in x[: i + 1]) for i in range(len(x)))
    if name == "booth":
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    if name == "de-jong5":
        centers = [(a, b) for b in [-32, -16, 0, 16, 32] for a in [-32, -16, 0, 16, 32]]
        return 1 / (
            1 / 500
            + sum(
                1 / (i + 1 + (x[0] - a) ** 6 + (x[1] - b) ** 6)
                for i, (a, b) in enumerate(centers)
            )
        )
    if name == "forrester":
        return (6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4)
    if name == "forrester-low-fidelity":
        return 0.5 * reference(module, "forrester", x) + 10 * (x[0] - 0.5) + 5
    raise ValueError(name)


def generate(module):
    source = pathlib.Path(__file__).parents[1] / "lib" / "problem.ml"
    import re

    entries = re.findall(
        r'entry "([^"]+)" \w+ (any|two|\(Fixed \d+\)|\(At_least \d+\)|\(Multiple_of \d+\))',
        source.read_text(),
    )
    rows = [f"# SciPy revision {REVISION}", "# id\tdimension\tcoordinates\texpected"]
    for name, rule in entries:
        dimensions = (
            [2, 5, 12]
            if rule == "any" or "At_least" in rule
            else [2 if rule == "two" else int(re.search(r"\d+", rule)[0])]
        )
        if "Multiple_of" in rule:
            dimensions = [4, 8, 12]
        for n in dimensions:
            for sample in range(5):
                # Fixed rational points: no dependence on random generator versions.
                x = np.array(
                    [(((i + 3) * 17 + sample * 23) % 97 + 1) / 100 for i in range(n)]
                )
                if sample >= 3:
                    x[::2] *= -1
                expected = float(reference(module, name, x))
                assert np.isfinite(expected), (name, x)
                rows.append(
                    f"{name}\t{n}\t"
                    + ",".join(format(v, ".17g") for v in x)
                    + f"\t{expected:.17g}"
                )
    pathlib.Path(__file__).with_name("reference.tsv").write_text("\n".join(rows) + "\n")
    print(f"Generated {len(rows) - 2} reference cases for {len(entries)} functions.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        generate(
            load_reference(
                pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else directory)
            )
        )
