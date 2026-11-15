"""Pure-NumPy WebAssembly implementation of baux's 56-function OxCaml benchmark suite.

Matches the domain definitions, rules, default dimensions, and surface formulas of
`admin/ui/bench_suite.ml` / `ocaml/lib/formulas.ml` for in-browser Pyodide execution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_LANG_C = np.array([[3.0, 5.0], [5.0, 2.0], [2.0, 1.0], [1.0, 4.0], [7.0, 9.0]])
_LANG_W = np.array([1.0, 2.0, 5.0, 2.0, 3.0])

_HART3_A = np.array(
    [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ]
)
_HART3_P = 1e-4 * np.array(
    [
        [3689.0, 1170.0, 2673.0],
        [4699.0, 4387.0, 7470.0],
        [1091.0, 8732.0, 5547.0],
        [381.0, 5743.0, 8828.0],
    ]
)
_HART6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_HART6_P = 1e-4 * np.array(
    [
        [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
        [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
        [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
        [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
    ]
)
_HART_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

_SHEKEL_C = np.array(
    [
        [4.0, 4.0, 4.0, 4.0],
        [1.0, 1.0, 1.0, 1.0],
        [8.0, 8.0, 8.0, 8.0],
        [6.0, 6.0, 6.0, 6.0],
        [3.0, 7.0, 3.0, 7.0],
        [2.0, 9.0, 2.0, 9.0],
        [5.0, 3.0, 5.0, 3.0],
        [8.0, 1.0, 8.0, 1.0],
        [6.0, 2.0, 6.0, 2.0],
        [7.0, 3.6, 7.0, 3.6],
    ]
)
_SHEKEL_B = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _branin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        (b - 5.1 * a**2 / (4.0 * math.pi**2) + 5.0 * a / math.pi - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * math.pi)) * np.cos(a)
        + 10.0
    )


def _goldstein(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    t1 = 1.0 + (a + b + 1.0) ** 2 * (
        19.0 - 14.0 * a + 3.0 * a**2 - 14.0 * b + 6.0 * a * b + 3.0 * b**2
    )
    t2 = 30.0 + (2.0 * a - 3.0 * b) ** 2 * (
        18.0 - 32.0 * a + 12.0 * a**2 + 48.0 * b - 36.0 * a * b + 27.0 * b**2
    )
    return t1 * t2


def _hartmann(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    a, p = (_HART3_A, _HART3_P) if d == 3 else (_HART6_A, _HART6_P)
    diff2 = (x[:, None, :] - p[None, :, :]) ** 2
    exponents = -np.sum(diff2 * a[None, :, :], axis=2)
    return -np.sum(_HART_ALPHA[None, :] * np.exp(exponents), axis=1)


def _powsum(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def _surface(name: str, x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    if name == "sphere":
        return np.sum(x * x, axis=1)
    if name == "rosenbrock":
        return np.sum(
            100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1
        )
    if name == "ackley":
        s1 = np.mean(x * x, axis=1)
        s2 = np.mean(np.cos(2.0 * math.pi * x), axis=1)
        return -20.0 * np.exp(-0.2 * np.sqrt(s1)) - np.exp(s2) + 20.0 + math.e
    if name == "rastrigin":
        return 10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * math.pi * x), axis=1)
    if name == "griewank":
        idx = np.sqrt(np.arange(1, d + 1, dtype=np.float64))
        return 1.0 + np.sum(x * x, axis=1) / 4000.0 - np.prod(np.cos(x / idx), axis=1)
    if name == "schwefel":
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    if name == "levy":
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(math.pi * w[:, 0]) ** 2
        mid = np.sum(
            (w[:, :-1] - 1.0) ** 2
            * (1.0 + 10.0 * np.sin(math.pi * w[:, :-1] + 1.0) ** 2),
            axis=1,
        )
        last = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * w[:, -1]) ** 2)
        return term1 + mid + last
    if name == "sum-squares":
        w = np.arange(1, d + 1, dtype=np.float64)
        return np.sum(w * (x * x), axis=1)
    if name == "rotated-hyperellipsoid":
        w = np.arange(d, 0, -1, dtype=np.float64)
        return np.sum(w * (x * x), axis=1)
    if name == "sum-powers":
        pows = np.arange(2, d + 2, dtype=np.float64)
        return np.sum(np.abs(x) ** pows, axis=1)
    if name == "trid":
        return np.sum((x - 1.0) ** 2, axis=1) - np.sum(x[:, :-1] * x[:, 1:], axis=1)
    if name == "zakharov":
        a = np.sum(0.5 * np.arange(1, d + 1, dtype=np.float64) * x, axis=1)
        return np.sum(x * x, axis=1) + a**2 + a**4
    if name == "dixon-price":
        w = np.arange(2, d + 1, dtype=np.float64)
        return (x[:, 0] - 1.0) ** 2 + np.sum(
            w * (2.0 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1
        )
    if name == "michalewicz":
        i = np.arange(1, d + 1, dtype=np.float64)
        return -np.sum(np.sin(x) * np.sin(i * x * x / math.pi) ** 20, axis=1)
    if name == "styblinski-tang":
        return 0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=1)
    if name == "perm-zero":
        j = np.arange(1, d + 1, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, d + 1):
            inner = np.sum((j + 10.0) * (x**k - (1.0 / j) ** k), axis=1)
            out += inner * inner
        return out
    if name == "perm":
        j = np.arange(1, d + 1, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for k in range(1, d + 1):
            inner = np.sum((j**k + 0.5) * ((x / j) ** k - 1.0), axis=1)
            out += inner * inner
        return out
    if name == "powell":
        a, b, c, dv = x[:, 0::4], x[:, 1::4], x[:, 2::4], x[:, 3::4]
        return np.sum(
            (a + 10.0 * b) ** 2
            + 5.0 * (c - dv) ** 2
            + (b - 2.0 * c) ** 4
            + 10.0 * (a - dv) ** 4,
            axis=1,
        )
    if name == "bukin6":
        return 100.0 * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0] ** 2)) + 0.01 * np.abs(
            x[:, 0] + 10.0
        )
    if name == "cross-in-tray":
        a, b = x[:, 0], x[:, 1]
        exp_t = np.exp(np.abs(100.0 - np.sqrt(a * a + b * b) / math.pi))
        return -1e-4 * (np.abs(np.sin(a) * np.sin(b) * exp_t) + 1.0) ** 0.1
    if name == "drop-wave":
        r2 = np.sum(x * x, axis=1)
        return -(1.0 + np.cos(12.0 * np.sqrt(r2))) / (2.0 + 0.5 * r2)
    if name == "eggholder":
        a, b = x[:, 0], x[:, 1] + 47.0
        return -b * np.sin(np.sqrt(np.abs(0.5 * a + b))) - a * np.sin(
            np.sqrt(np.abs(a - b))
        )
    if name == "gramacy-lee":
        a = x[:, 0]
        return np.sin(10.0 * math.pi * a) / (2.0 * a) + (a - 1.0) ** 4
    if name == "holder-table":
        a, b = x[:, 0], x[:, 1]
        return -np.abs(
            np.sin(a)
            * np.cos(b)
            * np.exp(np.abs(1.0 - np.sqrt(a * a + b * b) / math.pi))
        )
    if name == "langermann":
        r2 = (x[:, None, :2] - _LANG_C[None, :, :]) ** 2
        r2 = np.sum(r2, axis=2)
        return np.sum(_LANG_W * np.exp(-r2 / math.pi) * np.cos(math.pi * r2), axis=1)
    if name == "levy13":
        a, b = x[:, 0], x[:, 1]
        return (
            np.sin(3.0 * math.pi * a) ** 2
            + (a - 1.0) ** 2 * (1.0 + np.sin(3.0 * math.pi * b) ** 2)
            + (b - 1.0) ** 2 * (1.0 + np.sin(2.0 * math.pi * b) ** 2)
        )
    if name == "schaffer2":
        a, b = x[:, 0] ** 2, x[:, 1] ** 2
        return 0.5 + (np.sin(a - b) ** 2 - 0.5) / (1.0 + 0.001 * (a + b)) ** 2
    if name == "schaffer4":
        a, b = x[:, 0] ** 2, x[:, 1] ** 2
        return (
            0.5
            + (np.cos(np.sin(np.abs(a - b))) ** 2 - 0.5) / (1.0 + 0.001 * (a + b)) ** 2
        )
    if name == "shubert":
        k = np.arange(1, 6, dtype=np.float64)[None, :]
        s0 = np.sum(k * np.cos((k + 1.0) * x[:, 0:1] + k), axis=1)
        s1 = np.sum(k * np.cos((k + 1.0) * x[:, 1:2] + k), axis=1)
        return s0 * s1
    if name in ("bohachevsky1", "bohachevsky2", "bohachevsky3"):
        a, b = x[:, 0], x[:, 1]
        u, v = 3.0 * math.pi * a, 4.0 * math.pi * b
        base = a * a + 2.0 * b * b
        if name == "bohachevsky1":
            return base + 0.7 - 0.3 * np.cos(u) - 0.4 * np.cos(v)
        if name == "bohachevsky2":
            return base + 0.3 - 0.3 * np.cos(u) * np.cos(v)
        return base + 0.3 - 0.3 * np.cos(u + v)
    if name == "booth":
        a, b = x[:, 0], x[:, 1]
        return (a + 2.0 * b - 7.0) ** 2 + (2.0 * a + b - 5.0) ** 2
    if name == "matyas":
        return 0.26 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 0.48 * x[:, 0] * x[:, 1]
    if name == "mccormick":
        a, b = x[:, 0], x[:, 1]
        return np.sin(a + b) + (a - b) ** 2 - 1.5 * a + 2.5 * b + 1.0
    if name == "power-sum":
        targets = (8.0, 18.0, 44.0, 114.0)
        out = np.zeros(n, dtype=np.float64)
        for k, tgt in enumerate(targets, 1):
            out += (np.sum(x**k, axis=1) - tgt) ** 2
        return out
    if name == "three-hump-camel":
        a, b = x[:, 0], x[:, 1]
        return 2.0 * a**2 - 1.05 * a**4 + a**6 / 6.0 + a * b + b**2
    if name == "six-hump-camel":
        a, b = x[:, 0], x[:, 1]
        return (
            (4.0 - 2.1 * a**2 + a**4 / 3.0) * a**2 + a * b + (-4.0 + 4.0 * b**2) * b**2
        )
    if name == "de-jong5":
        a_grid = np.array([16.0 * ((i % 5) - 2) for i in range(25)])
        b_grid = np.array([16.0 * ((i // 5) - 2) for i in range(25)])
        idx = np.arange(1, 26, dtype=np.float64)
        denom = (
            idx[None, :]
            + (x[:, 0:1] - a_grid[None, :]) ** 6
            + (x[:, 1:2] - b_grid[None, :]) ** 6
        )
        return 1.0 / (0.002 + np.sum(1.0 / denom, axis=1))
    if name == "easom":
        a, b = x[:, 0], x[:, 1]
        return -(np.cos(a) * np.cos(b)) * np.exp(
            -((a - math.pi) ** 2 + (b - math.pi) ** 2)
        )
    if name == "beale":
        a, b = x[:, 0], x[:, 1]
        return (
            (1.5 - a + a * b) ** 2
            + (2.25 - a + a * b**2) ** 2
            + (2.625 - a + a * b**3) ** 2
        )
    if name == "branin":
        return _branin(x[:, 0], x[:, 1])
    if name == "colville":
        a, b, c, dv = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return (
            100.0 * (a**2 - b) ** 2
            + (a - 1.0) ** 2
            + (c - 1.0) ** 2
            + 90.0 * (c**2 - dv) ** 2
            + 10.1 * ((b - 1.0) ** 2 + (dv - 1.0) ** 2)
            + 19.8 * (b - 1.0) * (dv - 1.0)
        )
    if name == "forrester":
        a = x[:, 0]
        return (6.0 * a - 2.0) ** 2 * np.sin(12.0 * a - 4.0)
    if name == "goldstein-price":
        return _goldstein(x[:, 0], x[:, 1])
    if name in ("hartmann3", "hartmann6"):
        return _hartmann(x)
    if name == "hartmann4":
        return (1.1 + _hartmann(x[:, :4])) / 0.839
    if name == "shekel":
        diff2 = np.sum((x[:, None, :] - _SHEKEL_C[None, :, :]) ** 2, axis=2)
        return -np.sum(1.0 / (_SHEKEL_B[None, :] + diff2), axis=1)
    if name == "sphere6-rescaled":
        w = 2.0 ** np.arange(1, 7, dtype=np.float64)
        return (np.sum(w * (x[:, :6] ** 2), axis=1) - 1745.0) / 899.0
    if name == "rosenbrock4-rescaled":
        at = 15.0 * x[:, :4] - 5.0
        val = np.sum(
            100.0 * (at[:, 1:] - at[:, :-1] ** 2) ** 2 + (1.0 - at[:, :-1]) ** 2, axis=1
        )
        return (val - 382700.0) / 375500.0
    if name == "branin-rescaled":
        return (_branin(15.0 * x[:, 0] - 5.0, 15.0 * x[:, 1]) - 54.81) / 51.95
    if name == "branin-modified":
        return _branin(x[:, 0], x[:, 1]) + 5.0 * x[:, 0]
    if name == "forrester-low-fidelity":
        a = x[:, 0]
        f = (6.0 * a - 2.0) ** 2 * np.sin(12.0 * a - 4.0)
        return 0.5 * f + 10.0 * (a - 0.5) + 5.0
    if name == "goldstein-price-rescaled":
        return (
            np.log(_goldstein(4.0 * x[:, 0] - 2.0, 4.0 * x[:, 1] - 2.0)) - 8.693
        ) / 2.427
    if name == "hartmann6-rescaled":
        return (_hartmann(x[:, :6]) - 2.58) / 1.94
    raise KeyError(f"Unknown OxCaml benchmark: {name!r}")


def _unif(lo: float, hi: float):
    return lambda d: [[float(lo), float(hi)]] * d


def _fixed(bounds: list[tuple[float, float]]):
    return lambda _d: [[float(lo), float(hi)] for lo, hi in bounds]


def _permbounds(d: int) -> list[list[float]]:
    return [[-float(d), float(d)]] * d


def _tridbounds(d: int) -> list[list[float]]:
    b = float(d * d)
    return [[-b, b]] * d


_SPECS: dict[str, tuple[str, str, tuple[str, int], Any]] = {
    "ackley": ("Ackley", "multimodal", ("at_least", 1), _unif(-32.768, 32.768)),
    "bukin6": (
        "Bukin N.6",
        "multimodal",
        ("fixed", 2),
        _fixed([(-15.0, -5.0), (-3.0, 3.0)]),
    ),
    "cross-in-tray": ("Cross-in-Tray", "multimodal", ("fixed", 2), _unif(-10.0, 10.0)),
    "drop-wave": ("Drop-Wave", "multimodal", ("fixed", 2), _unif(-5.12, 5.12)),
    "eggholder": ("Eggholder", "multimodal", ("fixed", 2), _unif(-512.0, 512.0)),
    "gramacy-lee": ("Gramacy & Lee (1D)", "multimodal", ("fixed", 1), _unif(0.5, 2.5)),
    "griewank": ("Griewank", "multimodal", ("at_least", 1), _unif(-600.0, 600.0)),
    "holder-table": ("Holder Table", "multimodal", ("fixed", 2), _unif(-10.0, 10.0)),
    "langermann": ("Langermann", "multimodal", ("fixed", 2), _unif(0.0, 10.0)),
    "levy": ("Levy", "multimodal", ("at_least", 1), _unif(-10.0, 10.0)),
    "levy13": ("Levy N.13", "multimodal", ("fixed", 2), _unif(-10.0, 10.0)),
    "rastrigin": ("Rastrigin", "multimodal", ("at_least", 1), _unif(-5.12, 5.12)),
    "schaffer2": ("Schaffer N.2", "multimodal", ("fixed", 2), _unif(-100.0, 100.0)),
    "schaffer4": ("Schaffer N.4", "multimodal", ("fixed", 2), _unif(-100.0, 100.0)),
    "schwefel": ("Schwefel", "multimodal", ("at_least", 1), _unif(-500.0, 500.0)),
    "shubert": ("Shubert", "multimodal", ("fixed", 2), _unif(-10.0, 10.0)),
    "bohachevsky1": ("Bohachevsky N.1", "bowl", ("fixed", 2), _unif(-100.0, 100.0)),
    "bohachevsky2": ("Bohachevsky N.2", "bowl", ("fixed", 2), _unif(-100.0, 100.0)),
    "bohachevsky3": ("Bohachevsky N.3", "bowl", ("fixed", 2), _unif(-100.0, 100.0)),
    "perm-zero": ("Perm 0, d, β", "bowl", ("at_least", 1), _permbounds),
    "rotated-hyperellipsoid": (
        "Rotated Hyper-Ellipsoid",
        "bowl",
        ("at_least", 1),
        _unif(-65.536, 65.536),
    ),
    "sphere": ("Sphere", "bowl", ("at_least", 1), _unif(-5.12, 5.12)),
    "sum-powers": (
        "Sum of Different Powers",
        "bowl",
        ("at_least", 1),
        _unif(-1.0, 1.0),
    ),
    "sum-squares": ("Sum Squares", "bowl", ("at_least", 1), _unif(-10.0, 10.0)),
    "trid": ("Trid", "bowl", ("at_least", 2), _tridbounds),
    "booth": ("Booth", "plate", ("fixed", 2), _unif(-10.0, 10.0)),
    "matyas": ("Matyas", "plate", ("fixed", 2), _unif(-10.0, 10.0)),
    "mccormick": (
        "McCormick",
        "plate",
        ("fixed", 2),
        _fixed([(-1.5, 4.0), (-3.0, 4.0)]),
    ),
    "power-sum": ("Power Sum (4D)", "plate", ("fixed", 4), _unif(0.0, 4.0)),
    "zakharov": ("Zakharov", "plate", ("at_least", 1), _unif(-5.0, 10.0)),
    "three-hump-camel": ("Three-Hump Camel", "valley", ("fixed", 2), _unif(-5.0, 5.0)),
    "six-hump-camel": (
        "Six-Hump Camel",
        "valley",
        ("fixed", 2),
        _fixed([(-3.0, 3.0), (-2.0, 2.0)]),
    ),
    "dixon-price": ("Dixon-Price", "valley", ("at_least", 2), _unif(-10.0, 10.0)),
    "rosenbrock": ("Rosenbrock", "valley", ("at_least", 2), _unif(-5.0, 10.0)),
    "de-jong5": ("De Jong N.5", "ridge", ("fixed", 2), _unif(-65.536, 65.536)),
    "easom": ("Easom", "ridge", ("fixed", 2), _unif(-100.0, 100.0)),
    "michalewicz": ("Michalewicz", "ridge", ("at_least", 1), _unif(0.0, math.pi)),
    "beale": ("Beale", "other", ("fixed", 2), _unif(-4.5, 4.5)),
    "branin": ("Branin", "other", ("fixed", 2), _fixed([(-5.0, 10.0), (0.0, 15.0)])),
    "colville": ("Colville (4D)", "other", ("fixed", 4), _unif(-10.0, 10.0)),
    "forrester": ("Forrester (1D)", "other", ("fixed", 1), _unif(0.0, 1.0)),
    "goldstein-price": ("Goldstein-Price", "other", ("fixed", 2), _unif(-2.0, 2.0)),
    "hartmann3": ("Hartmann (3D)", "other", ("fixed", 3), _unif(0.0, 1.0)),
    "hartmann4": ("Hartmann (4D)", "other", ("fixed", 4), _unif(0.0, 1.0)),
    "hartmann6": ("Hartmann (6D)", "other", ("fixed", 6), _unif(0.0, 1.0)),
    "perm": ("Perm d, β", "other", ("at_least", 1), _permbounds),
    "powell": ("Powell", "other", ("multiple_of", 4), _unif(-4.0, 5.0)),
    "shekel": ("Shekel (4D)", "other", ("fixed", 4), _unif(0.0, 10.0)),
    "styblinski-tang": ("Styblinski-Tang", "other", ("at_least", 1), _unif(-5.0, 5.0)),
    "sphere6-rescaled": ("Sphere-6 (Rescaled)", "bowl", ("fixed", 6), _unif(0.0, 1.0)),
    "rosenbrock4-rescaled": (
        "Rosenbrock-4 (Rescaled)",
        "valley",
        ("fixed", 4),
        _unif(0.0, 1.0),
    ),
    "branin-rescaled": ("Branin (Rescaled)", "other", ("fixed", 2), _unif(0.0, 1.0)),
    "branin-modified": (
        "Branin (Modified)",
        "other",
        ("fixed", 2),
        _fixed([(-5.0, 10.0), (0.0, 15.0)]),
    ),
    "forrester-low-fidelity": (
        "Forrester Low-Fi (1D)",
        "other",
        ("fixed", 1),
        _unif(0.0, 1.0),
    ),
    "goldstein-price-rescaled": (
        "Goldstein-Price (Rescaled)",
        "other",
        ("fixed", 2),
        _unif(0.0, 1.0),
    ),
    "hartmann6-rescaled": (
        "Hartmann-6 (Rescaled)",
        "other",
        ("fixed", 6),
        _unif(0.0, 1.0),
    ),
}


def _defaultdim(rule: tuple[str, int]) -> int:
    kind, val = rule
    if kind in ("fixed", "multiple_of"):
        return val
    return max(2, val)


def _validdim(rule: tuple[str, int], d: int) -> bool:
    if d <= 0:
        return False
    kind, val = rule
    if kind == "fixed":
        return d == val
    if kind == "at_least":
        return d >= val
    return d % val == 0


def catalogue() -> list[dict[str, Any]]:
    """Return metadata dictionaries for all 56 benchmark problems."""
    items = []
    for name, (label, family, rule, bounds_fn) in _SPECS.items():
        dd = _defaultdim(rule)
        items.append(
            {
                "id": name,
                "label": label,
                "family": family,
                "default_dim": dd,
                "scalable": rule[0] != "fixed",
                "bounds": bounds_fn(dd),
            }
        )
    return items


def describe(name: str, dim: int = 0) -> dict[str, Any]:
    """Return bounds and dimensionality metadata for `name` at `dim` (0 = default)."""
    if name not in _SPECS:
        raise KeyError(f"Unknown OxCaml benchmark: {name!r}")
    label, family, rule, bounds_fn = _SPECS[name]
    target_dim = _defaultdim(rule) if int(dim) == 0 else int(dim)
    if not _validdim(rule, target_dim):
        raise ValueError(f"Invalid dim {target_dim} for {name!r} ({rule})")
    return {
        "id": name,
        "label": label,
        "family": family,
        "dim": target_dim,
        "scalable": rule[0] != "fixed",
        "bounds": bounds_fn(target_dim),
    }


def eval_one(name: str, point: np.ndarray) -> float:
    """Evaluate minimization objective f(point)."""
    arr = np.asarray(point, dtype=np.float64).reshape(1, -1)
    return float(eval_batch(name, arr)[0])


def eval_batch(
    name: str,
    points: np.ndarray,
    backend: str = "wasm",
    workers: int = 1,
    block_pairs: int = 128,
) -> bytearray:
    """Evaluate (N, D) batch and return float64 buffer bytes compatible with oxcaml.py."""
    del backend, workers, block_pairs
    mat = np.ascontiguousarray(points, dtype=np.float64)
    if name not in _SPECS:
        raise KeyError(f"Unknown OxCaml benchmark: {name!r}")
    _, _, rule, _ = _SPECS[name]
    if mat.ndim != 2 or not _validdim(rule, mat.shape[1]):
        raise ValueError(f"Invalid shape {mat.shape} for {name!r}")
    out = np.ascontiguousarray(_surface(name, mat), dtype=np.float64)
    return bytearray(out.tobytes())


def refcount_probe(_obj: object) -> dict[str, int]:
    """Return synthetic borrow telemetry when running inside browser WASM."""
    return {"local_borrow_delta": 0, "unique_steal": 1}
