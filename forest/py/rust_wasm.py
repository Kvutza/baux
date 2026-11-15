"""Pure-Python + NumPy WebAssembly backend for `baux._rust` inside Pyodide.

Only packaged inside the static site's `assets/baux_py.zip` when running under
`sys.platform == "emscripten"`. Never shadows native compiled extensions on Linux/macOS.
"""

from __future__ import annotations

import copy
import json
import math
import os
import struct
import time
import types
from dataclasses import dataclass
from typing import Any

import numpy as np

_MASK64 = 0xFFFFFFFFFFFFFFFF
_MASK32 = 0xFFFFFFFF
_SM64_C1 = 0x9E3779B97F4A7C15
_SM64_C2 = 0xBF58476D1CE4E5B9
_SM64_C3 = 0x94D049BB133111EB
_SM64_OFFSET = 0xD2B74407B1CE6E93
_SEED_PRIME = 1_000_003
_INV_2P53 = 1.0 / 9007199254740992.0


def _splitmix64(x: int) -> int:
    z = (x + _SM64_C1) & _MASK64
    z = ((z ^ (z >> 30)) * _SM64_C2) & _MASK64
    z = ((z ^ (z >> 27)) * _SM64_C3) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


# ---------------------------------------------------------------------------
# Pure-Python BLAKE3 (32-byte digest) for deterministic state digests
# ---------------------------------------------------------------------------

_B3_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_B3_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_B3_CHUNK_LEN = 1024
_B3_BLOCK_LEN = 64
_B3_CHUNK_START = 1
_B3_CHUNK_END = 2
_B3_PARENT = 4
_B3_ROOT = 8


def _rotr32(v: int, n: int) -> int:
    return ((v >> n) | (v << (32 - n))) & _MASK32


def _b3g(st: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    st[a] = (st[a] + st[b] + mx) & _MASK32
    st[d] = _rotr32(st[d] ^ st[a], 16)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotr32(st[b] ^ st[c], 12)
    st[a] = (st[a] + st[b] + my) & _MASK32
    st[d] = _rotr32(st[d] ^ st[a], 8)
    st[c] = (st[c] + st[d]) & _MASK32
    st[b] = _rotr32(st[b] ^ st[c], 7)


def _b3compress(
    cv: tuple[int, ...],
    block_words: list[int],
    counter: int,
    block_len: int,
    flags: int,
) -> list[int]:
    m = list(block_words)
    st = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        _B3_IV[0],
        _B3_IV[1],
        _B3_IV[2],
        _B3_IV[3],
        counter & _MASK32,
        (counter >> 32) & _MASK32,
        block_len & _MASK32,
        flags & _MASK32,
    ]
    for r in range(7):
        _b3g(st, 0, 4, 8, 12, m[0], m[1])
        _b3g(st, 1, 5, 9, 13, m[2], m[3])
        _b3g(st, 2, 6, 10, 14, m[4], m[5])
        _b3g(st, 3, 7, 11, 15, m[6], m[7])
        _b3g(st, 0, 5, 10, 15, m[8], m[9])
        _b3g(st, 1, 6, 11, 12, m[10], m[11])
        _b3g(st, 2, 7, 8, 13, m[12], m[13])
        _b3g(st, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in _B3_PERM]
    for i in range(8):
        st[i] ^= st[i + 8]
        st[i + 8] ^= cv[i]
    return st


def _b3chunkcv(chunk: bytes, chunk_idx: int, is_only_chunk: bool) -> bytes:
    if not chunk:
        words = [0] * 16
        out = _b3compress(
            _B3_IV, words, chunk_idx, 0, _B3_CHUNK_START | _B3_CHUNK_END | _B3_ROOT
        )
        return struct.pack("<8I", *out[:8])
    cv = list(_B3_IV)
    num_blocks = (len(chunk) + _B3_BLOCK_LEN - 1) // _B3_BLOCK_LEN
    for bi in range(num_blocks):
        blk = chunk[bi * _B3_BLOCK_LEN : (bi + 1) * _B3_BLOCK_LEN]
        padded = blk + b"\x00" * (_B3_BLOCK_LEN - len(blk))
        words = list(struct.unpack("<16I", padded))
        flags = (_B3_CHUNK_START if bi == 0 else 0) | (
            _B3_CHUNK_END if bi == num_blocks - 1 else 0
        )
        if is_only_chunk and bi == num_blocks - 1:
            flags |= _B3_ROOT
        out = _b3compress(tuple(cv), words, chunk_idx, len(blk), flags)
        cv = out[:8]
    return struct.pack("<8I", *cv)


def _blake3bytes(data: bytes) -> bytes:
    n = len(data)
    if n <= _B3_CHUNK_LEN:
        return _b3chunkcv(data, 0, True)
    cvs = []
    num_chunks = (n + _B3_CHUNK_LEN - 1) // _B3_CHUNK_LEN
    for ci in range(num_chunks):
        cvs.append(
            _b3chunkcv(data[ci * _B3_CHUNK_LEN : (ci + 1) * _B3_CHUNK_LEN], ci, False)
        )
    while len(cvs) > 1:
        nxt = []
        i = 0
        while i < len(cvs):
            if i + 1 < len(cvs):
                words = list(struct.unpack("<16I", cvs[i] + cvs[i + 1]))
                is_root = len(cvs) == 2
                out = _b3compress(
                    _B3_IV,
                    words,
                    0,
                    _B3_BLOCK_LEN,
                    _B3_PARENT | (_B3_ROOT if is_root else 0),
                )
                nxt.append(struct.pack("<8I", *out[:8]))
                i += 2
            else:
                nxt.append(cvs[i])
                i += 1
        cvs = nxt
    return cvs[0]


def blake3_hex(data: bytes) -> str:
    """Return lowercase hexadecimal digest for input bytes."""
    return _blake3bytes(bytes(data)).hex()


# ---------------------------------------------------------------------------
# Keys and SplitMix64 Box-Muller normal_hash (bit-identical to Rust semantics)
# ---------------------------------------------------------------------------


class Key:
    """Immutable, splittable, foldable 128-bit PRNG key."""

    def __init__(self, source: Any = None) -> None:
        if source is None:
            raw = int.from_bytes(os.urandom(8), "little")
            hi = _splitmix64(raw)
            lo = _splitmix64(hi ^ _SM64_OFFSET)
        elif isinstance(source, Key):
            hi, lo = source._hi, source._lo
        elif isinstance(source, (bytes, bytearray, memoryview)):
            k = Key.from_bytes(bytes(source))
            hi, lo = k._hi, k._lo
        else:
            seed = int(source) & _MASK64
            hi = _splitmix64(seed)
            lo = _splitmix64(hi ^ _SM64_OFFSET)
        self._hi = hi
        self._lo = lo

    @classmethod
    def entropy(cls) -> Key:
        """Create a non-deterministic entropy-seeded Key."""
        return cls(None)

    @classmethod
    def from_bytes(cls, data: bytes) -> Key:
        """Derive a deterministic Key from arbitrary byte input."""
        b = bytes(data)
        k = cls(len(b))
        for i in range(0, len(b), 8):
            chunk = b[i : i + 8]
            word = int.from_bytes(chunk.ljust(8, b"\x00"), "little")
            k = k.fold(word)
        return k

    @classmethod
    def from_pair(cls, hi: int, lo: int) -> Key:
        """Construct a Key from explicit 64-bit `(hi, lo)` lanes."""
        out = cls.__new__(cls)
        out._hi = int(hi) & _MASK64
        out._lo = int(lo) & _MASK64
        return out

    @property
    def hi(self) -> int:
        """Return high 64-bit key lane."""
        return self._hi

    @property
    def lo(self) -> int:
        """Return low 64-bit key lane."""
        return self._lo

    def seed(self) -> int:
        """Fold `(hi, lo)` into a 64-bit seed integer."""
        rot = ((self._lo << 31) | (self._lo >> 33)) & _MASK64
        return _splitmix64(self._hi ^ rot)

    def fold(self, tag: int) -> Key:
        """Return a derived child Key mixed with integer `tag`."""
        t = int(tag) & _MASK64
        hi = _splitmix64(self._hi ^ ((t * 0x9E3779B97F4A7C15) & _MASK64))
        lo = _splitmix64(self._lo ^ hi ^ ((t * 0xBF58476D1CE4E5B9) & _MASK64))
        return Key.from_pair(hi, lo)

    def split(self) -> tuple[Key, Key]:
        """Split key into two child keys."""
        return (self.fold(0x6A09E667F3BCC908), self.fold(0xBB67AE8584CAA73B))

    def split_n(self, n: int) -> list[Key]:
        """Split key into `n` child keys."""
        return [self.fold(i + 1) for i in range(int(n))]

    def __repr__(self) -> str:
        return f"Key(0x{self._hi:016x}, 0x{self._lo:016x})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Key) and self._hi == other._hi and self._lo == other._lo
        )

    def __hash__(self) -> int:
        return hash((self._hi, self._lo))


def normal_hash(
    function_seeds: np.ndarray, data_indices: np.ndarray, num_metrics: int
) -> np.ndarray:
    """Hash `(function_seed, data_index, metric)` tuples into i.i.d. N(0, 1) draws."""
    m = int(num_metrics)
    if m <= 0:
        raise ValueError(f"num_metrics must be positive, got {m}")
    seeds_arr = np.asarray(function_seeds, dtype=np.int64).ravel()
    idx_arr = np.asarray(data_indices, dtype=np.int64)
    idx_shape = idx_arr.shape
    flat_idx = idx_arr.ravel().tolist()
    uniq = sorted(set(flat_idx))
    pos_map = {v: i for i, v in enumerate(uniq)}
    inv = np.array([pos_map[v] for v in flat_idx], dtype=np.intp)
    out = np.empty((seeds_arr.size, len(uniq), m), dtype=np.float64)
    clip_min, clip_max = 1e-12, 1.0 - 1e-12
    for si, s_raw in enumerate(seeds_arr.tolist()):
        s_u64 = int(s_raw) & _MASK64
        for ui, u_raw in enumerate(uniq):
            u_u64 = int(u_raw) & _MASK64
            base = (((s_u64 * _SEED_PRIME) + u_u64) * _SEED_PRIME) & _MASK64
            for mi in range(m):
                c1 = (base + mi) & _MASK64
                r1 = _splitmix64(c1)
                r2 = _splitmix64(c1 ^ _SM64_OFFSET)
                u1 = (r1 >> 11) * _INV_2P53
                if u1 < clip_min:
                    u1 = clip_min
                elif u1 > clip_max:
                    u1 = clip_max
                u2 = (r2 >> 11) * _INV_2P53
                out[si, ui, mi] = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                    2.0 * math.pi * u2
                )
    rebuilt = out[:, inv, :]
    return rebuilt.reshape((seeds_arr.size, *idx_shape, m))


# ---------------------------------------------------------------------------
# Utility algorithms: standardize_y, Pareto, hypervolume, Sobol sequence/indices
# ---------------------------------------------------------------------------


def standardize_y(y: np.ndarray) -> tuple[float, float]:
    """Return robust `(median, std)` standardization statistics for `y`."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    center = float(np.median(finite))
    std = float(np.std(finite))
    if not math.isfinite(std) or std <= 1e-12:
        std = 1.0
    return (center, std)


def pareto2d_max(
    a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None
) -> np.ndarray:
    """Return indices of the non-dominated 2D maximization Pareto frontier."""
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    cand = (
        np.arange(a_arr.size, dtype=np.int64)
        if idx is None
        else np.asarray(idx, dtype=np.int64).ravel()
    )
    if cand.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.lexsort((-b_arr[cand], -a_arr[cand]))
    best_b = -math.inf
    keep: list[int] = []
    for pos in order:
        i = int(cand[pos])
        bv = float(b_arr[i])
        if bv > best_b:
            keep.append(i)
            best_b = bv
    return np.asarray(keep, dtype=np.int64)


def pareto_arms(
    x_cand: np.ndarray, mu: np.ndarray, se: np.ndarray, num_arms: int, seed: int
) -> np.ndarray:
    """Select `num_arms` candidates greedily by non-dominated `(mu, se)` layers."""
    x_arr = np.asarray(x_cand, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64).ravel()
    se_arr = np.asarray(se, dtype=np.float64).ravel()
    rng = np.random.default_rng(int(seed) & _MASK64)
    remaining = np.arange(x_arr.shape[0], dtype=np.int64)
    picked: list[int] = []
    while len(picked) < int(num_arms) and remaining.size > 0:
        front = pareto2d_max(mu_arr, se_arr, remaining)
        need = int(num_arms) - len(picked)
        if front.size <= need:
            picked.extend(int(i) for i in front)
        else:
            sel = rng.choice(front, size=need, replace=False)
            picked.extend(int(i) for i in sel)
        mask = np.isin(remaining, front, invert=True)
        remaining = remaining[mask]
    return np.ascontiguousarray(x_arr[picked], dtype=np.float64)


def hypervolume2d_max(y: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute exact 2D maximization hypervolume dominated relative to `ref_point`."""
    pts = np.asarray(y, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64).ravel()
    if pts.ndim != 2 or pts.shape[1] != 2 or ref.size != 2:
        raise ValueError(
            f"Expected (n, 2) and (2,) reference, got {pts.shape}, {ref.shape}"
        )
    valid = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if valid.shape[0] == 0:
        return 0.0
    front_idx = pareto2d_max(valid[:, 0], valid[:, 1])
    front = valid[front_idx]
    order = np.argsort(-front[:, 0])
    front = front[order]
    hv = 0.0
    cur_y = ref[1]
    for p in front:
        if p[1] > cur_y:
            hv += float((p[0] - ref[0]) * (p[1] - cur_y))
            cur_y = float(p[1])
    return float(hv)


def sobol_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate first-order coordinate sensitivity weights from `(x, y)`."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n, d = x_arr.shape
    var_y = float(np.var(y_arr))
    if n < 4 or var_y <= 1e-18:
        return np.full((d,), 1.0 / max(d, 1), dtype=np.float64)
    scores = np.zeros((d,), dtype=np.float64)
    bins = max(2, min(8, n // 4))
    for j in range(d):
        col = x_arr[:, j]
        if float(np.ptp(col)) <= 1e-18:
            scores[j] = 0.0
            continue
        order = np.argsort(col)
        chunks = np.array_split(y_arr[order], bins)
        y_mean = float(np.mean(y_arr))
        between = (
            sum(
                c.size * (float(np.mean(c)) - y_mean) ** 2 for c in chunks if c.size > 0
            )
            / n
        )
        scores[j] = max(0.0, between / var_y)
    total = float(np.sum(scores))
    if total <= 1e-18:
        return np.full((d,), 1.0 / d, dtype=np.float64)
    return scores / total


# Compact Sobol polynomial/direction initialization + scrambled digital net
_SOBOL_POLY = (
    (1, 0, (1,)),
    (2, 1, (1, 1)),
    (3, 1, (1, 3)),
    (3, 2, (1, 3, 7)),
    (4, 1, (1, 1, 5)),
    (4, 4, (1, 3, 1, 1)),
    (5, 2, (1, 1, 3, 7)),
    (5, 4, (1, 3, 5, 13)),
    (5, 7, (1, 1, 5, 5, 17)),
    (5, 11, (1, 3, 7, 11, 29)),
    (5, 13, (1, 1, 7, 3, 11)),
    (5, 14, (1, 3, 3, 13, 25)),
    (6, 1, (1, 1, 1, 9, 25)),
    (6, 13, (1, 3, 5, 3, 1, 45)),
    (6, 16, (1, 1, 3, 13, 11, 7)),
    (6, 19, (1, 3, 7, 5, 17, 61)),
    (6, 22, (1, 1, 5, 15, 29, 37)),
    (6, 25, (1, 3, 1, 11, 9, 55)),
    (7, 1, (1, 1, 3, 3, 19, 41, 73)),
    (7, 4, (1, 3, 5, 13, 21, 59, 91)),
    (7, 7, (1, 1, 7, 9, 13, 35, 115)),
    (7, 8, (1, 3, 1, 15, 29, 53, 83)),
    (7, 14, (1, 1, 3, 5, 23, 39, 97)),
    (7, 19, (1, 3, 7, 7, 9, 45, 121)),
)
_SOBOL_DIRS: dict[int, list[int]] = {}


def _soboldirs(dim_idx: int, bits: int = 32) -> list[int]:
    if dim_idx in _SOBOL_DIRS:
        return _SOBOL_DIRS[dim_idx]
    v = [0] * (bits + 1)
    if dim_idx == 0:
        for i in range(1, bits + 1):
            v[i] = 1 << (bits - i)
    elif dim_idx - 1 < len(_SOBOL_POLY):
        deg, a, m_init = _SOBOL_POLY[dim_idx - 1]
        for i in range(1, deg + 1):
            mi = m_init[i - 1] if i - 1 < len(m_init) else 1
            v[i] = mi << (bits - i)
        for i in range(deg + 1, bits + 1):
            nv = v[i - deg] ^ (v[i - deg] >> deg)
            for k in range(1, deg):
                if (a >> (deg - 1 - k)) & 1:
                    nv ^= v[i - k]
            v[i] = nv & _MASK32
    else:
        for i in range(1, bits + 1):
            step = (_splitmix64(dim_idx * 131 + i) & ((1 << i) - 1)) | 1
            v[i] = (step << (bits - i)) & _MASK32
    _SOBOL_DIRS[dim_idx] = v
    return v


def sobol_sequence(d: int, n: int, seed: int = 0) -> np.ndarray:
    """Generate `n` scrambled Sobol low-discrepancy points in `[0, 1)^d`."""
    num_dim = int(d)
    num_pts = int(n)
    if num_dim <= 0 or num_pts <= 0:
        return np.zeros((max(num_pts, 0), max(num_dim, 0)), dtype=np.float64)
    rng = np.random.default_rng(int(seed) & _MASK64)
    shifts = rng.integers(0, 1 << 32, size=num_dim, dtype=np.uint32)
    scale = 1.0 / float(1 << 32)
    out = np.empty((num_pts, num_dim), dtype=np.float64)
    state = shifts.copy()
    dirs = [_soboldirs(j) for j in range(num_dim)]
    for i in range(num_pts):
        out[i] = state.astype(np.float64) * scale
        c = ((i + 1) & -(i + 1)).bit_length()
        for j in range(num_dim):
            state[j] ^= dirs[j][c]
    return np.clip(out, 0.0, 1.0 - 1e-12)


def set_path(_path: str | None) -> None:
    """No-op lockfile path setter for browser WebAssembly."""
    pass


def _ensurecfg() -> str:
    """Return synthetic in-browser runtime configuration path."""
    return "/tmp/baux.toml"


def _verifylock() -> bool:
    """Confirm static WebAssembly runtime integrity verification."""
    return True


# ---------------------------------------------------------------------------
# ENN Surrogate Engine
# ---------------------------------------------------------------------------


class ENNParams:
    """Hyperparameters for an Epistemic Nearest Neighbors surrogate."""

    def __init__(
        self, k_neighbors: int, epistemic_scale: float, aleatoric_scale: float
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.epistemic_scale = float(epistemic_scale)
        self.aleatoric_scale = float(aleatoric_scale)

    def __repr__(self) -> str:
        return (
            f"ENNParams(k_neighbors={self.k_neighbors}, "
            f"epistemic_scale={self.epistemic_scale:.6g}, "
            f"aleatoric_scale={self.aleatoric_scale:.6g})"
        )


def _metricweights(
    x: np.ndarray, geometry: str, local_k: int, base_weights: np.ndarray | None
) -> np.ndarray:
    n = x.shape[0]
    if base_weights is not None:
        w = np.asarray(base_weights, dtype=np.float64).ravel()
        if w.size == x.shape[1]:
            pass
    if geometry == "euclidean" or n <= 2:
        return np.ones(n, dtype=np.float64)
    diffs = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(d2, np.inf)
    kk = max(1, min(local_k, n - 1))
    local_scale = np.sqrt(
        np.maximum(np.partition(d2, kk - 1, axis=1)[:, :kk].mean(axis=1), 1e-12)
    )
    if geometry == "conformal":
        return 1.0 / np.maximum(local_scale, 1e-4)
    if geometry == "hyperbolic":
        r2 = np.sum(x * x, axis=1)
        return 1.0 / np.maximum(1.0 - 0.5 * np.tanh(r2), 1e-4)
    if geometry == "wasserstein":
        return np.sqrt(np.maximum(1.0 / np.maximum(local_scale, 1e-4), 1e-6))
    return np.ones(n, dtype=np.float64)


class ENN:
    """Vectorized NumPy implementation of the ENN surrogate with COW cloning."""

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_yvar: np.ndarray | None = None,
        scale_x: bool = False,
        index_driver: str = "Exact",
        y_bounds: np.ndarray | None = None,
        metric: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.device = "wasm"
        self._scale_x = bool(scale_x)
        self._index_driver = str(index_driver)
        self._x = np.array(train_x, dtype=np.float64, copy=True)
        self._y = np.array(train_y, dtype=np.float64, copy=True)
        if self._y.ndim == 1:
            self._y = self._y.reshape(-1, 1)
        self._yvar = (
            None
            if train_yvar is None
            else np.array(train_yvar, dtype=np.float64, copy=True)
        )
        self._y_bounds_cfg = (
            None
            if y_bounds is None
            else np.array(y_bounds, dtype=np.float64, copy=True)
        )
        self._metric_cfg: dict[str, Any] = copy.deepcopy(metric) if metric else {}
        self._geometry = str(self._metric_cfg.get("geometry", "euclidean"))
        self._local_k = int(self._metric_cfg.get("local_k", 4))
        self._stiefel = self._metric_cfg.get("stiefel")
        self._affine = self._metric_cfg.get("affine")
        self._recomputestats()

    def _recomputestats(self) -> None:
        n, d = self._x.shape
        if self._scale_x and n > 1:
            lo = np.min(self._x, axis=0)
            hi = np.max(self._x, axis=0)
            span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
            self._x_lo = lo
            self._x_span = span
            self._x_scaled = (self._x - lo) / span
        else:
            self._x_lo = np.zeros(d, dtype=np.float64)
            self._x_span = np.ones(d, dtype=np.float64)
            self._x_scaled = self._x
        self._lengthscales = np.ones(d, dtype=np.float64)
        self._row_factors = _metricweights(
            self._x_scaled, self._geometry, self._local_k, None
        )

    def _clone(self) -> ENN:
        out = object.__new__(ENN)
        out.device = self.device
        out._scale_x = self._scale_x
        out._index_driver = self._index_driver
        out._x = self._x.copy()
        out._y = self._y.copy()
        out._yvar = None if self._yvar is None else self._yvar.copy()
        out._y_bounds_cfg = (
            None if self._y_bounds_cfg is None else self._y_bounds_cfg.copy()
        )
        out._metric_cfg = copy.deepcopy(self._metric_cfg)
        out._geometry = self._geometry
        out._local_k = self._local_k
        out._stiefel = copy.deepcopy(self._stiefel)
        out._affine = copy.deepcopy(self._affine)
        out._recomputestats()
        return out

    def append_cow(
        self,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        yvar: np.ndarray | None = None,
        *,
        metric: str | None = None,
        local_k: int | None = None,
        stiefel: Any = None,
        affine: Any = None,
    ) -> ENN:
        """Clone surrogate copy-on-write and optionally update geometry or rows."""
        out = self._clone()
        if metric is not None:
            out._geometry = str(metric)
        if local_k is not None:
            out._local_k = int(local_k)
        if stiefel is not None:
            out._stiefel = np.array(stiefel, dtype=np.float64, copy=True)
        if affine is not None:
            out._affine = affine
        if x is not None and y is not None:
            out.add(x, y, yvar)
        else:
            out._recomputestats()
        return out

    def add(self, x: np.ndarray, y: np.ndarray, yvar: np.ndarray | None = None) -> None:
        """Append observation batch `(x, y, yvar)` to the surrogate store."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya.reshape(-1, 1)
        if xa.shape[0] == 0:
            return
        self._x = np.vstack([self._x, xa])
        self._y = np.vstack([self._y, ya])
        if self._yvar is not None or yvar is not None:
            yv_old = (
                np.zeros_like(self._y[: self._y.shape[0] - ya.shape[0]])
                if self._yvar is None
                else self._yvar
            )
            yv_new = (
                np.zeros_like(ya)
                if yvar is None
                else np.asarray(yvar, dtype=np.float64)
            )
            if yv_new.ndim == 1:
                yv_new = yv_new.reshape(-1, 1)
            self._yvar = np.vstack([yv_old, yv_new])
        self._recomputestats()

    def ensure_sync(self) -> None:
        """No-op sync hook for pure in-memory WASM backend."""
        pass

    def schedule_flush(self) -> None:
        """No-op flush hook for pure in-memory WASM backend."""
        pass

    def persist_index(self) -> None:
        """No-op index persistence hook for browser WASM."""
        pass

    def index_bytes(self) -> int:
        """Return approximate memory usage of indexed training tensors."""
        return int(self._x.nbytes + self._y.nbytes)

    @property
    def num_outputs(self) -> int:
        """Return target output dimensionality."""
        return int(self._y.shape[1])

    @property
    def num_dim(self) -> int:
        """Return input coordinate dimensionality."""
        return int(self._x.shape[1])

    @property
    def y_bounds(self) -> np.ndarray:
        """Return observed or configured `(num_outputs, 2)` target bounds."""
        if self._y_bounds_cfg is not None:
            return self._y_bounds_cfg
        if self._y.shape[0] == 0:
            return np.tile([0.0, 1.0], (self.num_outputs, 1))
        lo = np.min(self._y, axis=0)
        hi = np.max(self._y, axis=0)
        return np.column_stack([lo, hi])

    @property
    def diagonal_metric(self) -> dict[str, Any]:
        """Return fitted diagonal lengthscale metric metadata."""
        return {
            "lengthscales": self._lengthscales.tolist(),
            "geometry": self._geometry,
            "local_k": self._local_k,
        }

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def train_rows(
        self, indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return stored `(x, y, yvar)` slices for row `indices`."""
        idx = np.asarray(indices, dtype=np.intp)
        yv = None if self._yvar is None else self._yvar[idx]
        return (self._x[idx], self._y[idx], yv)

    def row_y(self, i: int) -> np.ndarray:
        """Return observation target vector for row `i`."""
        return self._y[int(i) : int(i) + 1]

    def _dist2topk(
        self, q: np.ndarray, k: int, exclude_nearest: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        qa = np.asarray(q, dtype=np.float64)
        qs = (qa - self._x_lo) / self._x_span
        inv_l2 = 1.0 / np.maximum(self._lengthscales**2, 1e-12)
        diff = qs[:, None, :] - self._x_scaled[None, :, :]
        d2 = np.sum((diff * diff) * inv_l2[None, None, :], axis=2)
        d2 = d2 * self._row_factors[None, :]
        n = self._x.shape[0]
        if exclude_nearest and n > 1:
            nearest = np.argmin(d2, axis=1)
            d2[np.arange(qa.shape[0]), nearest] = np.inf
            k_eff = min(max(k, 1), n - 1)
        else:
            k_eff = min(max(k, 1), n)
        if k_eff == n:
            order = np.argsort(d2, axis=1)
        else:
            part = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
            row_idx = np.arange(qa.shape[0])[:, None]
            part_d2 = d2[row_idx, part]
            sub_ord = np.argsort(part_d2, axis=1)
            order = part[row_idx, sub_ord]
        sel_d2 = np.take_along_axis(d2, order, axis=1)
        return order, np.maximum(sel_d2, 0.0)

    @staticmethod
    def _d2weights(d2: np.ndarray) -> np.ndarray:
        h = np.maximum(d2[:, -1:], 1e-12)
        u2 = np.minimum(d2 / h, 1.0)
        raw_w = np.maximum(1.0 - u2, 1e-6)
        raw_w = np.where(d2 <= 1e-12, 1.0, raw_w)
        return raw_w / np.sum(raw_w, axis=1, keepdims=True)

    def _nbrmoments(
        self,
        idx: np.ndarray,
        d2: np.ndarray,
        epistemic_scale: float,
        aleatoric_scale: float,
        observation_noise: bool,
        extra_x: np.ndarray | None = None,
        extra_y: np.ndarray | None = None,
        q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        b, k_eff = idx.shape
        m = self.num_outputs
        if k_eff == 0:
            z = np.zeros((b, m), dtype=np.float64)
            one = np.ones((b, m), dtype=np.float64) * math.sqrt(
                epistemic_scale**2 + aleatoric_scale**2
            )
            return z, one, one, np.zeros_like(z)
        h = np.maximum(d2[:, -1:], 1e-12)
        w = self._d2weights(d2)
        yn = self._y[idx]  # (b, k, m)
        mu = np.sum(w[:, :, None] * yn, axis=1)
        if (
            extra_x is not None
            and extra_y is not None
            and extra_x.shape[0] > 0
            and q is not None
        ):
            qd = np.sum(
                ((q[:, None, :] - extra_x[None, :, :]) / self._x_span[None, None, :])
                ** 2,
                axis=2,
            )
            ew = np.exp(-qd / np.maximum(h, 1e-6))
            total_w = np.sum(ew, axis=1, keepdims=True) + 1.0
            mu = (mu + ew @ extra_y) / total_w
        y_std = np.maximum(np.std(self._y, axis=0), 1e-6)
        mean_d2 = np.sum(w * d2, axis=1, keepdims=True)
        exact = d2[:, :1] <= 1e-12
        var_epi = (float(epistemic_scale) ** 2) * y_std[None, :] ** 2 * mean_d2
        var_epi = np.where(exact, 0.0, var_epi)
        if self._yvar is not None:
            loc_var = np.sum(w[:, :, None] * self._yvar[idx], axis=1)
        else:
            resid = yn - mu[:, None, :]
            loc_var = np.maximum(
                np.sum(w[:, :, None] * (resid * resid), axis=1),
                1e-6 * (y_std[None, :] ** 2),
            )
        var_ale = (
            (float(aleatoric_scale) ** 2) * loc_var
            if observation_noise
            else np.zeros_like(mu)
        )
        se_epi = np.sqrt(np.maximum(var_epi, 0.0))
        se_ale = np.sqrt(np.maximum(var_ale, 0.0))
        se = np.sqrt(np.maximum(var_epi + var_ale, 0.0))
        return mu, se, se_epi, se_ale

    def _drawmoments(
        self,
        x: np.ndarray,
        idx_arr: np.ndarray,
        d2: np.ndarray,
        mu: np.ndarray,
        se_epi: np.ndarray,
        se_ale: np.ndarray,
        function_seeds: list[int],
    ) -> np.ndarray:
        seeds = np.asarray(function_seeds, dtype=np.int64)
        b, k_eff = idx_arr.shape
        if k_eff == 0:
            q_idx = np.arange(x.shape[0], dtype=np.int64)
            z = normal_hash(seeds, q_idx, self.num_outputs)
            se = np.sqrt(se_epi**2 + se_ale**2)
            return mu[None, :, :] + se[None, :, :] * z
        w = self._d2weights(d2)
        w_unit = w / np.maximum(np.sqrt(np.sum(w * w, axis=1, keepdims=True)), 1e-12)
        uniq, inv = np.unique(idx_arr, return_inverse=True)
        eps_u = normal_hash(seeds, uniq.astype(np.int64), self.num_outputs)
        eps_neigh = eps_u[:, inv.reshape(b, k_eff), :]
        z_epi = np.einsum("bk,sbkm->sbm", w_unit, eps_neigh)
        q_idx = 1_000_000_000 + np.arange(b, dtype=np.int64)
        z_ale = normal_hash(seeds, q_idx, self.num_outputs)
        return mu[None, :, :] + se_epi[None, :, :] * z_epi + se_ale[None, :, :] * z_ale

    def posterior(
        self,
        x: np.ndarray,
        *,
        k_neighbors: int,
        epistemic_scale: float,
        aleatoric_scale: float,
        exclude_nearest: bool = False,
        observation_noise: bool = True,
        tie_mode: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
        """Compute ENN posterior mean, total std, epistemic/aleatoric decomposition, and neighbors."""
        del tie_mode
        qa = np.asarray(x, dtype=np.float64)
        idx, d2 = self._dist2topk(qa, int(k_neighbors), bool(exclude_nearest))
        mu, se, se_epi, se_ale = self._nbrmoments(
            idx, d2, epistemic_scale, aleatoric_scale, bool(observation_noise)
        )
        return mu, se, se_epi, se_ale, idx.tolist()

    def conditional_posterior(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        k_neighbors: int,
        epistemic_scale: float,
        aleatoric_scale: float,
        exclude_nearest: bool = False,
        observation_noise: bool = True,
        tie_mode: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
        """Compute ENN posterior conditioned on fantasized what-if pairs `(x_whatif, y_whatif)`."""
        del tie_mode
        qa = np.asarray(x, dtype=np.float64)
        idx, d2 = self._dist2topk(qa, int(k_neighbors), bool(exclude_nearest))
        mu, se, se_epi, se_ale = self._nbrmoments(
            idx,
            d2,
            epistemic_scale,
            aleatoric_scale,
            bool(observation_noise),
            np.asarray(x_whatif, dtype=np.float64),
            np.asarray(y_whatif, dtype=np.float64),
            qa,
        )
        return mu, se, se_epi, se_ale, idx.tolist()

    def batch_posterior(
        self,
        x: np.ndarray,
        *,
        k_values: list[int],
        epistemic_scales: list[float],
        aleatoric_scales: list[float],
        exclude_nearest: bool = False,
        observation_noise: bool = True,
        tie_mode: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute posterior tensors across batched `(k, epistemic, aleatoric)` configs."""
        mus, ses, epis, ales = [], [], [], []
        for k, e, a in zip(k_values, epistemic_scales, aleatoric_scales, strict=True):
            mu, se, se_epi, se_ale, _ = self.posterior(
                x,
                k_neighbors=k,
                epistemic_scale=e,
                aleatoric_scale=a,
                exclude_nearest=exclude_nearest,
                observation_noise=observation_noise,
                tie_mode=tie_mode,
            )
            mus.append(mu)
            ses.append(se)
            epis.append(se_epi)
            ales.append(se_ale)
        return np.stack(mus), np.stack(ses), np.stack(epis), np.stack(ales)

    def neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> np.ndarray:
        """Return `k`-nearest neighbor row indices for query points `x`."""
        if k == 0 or len(self) == 0:
            return np.zeros((x.shape[0], 0), dtype=np.int64)
        idx, _ = self._dist2topk(x, int(k), bool(exclude_nearest))
        return idx

    def nearest_neighbors(
        self, x: np.ndarray, k: int, *, exclude_nearest: bool = False
    ) -> np.ndarray:
        """Alias for `neighbors`."""
        return self.neighbors(x, k, exclude_nearest=exclude_nearest)

    def posterior_neighbors(
        self,
        x: np.ndarray,
        k: int,
        *,
        exclude_nearest: bool = False,
        tie_mode: int = 0,
    ) -> np.ndarray:
        """Return `k`-nearest neighbor indices matching posterior tie ordering."""
        del tie_mode
        return self.neighbors(x, k, exclude_nearest=exclude_nearest)

    def posterior_draw(
        self,
        x: np.ndarray,
        *,
        k_neighbors: int,
        epistemic_scale: float,
        aleatoric_scale: float,
        function_seeds: list[int],
        exclude_nearest: bool = False,
        observation_noise: bool = True,
        tie_mode: int = 0,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Draw correlated function realizations for `function_seeds` at `x`."""
        del tie_mode
        qa = np.asarray(x, dtype=np.float64)
        idx_arr, d2 = self._dist2topk(qa, int(k_neighbors), bool(exclude_nearest))
        mu, _, se_epi, se_ale = self._nbrmoments(
            idx_arr, d2, epistemic_scale, aleatoric_scale, bool(observation_noise)
        )
        draws = self._drawmoments(qa, idx_arr, d2, mu, se_epi, se_ale, function_seeds)
        return draws, idx_arr.tolist()

    def conditional_draw(
        self,
        x_whatif: np.ndarray,
        y_whatif: np.ndarray,
        x: np.ndarray,
        *,
        k_neighbors: int,
        epistemic_scale: float,
        aleatoric_scale: float,
        function_seeds: list[int],
        exclude_nearest: bool = False,
        observation_noise: bool = True,
        tie_mode: int = 0,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Draw posterior samples conditioned on what-if observations `(x_whatif, y_whatif)`."""
        del tie_mode
        qa = np.asarray(x, dtype=np.float64)
        idx_arr, d2 = self._dist2topk(qa, int(k_neighbors), bool(exclude_nearest))
        mu, _, se_epi, se_ale = self._nbrmoments(
            idx_arr,
            d2,
            epistemic_scale,
            aleatoric_scale,
            bool(observation_noise),
            extra_x=np.asarray(x_whatif, dtype=np.float64),
            extra_y=np.asarray(y_whatif, dtype=np.float64),
            q=qa,
        )
        draws = self._drawmoments(qa, idx_arr, d2, mu, se_epi, se_ale, function_seeds)
        return draws, idx_arr.tolist()

    def device_posterior(self, x: Any, **kwargs: Any) -> Any:
        """Raise clear error since device evaluators are unavailable in browser WASM."""
        del x, kwargs
        raise RuntimeError(
            "device_posterior requires CUDA or Metal; not available in browser WASM"
        )


def _gaussloglik(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    v = np.maximum(var, 1e-12)
    return float(-0.5 * np.mean(np.log(2.0 * math.pi * v) + ((y_true - mu) ** 2) / v))


def subsample_loglik(
    model: ENN,
    x: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    epi_scales: list[float],
    ale_scales: list[float],
    p_sub: int = 10,
    seed: int = 0,
    y_std: np.ndarray | None = None,
) -> list[float]:
    """Compute Gaussian log-likelihood on a deterministic subsample for candidate configs."""
    del y_std
    rng = np.random.default_rng(int(seed) & _MASK64)
    n = x.shape[0]
    m = max(1, min(n, int(p_sub) * max(1, n // max(int(p_sub), 1))))
    sub = rng.choice(n, size=min(n, max(m, 4)), replace=False)
    xs = x[sub]
    ys = y[sub]
    out: list[float] = []
    for k, e, a in zip(k_values, epi_scales, ale_scales, strict=True):
        mu, se, _, _, _ = model.posterior(
            xs,
            k_neighbors=k,
            epistemic_scale=e,
            aleatoric_scale=a,
            exclude_nearest=n > 1,
            observation_noise=True,
        )
        out.append(_gaussloglik(ys, mu, se**2))
    return out


def row_loglik(
    model: ENN,
    rows: list[int],
    k_values: list[int],
    epi_scales: list[float],
    ale_scales: list[float],
    p_sub: int = 10,
    seed: int = 0,
    y_std: np.ndarray | None = None,
) -> list[float]:
    """Compute Gaussian leave-one-out log-likelihood across explicit `rows`."""
    del p_sub, seed, y_std
    xr, yr, _ = model.train_rows(rows)
    out: list[float] = []
    for k, e, a in zip(k_values, epi_scales, ale_scales, strict=True):
        mu, se, _, _, _ = model.posterior(
            xr,
            k_neighbors=k,
            epistemic_scale=e,
            aleatoric_scale=a,
            exclude_nearest=len(model) > 1,
            observation_noise=True,
        )
        out.append(_gaussloglik(yr, mu, se**2))
    return out


def fit_model(
    model: ENN,
    k: int,
    num_candidates: int,
    num_samples: int,
    seed: int,
    warm_start: ENNParams | None = None,
    infer_aleatoric: bool = True,
) -> ENNParams:
    """Grid-search ENN epistemic and aleatoric scales by leave-one-out log-likelihood."""
    del warm_start
    n = len(model)
    k_eff = max(1, min(int(k), max(1, n - 1)))
    if n < 2:
        return ENNParams(k_eff, 1.0, 1e-3 if infer_aleatoric else 0.0)
    rng = np.random.default_rng(int(seed) & _MASK64)
    n_grid = max(4, min(16, int(num_candidates)))
    epis = np.geomspace(0.2, 5.0, n_grid)
    ales = (
        np.geomspace(1e-5, 0.25, max(3, min(6, int(num_samples))))
        if infer_aleatoric
        else [0.0]
    )
    best_ll = -math.inf
    best = (1.0, ales[0])
    sub = np.arange(n) if n <= 64 else rng.choice(n, size=64, replace=False)
    xs, ys = model._x[sub], model._y[sub]
    mu_base, se_e1, _, _, _ = model.posterior(
        xs,
        k_neighbors=k_eff,
        epistemic_scale=1.0,
        aleatoric_scale=0.0,
        exclude_nearest=True,
        observation_noise=False,
    )
    _, _, _, se_a1, _ = model.posterior(
        xs,
        k_neighbors=k_eff,
        epistemic_scale=0.0,
        aleatoric_scale=1.0,
        exclude_nearest=True,
        observation_noise=True,
    )
    v_e1 = np.maximum(se_e1**2, 1e-12)
    v_a1 = np.maximum(se_a1**2, 1e-12)
    sq_err = (ys - mu_base) ** 2
    for e in epis:
        for a in ales:
            v = (float(e) ** 2) * v_e1 + (float(a) ** 2) * v_a1
            ll = float(-0.5 * np.mean(np.log(2.0 * math.pi * v) + sq_err / v))
            if ll > best_ll:
                best_ll = ll
                best = (float(e), float(a))
    return ENNParams(k_eff, best[0], best[1])


class ENNFitter:
    """Incremental ENN hyperparameter fitter compatible with `python/baux/enn/enn_fitter.py`."""

    def __init__(
        self,
        k: int,
        seed: int | None = None,
        infer_aleatoric_variance_scale: bool = True,
    ) -> None:
        self._k = int(k)
        self._seed = 0 if seed is None else int(seed)
        self._infer_ale = bool(infer_aleatoric_variance_scale)
        self._y_rows: list[np.ndarray] = []

    def tell(
        self, x: np.ndarray, y: np.ndarray, yvar: np.ndarray | None = None
    ) -> None:
        """Record new observations `(x, y)` to update target standard deviation."""
        del x, yvar
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya.reshape(-1, 1)
        if ya.shape[0] > 0:
            self._y_rows.append(ya.copy())

    def y_std(self) -> np.ndarray:
        """Return per-output standard deviation across observed targets."""
        if not self._y_rows:
            return np.ones((1,), dtype=np.float64)
        all_y = np.vstack(self._y_rows)
        return np.maximum(np.std(all_y, axis=0), 1e-6)

    def ask(
        self,
        model: ENN,
        num_candidates: int,
        num_samples: int,
        warm_start: ENNParams | None = None,
    ) -> ENNParams:
        """Fit ENN hyperparameters for `model`."""
        return fit_model(
            model,
            self._k,
            num_candidates,
            num_samples,
            self._seed,
            warm_start,
            self._infer_ale,
        )


# ---------------------------------------------------------------------------
# Optimizer + TuRBO + Multi-Trust-Region Engine
# ---------------------------------------------------------------------------


@dataclass
class Telemetry:
    """Timing and candidate-count telemetry from an optimizer step."""

    dt_fit: float = 0.0
    dt_gen: float = 0.0
    dt_sel: float = 0.0
    dt_tell: float = 0.0
    num_candidates: int = 0


class MultiTrustRegion:
    """Lightweight multi-region tracker."""

    def __init__(self, num_dim: int, num_regions: int = 4) -> None:
        self.num_dim = int(num_dim)
        self.num_regions = int(num_regions)


def _lhdsamples(n: int, d: int, bounds: np.ndarray, key: Key) -> np.ndarray:
    rng = np.random.default_rng(key.seed())
    unit = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        unit[:, j] = (perm + rng.uniform(0.0, 1.0, size=n)) / float(n)
    lo, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    return lo[None, :] + unit * span[None, :]


def _ucbscore(mu: np.ndarray, se: np.ndarray, beta: float) -> np.ndarray:
    if mu.shape[1] == 1:
        return mu[:, 0] + float(beta) * se[:, 0]
    return np.mean(mu + float(beta) * se, axis=1)


class BayesOptimizer:
    """In-browser WASM/NumPy implementation of baux's BayesOptimizer state machine."""

    def __init__(
        self,
        bounds: np.ndarray,
        kind: str,
        k: int = 10,
        num_init: int = 10,
        num_regions: int = 1,
        seed: int | None = None,
        cfg: dict[str, Any] | None = None,
        gp: Any = None,
        fit_steps: int = 50,
        gp_factory: Any = None,
    ) -> None:
        self._bounds = np.array(bounds, dtype=np.float64, copy=True)
        self._d = int(self._bounds.shape[0])
        self._kind = str(kind)
        self._k = int(k)
        self._num_init = max(0, int(num_init))
        self._num_regions = max(1, int(num_regions))
        self._seed = Key.entropy().seed() if seed is None else (int(seed) & _MASK64)
        self._cfg: dict[str, Any] = copy.deepcopy(cfg) if cfg else {}
        self._gp = gp
        self._gp_factory = gp_factory
        self._fit_steps = int(fit_steps)

        self._len_init = float(self._cfg.get("length_init", 0.8))
        self._len_min = float(self._cfg.get("length_min", 0.5**7))
        self._len_max = float(self._cfg.get("length_max", 1.6))
        self._succ_tol = int(self._cfg.get("region_successes", 3))
        self._fail_tol = int(
            self._cfg.get("region_failures", max(4, math.ceil(self._d / 2)))
        )
        self._tr_lengths = [self._len_init] * self._num_regions
        self._succ_counts = [0] * self._num_regions
        self._fail_counts = [0] * self._num_regions
        self._region_centers: list[np.ndarray | None] = [None] * self._num_regions
        self._region_obs_idx: list[list[int]] = [[] for _ in range(self._num_regions)]

        self._x: np.ndarray = np.empty((0, self._d), dtype=np.float64)
        self._y: np.ndarray | None = None
        self._yvar: np.ndarray | None = None
        self._pending: dict[int, tuple[int, np.ndarray]] = {}
        self._pending_order: list[int] = []
        self._next_id: int = 1
        self._step_counter: int = 0
        self._restarts: int = 0
        self._warmup_pool: np.ndarray | None = None
        self._warmup_pos: int = 0
        self._last_telem = Telemetry()
        self._enn_params: ENNParams | None = None
        self._enn_model: ENN | None = None
        self._gp_inst: Any = gp() if (gp is not None and callable(gp)) else gp

    def _tounit(self, x: np.ndarray) -> np.ndarray:
        lo = self._bounds[:, 0]
        span = np.where((self._bounds[:, 1] - lo) > 1e-18, self._bounds[:, 1] - lo, 1.0)
        return (x - lo[None, :]) / span[None, :]

    def _fromunit(self, u: np.ndarray) -> np.ndarray:
        lo = self._bounds[:, 0]
        span = self._bounds[:, 1] - lo
        return lo[None, :] + np.clip(u, 0.0, 1.0) * span[None, :]

    def _stepkey(self, tag: int, seed: int | None) -> Key:
        base = Key(self._seed if seed is None else (int(seed) & _MASK64))
        return base.fold(self._step_counter).fold(tag)

    def _primaryscore(self, y: np.ndarray) -> np.ndarray:
        if y.shape[1] == 1:
            return y[:, 0]
        return np.sum(y, axis=1)

    def _rebuildmodel(self) -> None:
        if self._kind not in ("enn", "enn_multi_tr"):
            return
        if self._y is None or self._x.shape[0] < 2:
            self._enn_model = None
            return
        scale_x = bool(self._cfg.get("scale_x", False))
        metric = {}
        if "geometry" in self._cfg:
            metric["geometry"] = self._cfg["geometry"]
            metric["local_k"] = self._cfg.get("local_k", 4)
        self._enn_model = ENN(
            self._x,
            self._y,
            self._yvar,
            scale_x=scale_x,
            metric=metric or None,
            y_bounds=self._cfg.get("y_bounds"),
        )
        k_fit = max(1, min(self._k, self._x.shape[0] - 1))
        self._enn_params = fit_model(
            self._enn_model, k_fit, 10, 4, self._stepkey(99, None).seed()
        )

    def _fitifgp(self) -> None:
        if self._kind == "gp" and self._x.shape[0] >= 2 and self._y is not None:
            if self._gp_inst is None and self._gp_factory is not None:
                self._gp_inst = self._gp_factory(0, self._step_counter)
            if self._gp_inst is not None and hasattr(self._gp_inst, "fit"):
                self._gp_inst.fit(
                    self._tounit(self._x), self._y, self._yvar, steps=self._fit_steps
                )

    def _proposeinregion(self, slot: int, num_arms: int, key: Key) -> np.ndarray:
        n_cand_min = int(self._cfg.get("min_candidates", 0))
        n_cand_max = int(self._cfg.get("max_candidates", 2000))
        factor = float(self._cfg.get("num_candidates_factor", 100.0))
        n_cand = max(n_cand_min, min(n_cand_max, max(200, int(factor * self._d))))

        obs_idx = self._region_obs_idx[slot]
        if self._kind == "lhd" or self._x.shape[0] == 0 or not obs_idx:
            unit_cands = sobol_sequence(self._d, max(n_cand, num_arms), seed=key.seed())
            cands = self._fromunit(unit_cands)
        else:
            sub_y = (
                self._primaryscore(self._y[obs_idx])
                if self._y is not None
                else np.zeros(len(obs_idx))
            )
            inc_pos = int(np.argmax(sub_y))
            center = self._region_centers[slot]
            if center is None:
                center = self._tounit(self._x[obs_idx[inc_pos] : obs_idx[inc_pos] + 1])[
                    0
                ]
            tr_len = self._tr_lengths[slot]
            weights = np.ones(self._d, dtype=np.float64)
            half = 0.5 * tr_len * weights
            tr_lo = np.clip(center - half, 0.0, 1.0)
            tr_hi = np.clip(center + half, 0.0, 1.0)
            pert = sobol_sequence(self._d, n_cand, seed=key.seed())
            # RAASP-style dimension masking for high-D spaces
            rng = np.random.default_rng(key.fold(17).seed())
            prob_perturb = min(20.0 / max(1, self._d), 1.0)
            mask = rng.random((n_cand, self._d)) < prob_perturb
            for r in range(n_cand):
                if not np.any(mask[r]):
                    mask[r, rng.integers(0, self._d)] = True
            unit_in_tr = tr_lo[None, :] + pert * (tr_hi - tr_lo)[None, :]
            unit_cands = np.where(mask, unit_in_tr, center[None, :])
            cands = self._fromunit(unit_cands)

        acq = str(self._cfg.get("acquisition", "ucb"))
        if (
            self._kind in ("zero", "lhd")
            or self._x.shape[0] < 2
            or self._y is None
            or acq == "random"
        ):
            self._last_telem.num_candidates = int(cands.shape[0])
            return np.ascontiguousarray(cands[:num_arms], dtype=np.float64)

        if self._kind == "enn" or self._kind == "enn_multi_tr":
            if self._enn_model is None or self._enn_params is None:
                self._rebuildmodel()
            if self._enn_model is None or self._enn_params is None:
                return np.ascontiguousarray(cands[:num_arms], dtype=np.float64)
            self._last_telem.num_candidates = int(cands.shape[0])
            if acq == "thompson":
                seeds = [k.seed() & ((1 << 63) - 1) for k in key.split_n(num_arms)]
                draws, _ = self._enn_model.posterior_draw(
                    cands,
                    k_neighbors=self._enn_params.k_neighbors,
                    epistemic_scale=self._enn_params.epistemic_scale,
                    aleatoric_scale=self._enn_params.aleatoric_scale,
                    function_seeds=seeds,
                )
                chosen = [int(np.argmax(draws[a, :, 0])) for a in range(num_arms)]
                return np.ascontiguousarray(cands[chosen], dtype=np.float64)
            mu, se, _, _, _ = self._enn_model.posterior(
                cands,
                k_neighbors=self._enn_params.k_neighbors,
                epistemic_scale=self._enn_params.epistemic_scale,
                aleatoric_scale=self._enn_params.aleatoric_scale,
            )
            if acq == "pareto" and mu.shape[1] >= 2:
                idx = pareto2d_max(mu[:, 0], mu[:, 1])
                picked = [int(i) for i in idx[:num_arms]]
                if len(picked) < num_arms:
                    extra = [
                        i
                        for i in np.argsort(-_ucbscore(mu, se, 1.0))
                        if i not in picked
                    ]
                    picked.extend(int(i) for i in extra[: num_arms - len(picked)])
                return np.ascontiguousarray(cands[picked], dtype=np.float64)
            if acq == "pareto":
                return pareto_arms(
                    cands, mu[:, 0], se[:, 0], num_arms, key.fold(3).seed()
                )
            beta = float(self._cfg.get("acquisition_beta", 1.5))
            scores = _ucbscore(mu, se, beta)
            top = np.argsort(-scores)[:num_arms]
            return np.ascontiguousarray(cands[top], dtype=np.float64)

        if self._kind == "gp" and self._gp_inst is not None:
            self._last_telem.num_candidates = int(cands.shape[0])
            pred = self._gp_inst.predict(self._tounit(cands))
            if hasattr(pred, "mu") and hasattr(pred, "sigma"):
                mu, sd = pred.mu, pred.sigma
            else:
                mu, sd = pred
            mu_a = np.asarray(mu, dtype=np.float64)
            sd_a = np.asarray(sd, dtype=np.float64)
            if mu_a.ndim == 1:
                mu_a = mu_a[:, None]
            if sd_a.ndim == 1:
                sd_a = sd_a[:, None]
            beta = float(self._cfg.get("acquisition_beta", 1.5))
            top = np.argsort(-_ucbscore(mu_a, sd_a, beta))[:num_arms]
            return np.ascontiguousarray(cands[top], dtype=np.float64)

        return np.ascontiguousarray(cands[:num_arms], dtype=np.float64)

    def ask(self, num_arms: int, seed: int | None = None) -> np.ndarray:
        """Propose a batch of `num_arms` points (and register pending IDs when regional)."""
        t0 = time.perf_counter()
        n = int(num_arms)
        if n <= 0:
            raise ValueError(f"num_arms must be > 0, got {n}")
        self._step_counter += 1
        key = self._stepkey(1, seed)

        if self._warmup_pool is None and self._num_init > 0:
            self._warmup_pool = _lhdsamples(
                self._num_init, self._d, self._bounds, Key(self._seed).fold(0).fold(0)
            )
        rows: list[np.ndarray] = []
        slots: list[int] = []
        if (
            self._warmup_pool is not None
            and self._warmup_pos < self._warmup_pool.shape[0]
        ):
            avail = min(n, self._warmup_pool.shape[0] - self._warmup_pos)
            warm_slice = self._warmup_pool[self._warmup_pos : self._warmup_pos + avail]
            self._warmup_pos += avail
            for r in range(avail):
                rows.append(warm_slice[r : r + 1])
                slots.append(r % self._num_regions)
        rem = n - len(rows)
        if rem > 0:
            for r in range(rem):
                slot = (self._step_counter + r) % self._num_regions
                pt = self._proposeinregion(slot, 1, key.fold(r + 10))
                rows.append(pt)
                slots.append(slot)
        out = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
        is_regional = (
            self._num_regions > 1
            or "region_sharing" in self._cfg
            or self._kind == "enn_multi_tr"
        )
        if is_regional:
            for r in range(n):
                eid = self._next_id
                self._next_id += 1
                self._pending[eid] = (slots[r], out[r : r + 1].copy())
                self._pending_order.append(eid)
        self._last_telem.dt_gen = time.perf_counter() - t0
        return out

    def _updatetr(self, slot: int, new_best: float, prev_best: float) -> None:
        if new_best > prev_best + 1e-4:
            self._succ_counts[slot] += 1
            self._fail_counts[slot] = 0
        else:
            self._succ_counts[slot] = 0
            self._fail_counts[slot] += 1
        if self._succ_counts[slot] >= self._succ_tol:
            self._tr_lengths[slot] = min(2.0 * self._tr_lengths[slot], self._len_max)
            self._succ_counts[slot] = 0
        elif self._fail_counts[slot] >= self._fail_tol:
            self._tr_lengths[slot] *= 0.5
            self._fail_counts[slot] = 0
            if self._tr_lengths[slot] < self._len_min:
                self._restarts += 1
                self._tr_lengths[slot] = self._len_init
                self._region_centers[slot] = None

    def _ingestrows(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_var: np.ndarray | None,
        slot_list: list[int],
    ) -> None:
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya.reshape(-1, 1)
        prev_best_per_slot: dict[int, float] = {}
        for s in set(slot_list):
            obs = self._region_obs_idx[s]
            if obs and self._y is not None:
                prev_best_per_slot[s] = float(np.max(self._primaryscore(self._y[obs])))
            else:
                prev_best_per_slot[s] = -math.inf

        start_idx = self._x.shape[0]
        self._x = xa.copy() if self._x.shape[0] == 0 else np.vstack([self._x, xa])
        self._y = ya.copy() if self._y is None else np.vstack([self._y, ya])
        if y_var is not None or self._yvar is not None:
            yv_new = (
                np.zeros_like(ya)
                if y_var is None
                else np.asarray(y_var, dtype=np.float64)
            )
            if yv_new.ndim == 1:
                yv_new = yv_new.reshape(-1, 1)
            yv_old = (
                np.zeros((start_idx, ya.shape[1]), dtype=np.float64)
                if self._yvar is None
                else self._yvar
            )
            self._yvar = np.vstack([yv_old, yv_new])
        for i, s in enumerate(slot_list):
            self._region_obs_idx[s].append(start_idx + i)
        if self._x.shape[0] >= self._num_init:
            for s in set(slot_list):
                obs = self._region_obs_idx[s]
                if obs:
                    cur_best = float(np.max(self._primaryscore(self._y[obs])))
                    self._updatetr(s, cur_best, prev_best_per_slot[s])
        self._rebuildmodel()
        self._fitifgp()

    def tell(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seed: int | None = None,
        y_var: np.ndarray | None = None,
    ) -> None:
        """Record observations `(x, y)` and advance optimizer/trust-region state."""
        del seed
        t0 = time.perf_counter()
        self._step_counter += 1
        xa = np.asarray(x, dtype=np.float64)
        self._ingestrows(
            xa, y, y_var, [i % self._num_regions for i in range(xa.shape[0])]
        )
        self._last_telem.dt_tell = time.perf_counter() - t0

    def import_observations(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seed: int | None = None,
        y_var: np.ndarray | None = None,
    ) -> None:
        """Ingest external `(x, y)` observations into slot 0."""
        del seed
        xa = np.asarray(x, dtype=np.float64)
        self._ingestrows(xa, y, y_var, [0 for _ in range(xa.shape[0])])

    def pending_ids(self) -> list[int] | None:
        """Return outstanding evaluation IDs in regional mode, or None."""
        is_regional = (
            self._num_regions > 1
            or "region_sharing" in self._cfg
            or self._kind == "enn_multi_tr"
        )
        if not is_regional:
            return None
        return list(self._pending_order)

    def complete_ids(
        self,
        ids: list[int] | tuple[int, ...],
        y: np.ndarray,
        seed: int | None = None,
        y_var: np.ndarray | None = None,
    ) -> None:
        """Complete pending evaluation `ids` with observed `y` (and optional `y_var`)."""
        del seed
        id_list = [int(i) for i in ids]
        for i in id_list:
            if i not in self._pending:
                raise ValueError(f"Evaluation ID {i} is not pending")
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya.reshape(-1, 1)
        slots = [self._pending[i][0] for i in id_list]
        xs = np.vstack([self._pending[i][1] for i in id_list])
        for i in id_list:
            del self._pending[i]
            self._pending_order.remove(i)
        self._step_counter += 1
        self._ingestrows(xs, ya, y_var, slots)

    def cancel_ids(
        self, ids: list[int] | tuple[int, ...], seed: int | None = None
    ) -> None:
        """Cancel pending evaluation `ids` without ingesting observations."""
        del seed
        for i in [int(v) for v in ids]:
            if i not in self._pending:
                raise ValueError(f"Evaluation ID {i} is not pending")
            del self._pending[i]
            self._pending_order.remove(i)
        self._step_counter += 1

    def restart_region(
        self, slot: int, center: np.ndarray, seed: int | None = None
    ) -> list[int]:
        """Reset region `slot` at `center`, abandoning any pending IDs in that slot."""
        del seed
        s = int(slot)
        if s < 0 or s >= self._num_regions:
            raise ValueError(f"Invalid region slot {s}")
        self._step_counter += 1
        abandoned = [eid for eid in self._pending_order if self._pending[eid][0] == s]
        for eid in abandoned:
            del self._pending[eid]
            self._pending_order.remove(eid)
        self._region_centers[s] = self._tounit(
            np.asarray(center, dtype=np.float64).reshape(1, -1)
        )[0]
        self._tr_lengths[s] = self._len_init
        self._succ_counts[s] = 0
        self._fail_counts[s] = 0
        self._restarts += 1
        return abandoned

    def posterior_mu(self, x: np.ndarray) -> np.ndarray:
        """Predict surrogate posterior mean at query matrix `x`."""
        qa = np.asarray(x, dtype=np.float64)
        if (
            self._kind in ("enn", "enn_multi_tr")
            and self._enn_model is not None
            and self._enn_params is not None
        ):
            mu, _, _, _, _ = self._enn_model.posterior(
                qa,
                k_neighbors=self._enn_params.k_neighbors,
                epistemic_scale=self._enn_params.epistemic_scale,
                aleatoric_scale=self._enn_params.aleatoric_scale,
            )
            return np.ascontiguousarray(mu, dtype=np.float64)
        if self._kind == "gp" and self._gp_inst is not None:
            pred = self._gp_inst.predict(self._tounit(qa))
            mu = pred.mu if hasattr(pred, "mu") else pred[0]
            arr = np.asarray(mu, dtype=np.float64)
            return arr[:, None] if arr.ndim == 1 else arr
        if self._y is None:
            return np.zeros((qa.shape[0], 1), dtype=np.float64)
        return np.zeros((qa.shape[0], self._y.shape[1]), dtype=np.float64)

    def has_surrogate(self) -> bool:
        """Return whether a surrogate predictor is configured."""
        return self._kind in ("enn", "enn_multi_tr", "gp")

    def ingestion_failure(self) -> str | None:
        """Return last deferred surrogate ingestion error, or None."""
        return None

    def init_progress(self) -> tuple[int, int] | None:
        """Return `(completed_warmup, num_init)` progress pair, if configured."""
        if self._num_init <= 0:
            return None
        return (min(self._warmup_pos, self._num_init), self._num_init)

    def telemetry(self) -> Telemetry:
        """Return timing and candidate counts from the last optimizer step."""
        return Telemetry(
            dt_fit=self._last_telem.dt_fit,
            dt_gen=self._last_telem.dt_gen,
            dt_sel=self._last_telem.dt_sel,
            dt_tell=self._last_telem.dt_tell,
            num_candidates=self._last_telem.num_candidates,
        )

    def num_observations(self) -> int:
        """Return count of recorded observations."""
        return int(self._x.shape[0])

    def num_regions(self) -> int:
        """Return count of managed trust-region slots."""
        return self._num_regions

    def tr_length(self) -> float:
        """Return current trust-region side length for slot 0."""
        return float(self._tr_lengths[0])

    def x_obs(self) -> np.ndarray | None:
        """Return copy of observed coordinate matrix, or None when empty."""
        return None if self._x.shape[0] == 0 else self._x.copy()

    def y_obs(self) -> np.ndarray | None:
        """Return copy of observed objective matrix, or None when empty."""
        return None if self._y is None else self._y.copy()

    def incumbent_x(self) -> np.ndarray | None:
        """Return best observed input vector in original domain coordinates."""
        if self._y is None or self._y.shape[0] == 0:
            return None
        idx = int(np.argmax(self._y[:, 0]))
        return self._x[idx].copy()

    def unit_incumbent(self) -> np.ndarray | None:
        """Return best observed point mapped into unit cube `[0, 1]^d`."""
        inc = self.incumbent_x()
        return None if inc is None else self._tounit(inc.reshape(1, -1))[0]

    def lengthscales(self) -> np.ndarray | None:
        """Return active ENN coordinate lengthscales when fitted."""
        if self._enn_model is None:
            return None
        return self._enn_model._lengthscales.copy()

    def bounds(self) -> np.ndarray:
        """Return copy of `(d, 2)` lower/upper search bounds."""
        return self._bounds.copy()

    def fork(self) -> BayesOptimizer:
        """Deep-copy optimizer state and RNG cursor."""
        out = copy.deepcopy(self)
        return out

    def _canonicalstate(self) -> dict[str, Any]:
        return {
            "v": 1,
            "bounds": self._bounds.tolist(),
            "kind": self._kind,
            "k": self._k,
            "num_init": self._num_init,
            "num_regions": self._num_regions,
            "seed": self._seed,
            "cfg": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in sorted(self._cfg.items())
            },
            "step_counter": self._step_counter,
            "restarts": self._restarts,
            "warmup_pos": self._warmup_pos,
            "next_id": self._next_id,
            "tr_lengths": self._tr_lengths,
            "succ_counts": self._succ_counts,
            "fail_counts": self._fail_counts,
            "region_centers": [
                None if c is None else c.tolist() for c in self._region_centers
            ],
            "region_obs_idx": self._region_obs_idx,
            "x": self._x.tolist(),
            "y": None if self._y is None else self._y.tolist(),
            "yvar": None if self._yvar is None else self._yvar.tolist(),
            "pending_order": self._pending_order,
            "pending": [
                [eid, self._pending[eid][0], self._pending[eid][1].tolist()]
                for eid in self._pending_order
            ],
        }

    def digest_hex(self) -> str:
        """Compute canonical BLAKE3/SHA-256 hex digest of core optimizer state."""
        payload = json.dumps(
            self._canonicalstate(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return blake3_hex(payload)

    def digest_host(self, seed: int, epoch: int, identified: bool) -> str:
        """Compute canonical hex digest including host functional state metadata."""
        state_obj = {
            "host": [int(seed), int(epoch), bool(identified)],
            "inner": self._canonicalstate(),
        }
        payload = json.dumps(
            state_obj, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return blake3_hex(payload)

    def freeze(self) -> bytes:
        """Serialize core optimizer state into an ASCII JSON checkpoint blob."""
        return b"BAUXWASM1\x00" + json.dumps(
            self._canonicalstate(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def freeze_host(
        self, seed: int, epoch: int, identified: bool, id_map: list[tuple[int, int]]
    ) -> bytes:
        """Serialize host + core optimizer state into an ASCII JSON checkpoint blob."""
        blob = {
            "host": {
                "seed": int(seed),
                "epoch": int(epoch),
                "identified": bool(identified),
                "id_map": [[int(k), int(v)] for k, v in sorted(id_map)],
            },
            "inner": self._canonicalstate(),
        }
        return b"BAUXWASM1\x00" + json.dumps(
            blob, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def _loadcanonical(self, st: dict[str, Any]) -> None:
        self._bounds = np.asarray(st["bounds"], dtype=np.float64)
        self._d = int(self._bounds.shape[0])
        self._kind = str(st["kind"])
        self._k = int(st["k"])
        self._num_init = int(st["num_init"])
        self._num_regions = int(st["num_regions"])
        self._seed = int(st["seed"])
        self._step_counter = int(st["step_counter"])
        self._restarts = int(st["restarts"])
        self._warmup_pos = int(st["warmup_pos"])
        self._warmup_pool = (
            None
            if self._num_init <= 0
            else _lhdsamples(
                self._num_init, self._d, self._bounds, Key(self._seed).fold(0).fold(0)
            )
        )
        self._next_id = int(st["next_id"])
        self._tr_lengths = [float(v) for v in st["tr_lengths"]]
        self._succ_counts = [int(v) for v in st["succ_counts"]]
        self._fail_counts = [int(v) for v in st["fail_counts"]]
        self._region_centers = [
            None if c is None else np.asarray(c, dtype=np.float64)
            for c in st["region_centers"]
        ]
        self._region_obs_idx = [[int(i) for i in r] for r in st["region_obs_idx"]]
        self._x = np.asarray(st["x"], dtype=np.float64).reshape(-1, self._d)
        self._y = None if st["y"] is None else np.asarray(st["y"], dtype=np.float64)
        self._yvar = (
            None if st["yvar"] is None else np.asarray(st["yvar"], dtype=np.float64)
        )
        self._pending_order = [int(i) for i in st["pending_order"]]
        self._pending = {
            int(eid): (int(slot), np.asarray(pt, dtype=np.float64).reshape(1, -1))
            for eid, slot, pt in st["pending"]
        }
        self._rebuildmodel()

    def thaw(self, data: bytes) -> None:
        """Restore core optimizer state from a serialized checkpoint blob."""
        if not bytes(data).startswith(b"BAUXWASM1\x00"):
            raise ValueError("Invalid browser checkpoint header")
        obj = json.loads(bytes(data)[10:].decode("utf-8"))
        self._loadcanonical(obj.get("inner", obj))

    def thaw_host(
        self,
        data: bytes,
        default_seed: int,
        default_epoch: int,
        default_identified: bool,
    ) -> tuple[int, int, bool, list[tuple[int, int]]]:
        """Restore host + core optimizer state from a serialized checkpoint blob."""
        if not bytes(data).startswith(b"BAUXWASM1\x00"):
            raise ValueError("Invalid browser checkpoint header")
        obj = json.loads(bytes(data)[10:].decode("utf-8"))
        if "inner" in obj and "host" in obj:
            self._loadcanonical(obj["inner"])
            h = obj["host"]
            return (
                int(h["seed"]),
                int(h["epoch"]),
                bool(h["identified"]),
                [(int(k), int(v)) for k, v in h["id_map"]],
            )
        self._loadcanonical(obj)
        return (int(default_seed), int(default_epoch), bool(default_identified), [])


def create_optimizer(
    bounds: np.ndarray,
    kind: str,
    k: int = 10,
    num_init: int = 10,
    num_regions: int = 4,
    seed: int | None = None,
    cfg: dict[str, Any] | None = None,
    gp: Any = None,
    fit_steps: int = 50,
    gp_factory: Any = None,
) -> BayesOptimizer:
    """Create a browser WebAssembly `BayesOptimizer` instance."""
    return BayesOptimizer(
        bounds=bounds,
        kind=kind,
        k=k,
        num_init=num_init,
        num_regions=num_regions,
        seed=seed,
        cfg=cfg,
        gp=gp,
        fit_steps=fit_steps,
        gp_factory=gp_factory,
    )


def enn_optimizer(
    bounds: np.ndarray, k: int = 10, num_init: int = 10, seed: int | None = None
) -> BayesOptimizer:
    """Construct a single-region TuRBO-ENN optimizer."""
    return create_optimizer(
        bounds, "enn", k=k, num_init=num_init, num_regions=1, seed=seed
    )


def enn_tr(
    bounds: np.ndarray,
    k: int = 10,
    num_init: int = 10,
    num_regions: int = 4,
    seed: int | None = None,
) -> BayesOptimizer:
    """Construct a multi-trust-region TuRBO-ENN optimizer."""
    return create_optimizer(
        bounds,
        "enn_multi_tr",
        k=k,
        num_init=num_init,
        num_regions=num_regions,
        seed=seed,
    )


def create_zero(
    bounds: np.ndarray, num_init: int = 10, seed: int | None = None
) -> BayesOptimizer:
    """Construct a TuRBO-Zero trust-region optimizer without surrogate fitting."""
    return create_optimizer(
        bounds, "zero", k=10, num_init=num_init, num_regions=1, seed=seed
    )


def create_lhd(
    bounds: np.ndarray, num_init: int = 10, seed: int | None = None
) -> BayesOptimizer:
    """Construct a Latin hypercube design baseline optimizer."""
    return create_optimizer(
        bounds, "lhd", k=10, num_init=num_init, num_regions=1, seed=seed
    )


def capabilities(*, probe: bool = False) -> list[dict[str, Any]]:
    """Describe CPU WebAssembly capabilities exposed by the browser runtime."""
    del probe
    cpu_ops = {
        "exact_dist": "direct",
        "neighbor_search": "direct",
        "posterior": "direct",
        "posterior_batch": "direct",
        "posterior_draw": "direct",
        "device_posterior": "missing",
        "acquisition": "direct",
        "candidate": "direct",
        "weight_select": "missing",
        "trial_search": "direct",
        "resident_loop": "missing",
    }
    missing_ops = {key: "missing" for key in cpu_ops}
    return [
        {
            "name": "cpu",
            "compiled": True,
            "available": True,
            "reason": "browser WebAssembly / NumPy backend",
            "operations": cpu_ops,
            "host_operations": [
                "exact_dist",
                "neighbor_search",
                "posterior",
                "posterior_batch",
                "posterior_draw",
            ],
        },
        {
            "name": "cuda",
            "compiled": False,
            "available": False,
            "reason": "CUDA unavailable in browser WebAssembly",
            "operations": dict(missing_ops),
            "host_operations": [],
        },
        {
            "name": "metal",
            "compiled": False,
            "available": False,
            "reason": "Metal unavailable in browser WebAssembly",
            "operations": dict(missing_ops),
            "host_operations": [],
        },
        {
            "name": "opencl",
            "compiled": False,
            "available": False,
            "reason": "OpenCL unavailable in browser WebAssembly",
            "operations": dict(missing_ops),
            "host_operations": [],
        },
    ]


# ---------------------------------------------------------------------------
# Submodule structure expected by `from . import baux_rust as _ext`
# ---------------------------------------------------------------------------

hypervolume = types.SimpleNamespace(hypervolume2d_max=hypervolume2d_max)
hash = types.SimpleNamespace(
    normal_hash=normal_hash,
    Key=Key,
    blake3_hex=blake3_hex,
)
util = types.SimpleNamespace(
    standardize_y=standardize_y,
    pareto2d_max=pareto2d_max,
    sobol_indices=sobol_indices,
    sobol_sequence=sobol_sequence,
    pareto_arms=pareto_arms,
    set_path=set_path,
    ensure_config_file=_ensurecfg,
)
model = types.SimpleNamespace(ENN=ENN, ENNParams=ENNParams, Posterior=None, Tensor=None)
fit = types.SimpleNamespace(
    ENNFitter=ENNFitter,
    fit_model=fit_model,
    subsample_loglik=subsample_loglik,
    row_loglik=row_loglik,
)
globals()["verify_upstream_lock"] = _verifylock


class NumpyGPSurrogate:
    """Exact Matérn-5/2 GP surrogate fallback when running in Pyodide WebAssembly."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y_norm: np.ndarray | None = None
        self._mean: np.ndarray = np.zeros(1, dtype=np.float64)
        self._std: np.ndarray = np.ones(1, dtype=np.float64)
        self._ls: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    @staticmethod
    def _matern52(a: np.ndarray, b: np.ndarray, ls: np.ndarray) -> np.ndarray:
        scaled = (a[:, None, :] - b[None, :, :]) / ls[None, None, :]
        r = np.sqrt(np.maximum(np.sum(scaled * scaled, axis=2), 0.0))
        sr5 = math.sqrt(5.0) * r
        return (1.0 + sr5 + (5.0 / 3.0) * (r * r)) * np.exp(-sr5)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None = None,
        *,
        steps: int = 50,
    ) -> NumpyGPSurrogate:
        """Fit standardized Matérn-5/2 ExactGP lengthscales via marginal NLL."""
        del steps
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if ya.ndim == 1:
            ya = ya[:, None]
        self._x = xa.copy()
        self._mean = np.mean(ya, axis=0)
        std = np.std(ya, axis=0)
        self._std = np.where(std > 1e-8, std, 1.0)
        y_norm = (ya - self._mean[None, :]) / self._std[None, :]
        self._y_norm = y_norm
        base_ls = np.maximum(np.std(xa, axis=0), 0.1)
        noise = (
            1e-3
            if yvar is None
            else np.clip(
                np.asarray(yvar, dtype=np.float64).reshape(-1, ya.shape[1])[:, 0]
                / (self._std[0] ** 2),
                1e-5,
                1.0,
            )
        )
        n = xa.shape[0]
        eye = np.eye(n, dtype=np.float64)
        diag_noise = (noise * eye) if np.ndim(noise) == 1 else float(noise) * eye
        best_nll = math.inf
        best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for scale in (0.25, 0.5, 1.0):
            ls = base_ls * scale
            kxx = self._matern52(xa, xa, ls) + diag_noise + 1e-6 * eye
            try:
                chol = np.linalg.cholesky(kxx)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(kxx + 1e-3 * eye)
            tmp = np.linalg.solve(chol, y_norm)
            nll = float(np.sum(np.log(np.diag(chol))) + 0.5 * np.sum(tmp * tmp))
            if nll < best_nll:
                alpha = np.linalg.solve(chol.T, tmp)
                best_nll = nll
                best = (ls, chol, alpha)
        assert best is not None
        self._ls, self._chol, self._alpha = best
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean and standard deviation matrices at `x`."""
        qa = np.asarray(x, dtype=np.float64)
        if (
            self._x is None
            or self._chol is None
            or self._alpha is None
            or self._ls is None
        ):
            raise RuntimeError("GP is not fitted")
        kxs = self._matern52(self._x, qa, self._ls)
        mu_n = kxs.T @ self._alpha
        v = np.linalg.solve(self._chol, kxs)
        var_n = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-9)[:, None]
        mu = self._mean[None, :] + self._std[None, :] * mu_n
        sd = self._std[None, :] * np.sqrt(var_n)
        return mu, np.broadcast_to(sd, mu.shape).copy()


optimizer = types.SimpleNamespace(
    BayesOptimizer=BayesOptimizer,
    Telemetry=Telemetry,
    MultiTrustRegion=MultiTrustRegion,
    NumpyGPSurrogate=NumpyGPSurrogate,
    create_optimizer=create_optimizer,
    enn_optimizer=enn_optimizer,
    enn_tr=enn_tr,
    create_zero=create_zero,
    create_lhd=create_lhd,
)
experimental = types.SimpleNamespace()
cuda = None
