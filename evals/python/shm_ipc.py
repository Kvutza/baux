"""Zero-copy page-aligned mmap worker IPC for BAUX Python workers.

Mirrors evals/julia/shm_ipc.jl to interface with baux::unified_mem::ShmSlab.
"""

from __future__ import annotations

import mmap
import os
import socket
import struct
import time
from collections.abc import Callable, Sequence

import numpy as np

SHM_MAGIC = 0x4241555853484D31
READY_TAG = 0x5244593100000001

DBELL_CMD = 16
DBELL_SEQ = 17
DBELL_CNT = 18
ACK_SEQ = 32
ACK_STATUS = 33
ACK_US = 34
ACK_THREADS = 35

CMD_NOP = 0
CMD_EVAL = 1
CMD_DETACH = 2
CMD_SHUTDOWN = 3
CMD_SNAP = 4
CMD_TRAJ = 5

_U64_STRUCT = struct.Struct("<Q")


def isolate_stdio() -> None:
    """Redirect stdout (fd 1) to stderr (fd 2) to protect IPC streams."""
    try:
        os.dup2(2, 1)
    except OSError:
        pass


class SlabView:
    """Zero-copy memory mapped view into Rust ShmSlab."""

    def __init__(self, fileno: int, sz: int, expected_dim: int = 0):
        self.mm = mmap.mmap(fileno, sz, access=mmap.ACCESS_WRITE)
        hdr = struct.unpack_from("<10Q", self.mm, 0)
        if hdr[0] != SHM_MAGIC or hdr[1] != 1:
            raise ValueError(f"Invalid ShmSlab header: magic={hdr[0]:#x}, ver={hdr[1]}")
        self.max_batch = int(hdr[2])
        self.num_dim = int(hdr[3])
        self.state_len = int(hdr[4])
        self.cand_off = int(hdr[5])
        self.score_off = int(hdr[6])
        self.viol_off = int(hdr[7])
        self.state_off = int(hdr[8])
        self.num_obj = max(1, int(hdr[9]))
        if expected_dim > 0 and self.num_dim != expected_dim:
            raise ValueError(f"num_dim mismatch: slab={self.num_dim} vs {expected_dim}")
        self._views()

    def _views(self) -> None:
        self.cands = np.ndarray(
            (self.max_batch, self.num_dim),
            dtype=np.float32,
            buffer=self.mm,
            offset=self.cand_off,
        )
        self.scores = np.ndarray(
            (self.max_batch, self.num_obj),
            dtype=np.float64,
            buffer=self.mm,
            offset=self.score_off,
        )
        self.viols = np.ndarray(
            (self.max_batch,),
            dtype=np.float64,
            buffer=self.mm,
            offset=self.viol_off,
        )
        self.state = np.ndarray(
            (self.state_len,),
            dtype=np.float32,
            buffer=self.mm,
            offset=self.state_off,
        )

    def load_slot(self, slot: int) -> int:
        """Read a u64 doorbell or header slot from page 0."""
        return _U64_STRUCT.unpack_from(self.mm, slot * 8)[0]

    def store_slot(self, slot: int, val: int) -> None:
        """Write a u64 acknowledgment slot into page 0."""
        _U64_STRUCT.pack_into(self.mm, slot * 8, val)

    def close(self) -> None:
        """Release NumPy buffer views and close the underlying mmap."""
        del self.cands, self.scores, self.viols, self.state
        self.mm.close()


def open_slab(path: str, expected_dim: int = 0) -> SlabView:
    """Open and map an existing ShmSlab file."""
    sz = os.path.getsize(path)
    if sz < 16384:
        raise ValueError(f"Slab file too small: {sz} bytes")
    fd = os.open(path, os.O_RDWR)
    try:
        view = SlabView(fd, sz, expected_dim)
    finally:
        os.close(fd)
    return view


_WORKER_ERRORS = (
    ArithmeticError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _eval(eval_batch: Callable, cands: np.ndarray) -> tuple:
    try:
        return eval_batch(cands)
    except _WORKER_ERRORS:
        return None, None


def _traj(eval_traj: Callable, cands: np.ndarray, state: np.ndarray) -> tuple:
    try:
        return eval_traj(cands, state)
    except TypeError:
        return _fallback(eval_traj, cands)
    except _WORKER_ERRORS:
        return None, None


def _fallback(eval_traj: Callable, cands: np.ndarray) -> tuple:
    try:
        return eval_traj(cands)
    except _WORKER_ERRORS:
        return None, None


def _batch(
    slab: SlabView,
    cnt: int,
    eval_batch: Callable,
    penalty: float,
) -> tuple[int, int]:
    t0 = time.perf_counter_ns()
    sc, vi = _eval(eval_batch, slab.cands[:cnt])
    if sc is None:
        slab.scores[:cnt] = penalty
        slab.viols[:cnt] = 1e6
        return 1, (time.perf_counter_ns() - t0) // 1000
    slab.scores[:cnt] = sc.reshape((cnt, slab.num_obj))
    slab.viols[:cnt] = vi.reshape((cnt,))
    return 0, (time.perf_counter_ns() - t0) // 1000


def _rollout(
    slab: SlabView,
    cnt: int,
    eval_traj: Callable,
    penalty: float,
) -> tuple[int, int]:
    t0 = time.perf_counter_ns()
    sc, st = _traj(eval_traj, slab.cands[:cnt], slab.state)
    if sc is None:
        slab.scores[:cnt] = penalty
        slab.viols[:cnt] = 1e6
        return 1, (time.perf_counter_ns() - t0) // 1000
    slab.scores[:cnt] = sc.reshape((cnt, slab.num_obj))
    slab.viols[:cnt] = 0.0
    if slab.state_len > 0 and st is not None:
        slab.state[: min(slab.state_len, len(st))] = st[: slab.state_len]
    return 0, (time.perf_counter_ns() - t0) // 1000


def serve_atomic(
    slab: SlabView,
    eval_batch: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    eval_traj: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    penalty: float = -1e9,
) -> int:
    """Poll atomic doorbell slots on page 0 and evaluate batches."""
    last_seq = 0
    while True:
        seq = slab.load_slot(DBELL_SEQ)
        if seq <= last_seq:
            time.sleep(0.0001)
            continue
        cmd = slab.load_slot(DBELL_CMD)
        cnt = min(slab.max_batch, max(1, slab.load_slot(DBELL_CNT)))
        if cmd in (CMD_DETACH, CMD_SHUTDOWN):
            slab.store_slot(ACK_STATUS, 0)
            slab.store_slot(ACK_SEQ, seq)
            return int(cmd)
        if cmd == CMD_TRAJ and eval_traj is not None:
            st, us = _rollout(slab, cnt, eval_traj, penalty)
        else:
            st, us = _batch(slab, cnt, eval_batch, penalty)
        slab.store_slot(ACK_STATUS, st)
        slab.store_slot(ACK_US, us)
        slab.store_slot(ACK_THREADS, 1)
        slab.store_slot(ACK_SEQ, seq)
        last_seq = seq


def _bounds(conn: socket.socket, key: str, cache: dict, build_pool: Callable) -> None:
    _, _, (low, high) = cache.setdefault(key, build_pool(key))
    low_s = ",".join(str(x) for x in low)
    high_s = ",".join(str(x) for x in high)
    conn.sendall(f"BOUNDS\t{low_s}\t{high_s}\n".encode())


def _attach(
    conn: socket.socket,
    parts: list[str],
    cache: dict,
    build_pool: Callable,
    penalty: float,
) -> bool:
    key = parts[2] if len(parts) >= 3 else ""
    ev_b, ev_t, (low, _) = cache.setdefault(key, build_pool(key))
    slab = open_slab(parts[1], len(low))
    try:
        conn.sendall(f"OK\t{len(low)}\t{slab.state_len}\t{slab.num_obj}\n".encode())
        cmd = serve_atomic(slab, ev_b, ev_t, penalty)
    finally:
        slab.close()
    return cmd != CMD_SHUTDOWN


def _dispatch(
    conn: socket.socket,
    cache: dict,
    build_pool: Callable,
    penalty: float,
) -> bool:
    line = conn.recv(4096).decode("utf-8").strip()
    if not line or line == "PING":
        if line == "PING":
            conn.sendall(b"PONG\n")
        return True
    if line == "SHUTDOWN":
        conn.sendall(b"BYE\n")
        return False
    parts = line.split("\t")
    if parts[0] == "BOUNDS" and len(parts) >= 2:
        _bounds(conn, parts[1], cache, build_pool)
        return True
    if parts[0] == "OPEN" and len(parts) >= 2:
        return _attach(conn, parts, cache, build_pool, penalty)
    conn.sendall(b"ERR\tunknown\n")
    return True


def _accept(
    server: socket.socket, cache: dict, build_pool: Callable, penalty: float
) -> bool:
    conn, _ = server.accept()
    try:
        res = _dispatch(conn, cache, build_pool, penalty)
    finally:
        conn.close()
    return res


def run_daemon(
    sock_path: str,
    build_pool: Callable[
        [str], tuple[Callable, Callable, tuple[Sequence[float], Sequence[float]]]
    ],
    penalty: float = -1e9,
) -> None:
    """Listen on UNIX domain socket and serve atomic evaluations."""
    if os.path.exists(sock_path):
        os.remove(sock_path)
    os.makedirs(os.path.dirname(os.path.abspath(sock_path)), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(1)
    cache: dict = {}
    running = True
    try:
        while running:
            running = _accept(server, cache, build_pool, penalty)
    finally:
        server.close()
        if os.path.exists(sock_path):
            os.remove(sock_path)
