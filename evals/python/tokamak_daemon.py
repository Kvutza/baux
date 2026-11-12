"""Zero-copy persistent daemon for TORAX and TokaGym tokamak workers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import click
import numpy as np

from evals.python.shm_ipc import isolate_stdio, run_daemon


def _torax() -> tuple[Callable, Callable, tuple[Sequence[float], Sequence[float]]]:
    from evals.torax.torax_rollout import TORAX_STEPS, TOTAL_PARAMS, ToraxClosedLoopEnv

    env = ToraxClosedLoopEnv.shared()
    dummy = np.zeros((1, TOTAL_PARAMS), dtype=np.float32)
    env.eval_batch(dummy)

    def eval_batch(cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = env.eval_batch(cands)
        viols = np.zeros((len(scores),), dtype=np.float64)
        return scores, viols

    def eval_traj(
        cands: np.ndarray, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        w = cands[0]
        n_eps = TORAX_STEPS * 19
        if state is not None and len(state) >= n_eps:
            eps = np.array(state[:n_eps], dtype=np.float32).reshape(TORAX_STEPS, 19)
        else:
            eps = np.zeros((TORAX_STEPS, 19), dtype=np.float32)
        score, obs, u, r = env.eval_traj(w, eps)
        st = np.concatenate([obs.flatten(), u.flatten(), r.flatten()]).astype(
            np.float32
        )
        return np.array([score], dtype=np.float64), st

    bounds = ([-1.0] * TOTAL_PARAMS, [1.0] * TOTAL_PARAMS)
    return eval_batch, eval_traj, bounds


def _tokagym() -> tuple[Callable, Callable, tuple[Sequence[float], Sequence[float]]]:
    from evals.tokagym.tokagym_rollout import (
        ACTION_DIM,
        TOKAGYM_STEPS,
        TOTAL_PARAMS,
        TokaGymClosedLoopEnv,
    )

    env = TokaGymClosedLoopEnv.shared()
    dummy = np.zeros((1, TOTAL_PARAMS), dtype=np.float32)
    env.eval_batch(dummy)

    def eval_batch(cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = env.eval_batch(cands)
        viols = np.zeros((len(scores),), dtype=np.float64)
        return scores, viols

    def eval_traj(
        cands: np.ndarray, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        w = cands[0]
        n_eps = TOKAGYM_STEPS * ACTION_DIM
        if state is not None and len(state) >= n_eps:
            eps = np.array(state[:n_eps], dtype=np.float32).reshape(
                TOKAGYM_STEPS, ACTION_DIM
            )
        else:
            eps = np.zeros((TOKAGYM_STEPS, ACTION_DIM), dtype=np.float32)
        score, obs, u, r = env.eval_traj(w, eps)
        st = np.concatenate([obs.flatten(), u.flatten(), r.flatten()]).astype(
            np.float32
        )
        return np.array([score], dtype=np.float64), st

    bounds = ([-1.0] * TOTAL_PARAMS, [1.0] * TOTAL_PARAMS)
    return eval_batch, eval_traj, bounds


def _builder(kind: str) -> Callable:
    def builder(_key: str):
        if kind == "tokagym":
            return _tokagym()
        return _torax()

    return builder


def run_server(kind: str, sock: str) -> None:
    """Start the persistent UNIX socket daemon for the selected tokamak worker."""
    isolate_stdio()
    builder = _builder(kind)
    pen = -500.0 if kind == "tokagym" else -400.0
    run_daemon(sock, builder, penalty=pen)


cli = click.Command(
    "tokamak_daemon",
    callback=run_server,
    params=[
        click.Option(
            ["--kind"], type=click.Choice(["torax", "tokagym"]), default="torax"
        ),
        click.Option(["--sock"], required=True, help="UNIX domain socket path"),
    ],
)

if __name__ == "__main__":
    cli()
