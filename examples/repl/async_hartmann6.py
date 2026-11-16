"""Async Hartmann-6D."""

import baux
import numpy as np

c_h = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ],
    dtype=float,
)
a_h = np.array([1.0, 1.2, 3.0, 3.2])
p_h = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ],
    dtype=float,
)


def hart6(x):
    """Evaluate 6-dimensional Hartmann test function."""
    diff = x[:, None, :] - p_h[None, :, :]
    ex = np.sum(c_h[None, :, :] * diff**2, axis=-1)
    return np.sum(a_h[None, :] * np.exp(-ex), axis=-1)[:, None]


bounds = np.array([[0.0, 1.0]] * 6)
cfg = (
    baux.lhd(12)
    | baux.enn(num_samples=150)
    | baux.ucb()
    | baux.raasp(240)
    | baux.regions(2)
)
blank = baux.init(bounds, pipe=cfg, key=baux.key(31))
st = blank
st, bw = baux.ask(st, 12)
st = baux.complete(st, bw.ids[6:], hart6(bw.points[6:]))
st = baux.complete(st, bw.ids[:6], hart6(bw.points[:6]))

blob = baux.freeze(st)
d_pre = baux.digest(st)
state = baux.thaw(blank, blob)
assert baux.digest(state) == d_pre

while state.num_observations < 84:
    state, bat = baux.ask(state, 6)
    state = baux.cancel(state, bat.ids[4:])
    state = baux.complete(state, bat.ids[:4], hart6(bat.points[:4]))

st_zero = baux.init(
    bounds,
    pipe=baux.lhd(12) | baux.zero() | baux.raasp(240),
    key=baux.key(31),
)
while st_zero.num_observations < 84:
    st_zero, bz = baux.ask(st_zero, 4)
    st_zero, _ = baux.tell(st_zero, bz, hart6(bz.points))

pending = len(state.pending_ids or [])
print(f"Checkpoint bytes={len(blob)} BLAKE3={d_pre[:16]}... pending_end={pending}")
print(
    f"Hartmann-6D (opt=3.3224): Async ENN={state.best_y:.4f} vs "
    f"Zero={st_zero.best_y:.4f}"
)

ys_enn = np.maximum.accumulate(state.y_obs[:, 0])
ys_zero = np.maximum.accumulate(st_zero.y_obs[:, 0])

fig = {
    "kind": "curves",
    "title": "Hartmann-6D Async Straggler Cancellation vs Zero",
    "subtitle": (
        "2-region async complete() + cancel() with freeze/thaw BLAKE3 continuity"
    ),
    "x_label": "Completed Evals",
    "y_label": "Best Hartmann-6D",
    "series": [
        {
            "label": "Async TuRBO-ENN",
            "y": ys_enn.tolist(),
        },
        {
            "label": "TuRBO-Zero",
            "y": ys_zero.tolist(),
            "dashed": True,
        },
    ],
}
