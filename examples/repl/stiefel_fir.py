"""Mixed Manifold + Stein Shrinkage."""

import math

import baux
import numpy as np

space = {
    "weights": baux.Simplex(4),
    "phase": baux.Periodic(0.0, 2.0 * math.pi),
    "decay": baux.Log(1e-3, 1.0),
    "window": baux.Categorical(["hann", "kaiser", "blackman"]),
}
w_star = np.array([0.52, 0.28, 0.14, 0.06])


def clean_fir(p):
    """Compute noiseless FIR filter objective score."""
    w = np.asarray(p["weights"])
    win_table = {"hann": 0.08, "kaiser": 0.14, "blackman": 0.05}
    win = win_table[p["window"]]
    return -(
        float(np.sum((w - w_star) ** 2))
        + 0.08 * (1.0 - math.cos(float(p["phase"]) - 1.1))
        + 0.03 * (math.log10(float(p["decay"])) + 1.5) ** 2
        - win
    )


def run_fir(cfg, seed_noise=13):
    """Run Bayesian optimization loop for FIR filter design."""
    rng = np.random.default_rng(seed_noise)
    st = baux.init(space, pipe=cfg, key=baux.key(8))
    clean_trace = []
    while st.num_observations < 44:
        st, bat = baux.ask(st, 4)
        ys = np.array([[clean_fir(p) + 0.04 * rng.normal()] for p in bat.params])
        st, _ = baux.tell(st, bat, ys)
        clean_trace.append(clean_fir(st.best_params))
    return st, clean_trace


st_unshrunk, tr_un = run_fir(baux.lhd(8) | baux.enn() | baux.ucb() | baux.tr_bounds())
frame = baux.Stiefel.from_factors(np.eye(st_unshrunk.dim, 2))
state, tr_stein = run_fir(
    baux.lhd(8)
    | baux.enn()
    | baux.stein(frame=frame, rank=2)
    | baux.ucb()
    | baux.tr_bounds()
)

val_un = clean_fir(st_unshrunk.best_params)
val_st = clean_fir(state.best_params)
print(f"Embedded dim={state.dim} (max clean score = 0.14000)")
print(f"Unshrunk ENN : noiseless={val_un:+.5f}")
print(f"Stein(rank=2): noiseless={val_st:+.5f}")
print("Stein best_params:", state.best_params)

fig = {
    "kind": "curves",
    "title": "Noiseless FIR Filter Quality at Selected Incumbent",
    "subtitle": (
        "Stein subspace shrinkage suppresses false spikes from 0.04 observation noise"
    ),
    "x_label": "Batch Step",
    "y_label": "Noiseless Score",
    "series": [
        {"label": "Stein(frame, rank=2)", "y": tr_stein},
        {"label": "Unshrunk ENN", "y": tr_un, "dashed": True},
    ],
}
