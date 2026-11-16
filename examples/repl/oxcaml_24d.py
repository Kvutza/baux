"""24D Levy: ENN vs Conformal vs Zero."""

import time

import baux
import numpy as np
from baux.benchmarks import synthetic

desc = synthetic.describe("levy", 24)
bounds = np.array(desc["bounds"], dtype=float)


def bench(pipe):
    """Benchmark optimization pipeline on 24-dimensional Levy."""
    t0 = time.perf_counter()
    st = baux.init(bounds, pipe=pipe, seed=0)
    while st.num_observations < 128:
        st, bat = baux.ask(st, 4)
        ys = -synthetic.eval_batch("levy", bat.points).reshape(-1, 1)
        st, _ = baux.tell(st, bat, ys)
    return st, (time.perf_counter() - t0) * 1e3


state, ms_enn = bench(
    baux.lhd(16) | baux.enn(num_samples=150) | baux.ucb() | baux.raasp(360)
)
st_conf, ms_conf = bench(
    baux.lhd(16)
    | baux.enn(k=5, geometry="conformal", num_samples=150)
    | baux.ucb()
    | baux.raasp(360)
)
st_zero, ms_zero = bench(baux.lhd(16) | baux.zero() | baux.raasp(360))

n_cat = len(synthetic.catalogue())
print(f"OxCaml catalogue size: {n_cat} functions")
print("24D Levy @ 128 evals (max -f, opt=0.0):")
print(f"  TuRBO-ENN (UCB)      : best={state.best_y:8.2f}  wall={ms_enn:5.0f} ms")
print(f"  TuRBO-ENN (Conformal): best={st_conf.best_y:8.2f}  wall={ms_conf:5.0f} ms")
print(f"  TuRBO-Zero           : best={st_zero.best_y:8.2f}  wall={ms_zero:5.0f} ms")

series = [
    {
        "label": "TuRBO-ENN",
        "y": np.maximum.accumulate(state.y_obs[:, 0]).tolist(),
    },
    {
        "label": "Conformal ENN",
        "y": np.maximum.accumulate(st_conf.y_obs[:, 0]).tolist(),
    },
    {
        "label": "TuRBO-Zero",
        "y": np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist(),
        "dashed": True,
    },
]

fig = {
    "kind": "curves",
    "title": "24D Levy Scaling (Equal 360-Candidate Budget)",
    "subtitle": ("Higher is better (-Levy loss; 0.0 is global optimum)"),
    "x_label": "Evaluations",
    "y_label": "Best Objective",
    "series": series,
}
