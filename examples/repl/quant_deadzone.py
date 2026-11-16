"""FP16 Quantization Deadzone on SO(4)."""

import baux
import numpy as np

v_star = np.array([0.08, -0.06, 0.09, -0.05, 0.07, -0.04])
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def cayley(v):
    """Compute SO(4) rotation matrix via Cayley transform."""
    A = np.zeros((4, 4))
    for k, (i, j) in enumerate(pairs):
        A[i, j], A[j, i] = v[k], -v[k]
    eye = np.eye(4)
    return np.linalg.solve(eye + A, eye - A)


R_star = cayley(v_star)
bounds = np.array([[-0.35, 0.35]] * 6)


def run_so4(quant):
    """Run TuRBO trust region optimization on SO(4) Procrustes."""
    tr_len = baux.TrustRegionBounds(length_init=0.14, length_min=1e-4, length_max=0.4)
    cfg = (
        baux.lhd(8)
        | baux.enn()
        | baux.ucb()
        | baux.regions(
            1,
            success_tolerance=2,
            failure_tolerance=2,
            length=tr_len,
        )
    )
    st = baux.init(bounds, pipe=cfg, seed=21)
    while st.num_observations < 96:
        st, bat = baux.ask(st, 4)
        ys = []
        for row in bat.points:
            w = np.float16(32.0 + row).astype(np.float64) - 32.0 if quant else row
            err = -float(np.sum((cayley(w) - R_star) ** 2))
            ys.append([err])
        st, _ = baux.tell(st, bat, np.array(ys))
    return st


state = run_so4(False)
st_fp16 = run_so4(True)
print("SO(4) Procrustes @ bias=32.0 (ULP ~ 0.03125 in float16):")
print(
    f"  FP32 weights : best_err={-state.best_y:.3e} | tr_length={state.tr_length:.4f}"
)
print(
    f"  FP16 weights : best_err={-st_fp16.best_y:.3e} | "
    f"tr_length={st_fp16.tr_length:.4f}"
)

fig = {
    "kind": "curves",
    "title": "SO(4) Procrustes Frobenius Error (log scale)",
    "subtitle": (
        "float16 storage around bias=32 hits rounding floor while "
        "float32 descends smoothly"
    ),
    "x_label": "Evaluations",
    "y_label": "Frobenius Err",
    "log_y": True,
    "series": [
        {
            "label": "FP32 weights",
            "y": (-np.maximum.accumulate(state.y_obs[:, 0])).tolist(),
        },
        {
            "label": "FP16 @ bias=32",
            "y": (-np.maximum.accumulate(st_fp16.y_obs[:, 0])).tolist(),
        },
    ],
}
