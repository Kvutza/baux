"""SPARC Tokamak: 8-PF Equilibrium."""

import json
import os

import baux
import numpy as np


def load_sparc():
    """Load SPARC machine geometry from environment."""
    paths = (
        "baux_env.json",
        "forest/output/assets/baux_env.json",
        ".cache/pages-bundle/assets/baux_env.json",
    )
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["fuse"]["machines"]["sparc"]
    raise FileNotFoundError("baux_env.json not found")


sparc = load_sparc()
coils = sparc["coils"]
pf_slice = slice(14, 22)
pf_coils = coils[pf_slice]
nom_I = np.array([c["current_ma"] for c in pf_coils], dtype=float)
lcfs = np.array(sparc["lcfs"], dtype=float)
axR, axZ = float(sparc["axisR"]), float(sparc["axisZ"])
rx, zx = float(sparc["rx"]), float(sparc["zx"])
Ip = float(sparc["ipNominal"])


def agm_KE(k2):
    """Compute complete elliptic integrals K and E via AGM."""
    k2 = np.clip(k2, 0.0, 1.0 - 1e-10)
    a = np.ones_like(k2)
    b = np.sqrt(1.0 - k2)
    c2 = k2.copy()
    s = np.zeros_like(k2)
    p = 1.0
    for _ in range(6):
        s, p = s + p * c2, p * 2.0
        an, bn = 0.5 * (a + b), np.sqrt(a * b)
        c2, a, b = 0.25 * (a - b) ** 2, an, bn
    K = 0.5 * np.pi / a
    return K, K * (1.0 - 0.5 * s)


def fpsi(R, Z, Rc, Zc):
    """Compute poloidal magnetic flux from circular filament."""
    denom = (Rc + R) ** 2 + (Z - Zc) ** 2 + 1e-8
    k2 = (4.0 * Rc * R) / denom
    K, E = agm_KE(k2)
    k = np.sqrt(np.maximum(k2, 1e-12))
    return 0.4 * np.sqrt(Rc * R) * ((2.0 / k - k) * K - (2.0 / k) * E)


ptsR = np.concatenate([[axR], lcfs[:, 0], [rx + 2e-3, rx - 2e-3, rx, rx]])
ptsZ = np.concatenate([[axZ], lcfs[:, 1], [zx, zx, zx + 2e-3, zx - 2e-3]])
G = np.zeros((len(ptsR), 8))
psi_fix = Ip * fpsi(ptsR, ptsZ, axR, axZ)
for ic, c in enumerate(coils):
    g = fpsi(ptsR, ptsZ, float(c["r"]), float(c["z"]))
    if pf_slice.start <= ic < pf_slice.stop:
        G[:, ic - pf_slice.start] = g
    else:
        psi_fix += float(c["current_ma"]) * g


def sparc_score(xm):
    """Evaluate flux-surface variance and X-point penalty."""
    cur = np.asarray(xm, dtype=float)
    psi = psi_fix[None, :] + cur @ G.T
    lc = psi[:, 1 : 1 + len(lcfs)]
    sh = np.mean((lc - np.mean(lc, axis=1, keepdims=True)) ** 2, axis=1)
    dR = (psi[:, -4] - psi[:, -3]) / 4e-3
    dZ = (psi[:, -2] - psi[:, -1]) / 4e-3
    pen = (
        np.sqrt(sh + 1e-12)
        + 0.35 * np.sqrt(dR**2 + dZ**2 + 1e-12)
        + 1e-3 * np.mean(cur**2, axis=1)
    )
    return -pen[:, None]


bounds = np.array([[-13.0, 13.0]] * 8)


def run_eq(pipe, n_evals=96):
    """Run Bayesian optimization loop for SPARC PF equilibrium."""
    st = baux.init(bounds, pipe=pipe, seed=1)
    while st.num_observations < n_evals:
        st, bat = baux.ask(st, 4)
        st, _ = baux.tell(st, bat, sparc_score(bat.points))
    return st


state = run_eq(baux.lhd(12) | baux.enn() | baux.ucb() | baux.raasp(300))
st_zero = run_eq(baux.lhd(12) | baux.zero() | baux.raasp(300))
st_lhd = run_eq(baux.lhd(96, only=True))
nom_s = float(sparc_score(nom_I[None, :])[0, 0])
print(
    f"SPARC PF equilibrium (96 evals): ENN={state.best_y:.4f} | "
    f"Zero={st_zero.best_y:.4f} | LHD={st_lhd.best_y:.4f} | "
    f"FUSE-nom={nom_s:.4f}"
)
print("ENN coil currents [MA]:", np.round(state.best_x, 3))

nr, nz = 28, 28
rr, zz = np.meshgrid(np.linspace(0.45, 4.25, nr), np.linspace(-2.45, 2.45, nz))


def grid_psi(I_pf):
    """Compute 2D poloidal flux grid for given PF coil currents."""
    val = Ip * fpsi(rr, zz, axR, axZ)
    for ic, c in enumerate(coils):
        cur = (
            float(I_pf[ic - pf_slice.start])
            if pf_slice.start <= ic < pf_slice.stop
            else float(c["current_ma"])
        )
        val = val + cur * fpsi(rr, zz, float(c["r"]), float(c["z"]))
    return val


init_I = state.x_obs[0]


def lcfs_mean(I_pf):
    """Compute mean poloidal flux along LCFS for PF currents."""
    psi = psi_fix[None, :] + I_pf[None, :] @ G.T
    return float(np.mean(psi[0, 1 : 1 + len(lcfs)]))


coil_viz = []
for ic, c in enumerate(coils):
    if pf_slice.start <= ic < pf_slice.stop:
        cur = float(state.best_x[ic - pf_slice.start])
    else:
        cur = float(c["current_ma"])
    I_norm = cur / 13.0
    I_ma = cur
    coil_viz.append(
        [
            float(c["r"]),
            float(c["z"]),
            float(c["w"]),
            float(c["h"]),
            I_norm,
            I_ma,
        ]
    )

fig = {
    "kind": "tokamak",
    "title": "SPARC 8-PF Shaping Separatrix vs FUSE Target LCFS",
    "subtitle": (
        "AGM Biot-Savart flux contours matched to TEQUILA "
        "equilibrium & lower X-point null"
    ),
    "r_range": [0.45, 4.25],
    "z_range": [-2.45, 2.45],
    "first_wall": sparc["first_wall"],
    "lcfs": sparc["lcfs"],
    "x_point": [rx, zx],
    "coils": coil_viz,
    "flux_grid": {
        "nr": nr,
        "nz": nz,
        "psi_init": grid_psi(init_I).ravel().tolist(),
        "psi_opt": grid_psi(state.best_x).ravel().tolist(),
        "psi_init_target": lcfs_mean(init_I),
        "psi_opt_target": lcfs_mean(state.best_x),
    },
    "curves": {
        "title": "PF Shaping Objective Trajectory (96 evals)",
        "y_label": "Equilibrium Score",
        "x_label": "Evaluations",
        "series": [
            {
                "label": "TuRBO-ENN",
                "y": np.maximum.accumulate(state.y_obs[:, 0]).tolist(),
            },
            {
                "label": "TuRBO-Zero",
                "y": np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist(),
            },
            {
                "label": "LHD-only",
                "y": np.maximum.accumulate(st_lhd.y_obs[:, 0]).tolist(),
                "dashed": True,
            },
        ],
    },
}
