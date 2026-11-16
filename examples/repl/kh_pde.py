"""2D Kelvin-Helmholtz Shear PDE."""

import time

import baux
import numpy as np

N = 24
k1 = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi
kx, kz = np.meshgrid(k1, k1)
k2 = kx**2 + kz**2
k2_inv = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 > 0)
dealias = (np.abs(kx) < (2.0 / 3.0) * N * np.pi) & (
    np.abs(kz) < (2.0 / 3.0) * N * np.pi
)
grid1d = np.linspace(0.0, 1.0, N, endpoint=False)
X, Z = np.meshgrid(grid1d, grid1d)

w_bar = 4.0 / np.cosh((Z - 0.25) / 0.05) ** 2 - 4.0 / np.cosh((Z - 0.75) / 0.05) ** 2
w0 = w_bar + 1.2 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Z)
modes = [(1, 0), (1, 1), (1, -1), (2, 0), (2, 1)]
basis = np.stack(
    [f(2.0 * np.pi * (mx * X + mz * Z)) for mx, mz in modes for f in (np.sin, np.cos)],
    axis=0,
)
drive = 5.0 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Z) + 3.0 * np.cos(
    4.0 * np.pi * X
) * np.sin(2.0 * np.pi * Z)


def sim_kh(ctrl, nsteps=18, dt=0.025):
    """Simulate 2D KH shear layer under modal forcing."""
    force = np.tensordot(ctrl, basis, axes=(0, 0)) + drive
    w = w0.copy()
    for _ in range(nsteps):
        wh = np.fft.fft2(w)
        psih = wh * k2_inv
        u = np.fft.ifft2(1j * kz * psih).real
        v = np.fft.ifft2(-1j * kx * psih).real
        wx = np.fft.ifft2(1j * kx * wh * dealias).real
        wz = np.fft.ifft2(1j * kz * wh * dealias).real
        rhs = -(u * wx + v * wz) + force
        numer = wh + dt * np.fft.fft2(rhs) * dealias
        denom = 1.0 + 3e-3 * k2 * dt
        w = np.fft.ifft2(numer / denom).real
    cost = float(np.mean((w - w_bar) ** 2)) + 0.002 * float(np.sum(ctrl**2))
    return w, -cost


w_unc, y_unc = sim_kh(np.zeros(10))
bounds = np.array([[-6.0, 6.0]] * 10)


def run_kh(pipe):
    """Run BO loop for Kelvin-Helmholtz shear control."""
    st = baux.init(bounds, pipe=pipe, seed=3)
    while st.num_observations < 72:
        st, bat = baux.ask(st, 4)
        ys = np.array([[sim_kh(r)[1]] for r in bat.points])
        st, _ = baux.tell(st, bat, ys)
    return st


t0 = time.perf_counter()
state = run_kh(baux.lhd(12) | baux.enn(num_samples=150) | baux.ucb() | baux.raasp(300))
t_enn = (time.perf_counter() - t0) * 1e3
st_zero = run_kh(baux.lhd(12) | baux.zero() | baux.raasp(300))
w_opt, _ = sim_kh(state.best_x)
vmax = float(max(np.max(np.abs(w_unc)), np.max(np.abs(w_opt))))

print(
    f"2D KH shear (72 evals): uncontrolled={y_unc:.3f} -> "
    f"ENN={state.best_y:.3f} ({t_enn:.0f}ms) vs "
    f"Zero={st_zero.best_y:.3f}"
)
fig = {
    "kind": "field2d",
    "title": "2D Incompressible Shear Vorticity w(x, z)",
    "subtitle": (
        "10-mode spectral actuator control suppressing Kelvin-Helmholtz vortex rollup"
    ),
    "panels": [
        {
            "label": "Uncontrolled blowup",
            "nx": N,
            "nz": N,
            "vals": w_unc.ravel().tolist(),
            "vmin": -vmax,
            "vmax": vmax,
        },
        {
            "label": "TuRBO-ENN damped",
            "nx": N,
            "nz": N,
            "vals": w_opt.ravel().tolist(),
            "vmin": -vmax,
            "vmax": vmax,
        },
    ],
    "curves": {
        "title": "Perturbation Enstrophy Suppression (72 evals)",
        "y_label": "Negative Cost",
        "x_label": "Evaluations",
        "series": [
            {
                "label": "TuRBO-ENN (UCB)",
                "y": np.maximum.accumulate(state.y_obs[:, 0]).tolist(),
            },
            {
                "label": "TuRBO-Zero",
                "y": np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist(),
            },
            {
                "label": "Uncontrolled",
                "y": [y_unc] * state.num_observations,
                "dashed": True,
            },
        ],
    },
}
