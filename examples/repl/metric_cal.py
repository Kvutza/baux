"""Non-Euclidean ENN & McShane-Whitney."""

import baux
import numpy as np

rng = np.random.default_rng(19)
x_tr = np.concatenate(
    [
        np.linspace(-1.6, 0.25, 22),
        np.linspace(1.35, 2.2, 16),
        np.array([0.82]),
    ]
)[:, None]


def f_true(x):
    """Evaluate 1D ground-truth synthetic test function."""
    return np.sin(4.2 * x) * np.exp(-0.22 * x**2) + 0.35 * x


y_tr = f_true(x_tr) + 0.03 * rng.normal(size=x_tr.shape)
y_var = np.full_like(y_tr, 0.03**2)
xg = np.linspace(-1.6, 2.2, 81)[:, None]
yg = f_true(xg).ravel()
kfit = baux.key(29)
flags = baux.enn.PosteriorFlags(observation_noise=False)

posts = {}
for geom in ["euclidean", "conformal", "hyperbolic"]:
    es = baux.with_geometry(
        baux.enn_init(x_tr, y_tr, y_var, scale_x=True),
        geom,
        local_k=4,
    )
    pr = baux.fit(es, k=6, infer_aleatoric=False, key=kfit)
    po = baux.posterior(es, xg, pr, flags=flags)
    rmse = float(np.sqrt(np.mean((po.mu.ravel() - yg) ** 2)))
    cov95 = float(np.mean(np.abs(yg - po.mu.ravel()) <= 2.0 * po.se.ravel()))
    posts[geom] = (po.mu.ravel(), po.se.ravel())
    print(f"{geom:10s} | RMSE={rmse:.4f} | empirical 95% cov={100.0 * cov95:5.1f}%")

dx = np.abs(x_tr[:, 0:1] - x_tr[:, 0:1].T)
dy = np.abs(y_tr[:, 0:1] - y_tr[:, 0:1].T)
L_est = float(np.max(np.divide(dy, dx, out=np.zeros_like(dy), where=dx > 0.08)))
dxg = np.abs(xg[:, 0:1] - x_tr[:, 0].T)
mw_hi = np.min(y_tr[:, 0][None, :] + L_est * dxg, axis=1)
mw_lo = np.max(y_tr[:, 0][None, :] - L_est * dxg, axis=1)
mw_cov = 100.0 * np.mean((yg >= mw_lo) & (yg <= mw_hi))
print(f"McShane-Whitney L_est={L_est:.2f} | envelope cov={mw_cov:.1f}%")

mu_c, se_c = posts["conformal"]
mu_e, _ = posts["euclidean"]
fig = {
    "kind": "band1d",
    "title": "Conformal ENN Posterior vs McShane-Whitney Envelope",
    "subtitle": (
        "Sparse sampling gap in [0.25, 1.35] tests epistemic "
        "expansion and Lipschitz bounds"
    ),
    "x_grid": xg.ravel().tolist(),
    "y_true": yg.tolist(),
    "mu": mu_c.tolist(),
    "lo_2se": (mu_c - 2.0 * se_c).tolist(),
    "hi_2se": (mu_c + 2.0 * se_c).tolist(),
    "mw_lo": mw_lo.tolist(),
    "mw_hi": mw_hi.tolist(),
    "train_x": x_tr.ravel().tolist(),
    "train_y": y_tr.ravel().tolist(),
    "curves": {
        "title": ("Pointwise Uncertainty Width (2 * se) Across Spatial Gap"),
        "x_label": "Domain x",
        "y_label": "2 * se(x)",
        "series": [
            {
                "label": "Conformal 2*se",
                "x": xg.ravel().tolist(),
                "y": (2.0 * se_c).tolist(),
            },
            {
                "label": "Euclidean |err|",
                "x": xg.ravel().tolist(),
                "y": np.abs(mu_e - yg).tolist(),
                "dashed": True,
            },
        ],
    },
}
