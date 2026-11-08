"""Quantitative evaluations for non-Cartesian manifold embeddings, hyperbolic k-NN, Stein shrinkage, and WLS calibration.

References:
- Poincaré, H. (1882). "Théorie des groupes fuchsiens." Acta Math., 1:1-62.
- Cannon, J. W., Floyd, W. J., Kenyon, R., & Parry, W. R. (1997). "Hyperbolic
  geometry." Flavors of Geometry, MSRI Publ., 31:59-115.
- Nickel, M. & Kiela, D. (2017). "Poincaré embeddings for learning hierarchical
  representations." NeurIPS 30 (arXiv:1705.08039).
- Zelnik-Manor, L. & Perona, P. (2004). "Self-tuning spectral clustering."
  NeurIPS 17:1601-1608.
- Stein, C. (1956). "Inadmissibility of the usual estimator for the mean of a
  multivariate normal distribution." Proc. Third Berkeley Symp., 1:197-206.
- James, W. & Stein, C. (1961). "Estimation with quadratic loss." Proc. Fourth
  Berkeley Symp., 1:361-379.
- Efron, B. & Morris, C. (1973). "Stein's estimation rule and its competitors—an
  empirical Bayes approach." J. Amer. Statist. Assoc., 68(341):117-130.
- Higham, N. J. (1986). "Computing the polar decomposition—with applications."
  SIAM J. Sci. Stat. Comput., 7(4):1160-1174.
- Aitken, A. C. (1935). "On least squares and linear combination of observations."
  Proc. R. Soc. Edinburgh, 55:42-48.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random
  World. Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018).
  "Distribution-free predictive inference for regression." J. Amer. Statist.
  Assoc., 113(523):1094-1111 (arXiv:1604.04173).
- Johnson, W. B. & Lindenstrauss, J. (1984). "Extensions of Lipschitz mappings
  into a Hilbert space." Contemp. Math., 26:189-206.
- Achlioptas, D. (2003). "Database-friendly random projections: Johnson-
  Lindenstrauss with binary coins." J. Comput. Syst. Sci., 66(4):671-687.
- Bhattacharyya, A. (1943). "On a measure of divergence between two statistical
  populations defined by their probability distributions." Bull. Calcutta Math.
  Soc., 35:99-109.
- Čencov, N. N. (1982). Statistical Decision Rules and Optimal Inference.
  Translations of Mathematical Monographs, Vol. 53, AMS.
"""

from __future__ import annotations

import math
import time

import numpy as np

import baux


def _loopair(
    x: np.ndarray, y: np.ndarray, metric: str, local_k: int = 3
) -> tuple[float, float]:
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    if metric == "euclidean":
        delta = d2.copy()
    else:
        sorted_d2 = np.sort(np.where(d2 > 1e-18, d2, np.inf), axis=1)
        sig = np.sqrt(np.maximum(sorted_d2[:, local_k - 1], 1e-12))
        prod = np.maximum(sig[:, None] * sig[None, :], 1e-12)
        delta = (
            d2 / prod
            if metric == "conformal"
            else (d2 + (sig[:, None] - sig[None, :]) ** 2) / (2.0 * prod)
        )
    np.fill_diagonal(delta, np.inf)
    idx = np.argsort(delta, axis=1)[:, :8]
    top_d = np.take_along_axis(delta, idx, axis=1)
    w = 1.0 / np.maximum(0.36 * top_d + 0.0025, 1e-18)
    wn = w / np.sum(w, axis=1, keepdims=True)
    y_knn = y.ravel()[idx]
    mu = np.sum(wn * y_knn, axis=1)
    res2 = np.sum(wn * ((y_knn - mu[:, None]) ** 2), axis=1)
    v0 = np.maximum(res2 + 0.02 * np.sum(wn * top_d, axis=1) + 1e-4, 1e-12)
    diff2 = (y.ravel() - mu) ** 2
    var = np.maximum(float(np.mean(diff2 / v0)) * v0, 1e-12)
    return float(np.sqrt(np.mean(diff2))), float(
        0.5 * np.mean(diff2 / var + np.log(2.0 * np.pi * var))
    )


def eval_hyperbolic() -> dict[str, float]:
    """Evaluate Poincaré hyperbolic and Conformal local-scale k-NN vs flat Euclidean on f(x) = sin(1 / (x + 0.05))."""
    x = np.concatenate(
        [np.linspace(0.0, 0.16, 72, endpoint=False), np.linspace(0.16, 1.0, 24)]
    ).reshape(-1, 1)
    y = np.sin(1.0 / (x[:, 0] + 0.05)).reshape(-1, 1)
    rmse_euc, nll_euc = _loopair(x, y, "euclidean", local_k=3)
    rmse_conf, nll_conf = _loopair(x, y, "conformal", local_k=3)
    rmse_hyp, nll_hyp = _loopair(x, y, "hyperbolic", local_k=3)
    return {
        "rmse_euclidean": rmse_euc,
        "rmse_conformal": rmse_conf,
        "rmse_hyperbolic": rmse_hyp,
        "nll_euclidean": nll_euc,
        "nll_conformal": nll_conf,
        "nll_hyperbolic": nll_hyp,
    }


def eval_stein() -> dict[str, float]:
    """Evaluate Stiefel `Shrinkage.from_factors` + `transform_stein` across M=6 outputs with rank F=2 structure."""
    k_theta, k_se, k_noise = baux.split(baux.key("eval/stein"), 3)
    factors = np.array(
        [[1.2, -0.4], [0.9, 0.3], [-0.8, 1.1], [0.5, 0.9], [-1.0, -0.6], [0.7, -0.8]]
    )
    sh = baux.Shrinkage.from_factors(factors)
    stiefel_err = float(np.linalg.norm(sh.loadings.T @ sh.loadings - np.eye(2)))
    theta = (
        np.random.default_rng(k_theta.seed()).normal(0.0, 1.0, size=(200, 2))
        @ sh.loadings.T
    )
    se = np.random.default_rng(k_se.seed()).uniform(0.25, 0.75, size=(200, 6))
    mu_raw = theta + se * np.random.default_rng(k_noise.seed()).normal(
        0.0, 1.0, size=(200, 6)
    )
    mu_stein, se_stein = sh.transform_stein(mu_raw, se)
    mse_raw = float(np.mean((mu_raw - theta) ** 2))
    mse_stein = float(np.mean((mu_stein - theta) ** 2))
    return {
        "mse_raw": mse_raw,
        "mse_stein": mse_stein,
        "risk_ratio": mse_stein / mse_raw,
        "stiefel_err": stiefel_err,
        "se_reduction": float(np.mean(se_stein) / np.mean(se)),
    }


def eval_calibrator() -> dict[str, float]:
    """Evaluate WLS `Calibrator.fit_wls` and `conformal_scale` vs unweighted OLS under heteroscedastic noise."""
    rng_cal = np.random.default_rng(baux.key("eval/calibrator/cal").seed())
    rng_test = np.random.default_rng(baux.key("eval/calibrator/test").seed())
    a_true, b_true, alpha = 0.5, 1.8, 0.1
    mu_cal = rng_cal.uniform(-2.0, 2.0, size=(500, 1))
    se_cal = np.linspace(0.05, 2.0, 500)[:, None]
    y_cal = a_true + b_true * mu_cal + se_cal * rng_cal.normal(0.0, 1.0, size=(500, 1))
    cal_ols = baux.Calibrator.fit(mu_cal, y_cal, se_cal)
    cal_wls = baux.Calibrator.fit_wls(mu_cal, y_cal, se_cal)
    q = cal_wls.conformal_scale(mu_cal, y_cal, se_cal, alpha=alpha)
    mu_t = rng_test.uniform(-2.0, 2.0, size=(1000, 1))
    se_t = rng_test.choice(se_cal[:, 0], size=(1000, 1), replace=True)
    y_t = a_true + b_true * mu_t + se_t * rng_test.normal(0.0, 1.0, size=(1000, 1))
    cov = float(
        np.mean(np.abs(y_t - (cal_wls.a[0] + cal_wls.b[0] * mu_t)) <= q[0] * se_t)
    )
    return {
        "slope_err_ols": float(abs(cal_ols.b[0] - b_true)),
        "slope_err_wls": float(abs(cal_wls.b[0] - b_true)),
        "intercept_err_ols": float(abs(cal_ols.a[0] - a_true)),
        "intercept_err_wls": float(abs(cal_wls.a[0] - a_true)),
        "conformal_q": float(q[0]),
        "empirical_coverage": cov,
        "target_coverage": 1.0 - alpha,
    }


def _mahalanobisrate() -> float:
    k_cat, k_draws = baux.split(baux.key("eval/mahalanobis"), 2)
    cat = baux.Categorical(
        tuple(f"c{i}" for i in range(16)), encoding="rademacher", key=k_cat
    )
    cat_sp, cb = baux.Space(cat), cat.codebook()
    rng = np.random.default_rng(k_draws.seed())
    matches = 0
    for _ in range(100):
        v = rng.normal(0.0, 1.0, size=cat.width)
        g = rng.uniform(0.1, 10.0, size=cat.width)
        exact = cat.categories[
            int(np.argmin(np.sum(g[None, :] * ((v[None, :] - cb) ** 2), axis=1)))
        ]
        matches += int(cat_sp.unflatten(v, weights=g) == exact)
    return matches / 100.0


def _simplexerrs() -> tuple[float, float]:
    simp = baux.Space(baux.Simplex(4))
    p, q = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])
    psi_d2 = float(np.sum((simp.ravel(p) - simp.ravel(q)) ** 2))
    h_err = abs(psi_d2 - float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    ret_p = simp.unflatten(np.array([-0.5, 1.0, 2.0, 0.0]))
    return h_err, float(abs(np.sum(ret_p) - 1.0))


def eval_embeddings() -> dict[str, float]:
    """Evaluate `Space` with `Periodic`, `Simplex`, `Log`, and `Categorical` Riemannian embeddings."""
    s1 = baux.Space(baux.Periodic(0.0, 2.0 * math.pi))
    d_wrap = float(np.linalg.norm(s1.ravel(0.01) - s1.ravel(2.0 * math.pi - 0.01)))
    d_opp = float(np.linalg.norm(s1.ravel(0.01) - s1.ravel(math.pi)))
    h_err, s_err = _simplexerrs()
    log_mid = float(baux.Space(baux.Log(1e-4, 1.0)).unflatten(0.5))
    return {
        "s1_wrap_dist": d_wrap,
        "s1_opp_dist": d_opp,
        "hellinger_err": h_err,
        "simplex_sum_err": s_err,
        "log_mid_err": abs(log_mid - math.sqrt(1e-4 * 1.0)),
        "mahalanobis_match_rate": _mahalanobisrate(),
    }


def eval_pmap() -> dict[str, float]:
    """Evaluate parallel `pmap` + `scan` trajectory throughput and BLAKE3 digest determinism."""
    cfg = baux.chain(baux.sobol(4), baux.enn(k=3, num_samples=8), baux.ucb())
    k_init, k_pmap = baux.split(baux.key("eval/pmap"), 2)
    root = baux.init([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]], cfg, key=k_init)
    branches, steps = 4, 4

    def branch_fn(idx: int, st: baux.OptimizerState, bk: baux.Key) -> tuple[str, float]:
        shift = 0.2 * float(idx)
        fin, _, traces = baux.scan(
            st, bk, steps, lambda x: -np.sum((x - shift) ** 2, axis=1), num_arms=2
        )
        return fin.digest(), float(traces[-1]["best_y"])

    t0 = time.perf_counter()
    run_a = baux.pmap(root, k_pmap, branches, branch_fn)
    dt = max(time.perf_counter() - t0, 1e-9)
    run_b = baux.pmap(root, k_pmap, branches, branch_fn)
    digests_a = [d for d, _ in run_a]
    matches = sum(int(a == b) for a, b in zip(run_a, run_b))
    return {
        "digest_match_rate": matches / float(branches),
        "unique_branch_ratio": len(set(digests_a)) / float(branches),
        "steps_per_sec": (branches * steps) / dt,
        "mean_best_y": float(np.mean([y for _, y in run_a])),
    }


def run_evals() -> dict[str, dict[str, float]]:
    """Run all 5 geometry, manifold embedding, and functional combinator evaluations and return their metrics."""
    return {
        "hyperbolic_knn": eval_hyperbolic(),
        "stein_shrinkage": eval_stein(),
        "wls_calibrator": eval_calibrator(),
        "manifold_embeddings": eval_embeddings(),
        "parallel_pmap": eval_pmap(),
    }


eval_hyperbolic_knn = eval_hyperbolic
eval_stein_shrinkage = eval_stein
eval_wls_calibrator = eval_calibrator
eval_manifold_embeddings = eval_embeddings
eval_parallel_pmap = eval_pmap
run_all_evals = run_evals
