(* admin/ui/py_repl.ml
   Interactive in-browser Python REPL & inline runnable snippet controller.
   Talks to self-hosted stripped Pyodide via dedicated Web Worker (assets/repl_worker.js).
   Zero innerHTML / outerHTML / eval_string; pure DOM / createElementNS construction.
*)

[@@@warning "-32-27-33"]

open Js_of_ocaml
open Dom_html

let js = Js.string
let to_s = Js.to_string
let svg_ns = "http://www.w3.org/2000/svg"

type preset = {
  id : string;
  label : string;
  short_label : string;
  description : string;
  code : string;
}

let presets : preset list =
  [
    {
      id = "sparc_pf";
      label = "SPARC Tokamak: 8-PF Equilibrium";
      short_label = "SPARC 8-PF";
      description =
        "Optimize 8 SPARC shaping PF coil currents via AGM Biot-Savart flux + X-point null (ENN vs Zero vs LHD).";
      code =
        "import json, os\n"
        ^ "import numpy as np\n"
        ^ "import baux\n"
        ^ "from baux.config import CandidateConfig\n\n"
        ^ "env = next(p for p in ['/home/pyodide/baux_env.json', 'baux_env.json', 'assets/baux_env.json', 'forest/output/assets/baux_env.json'] if os.path.exists(p))\n"
        ^ "sparc = json.load(open(env))['fuse']['machines']['sparc']\n\n"
        ^ "coils = sparc['coils']\n"
        ^ "nom_I = np.array([c['current_ma'] for c in coils[14:22]], dtype=float)\n"
        ^ "lcfs = np.array(sparc['lcfs'], dtype=float)\n"
        ^ "axR, axZ = float(sparc['axisR']), float(sparc['axisZ'])\n"
        ^ "rx, zx, Ip = float(sparc['rx']), float(sparc['zx']), float(sparc['ipNominal'])\n\n"
        ^ "def agm_KE(k2):\n"
        ^ "    k2 = np.clip(k2, 0.0, 1.0 - 1e-10)\n"
        ^ "    a, b, c2, s, p = np.ones_like(k2), np.sqrt(1.0 - k2), k2.copy(), np.zeros_like(k2), 1.0\n"
        ^ "    for _ in range(6):\n"
        ^ "        s, p = s + p * c2, p * 2.0\n"
        ^ "        an, bn = 0.5 * (a + b), np.sqrt(a * b)\n"
        ^ "        c2, a, b = 0.25 * (a - b) ** 2, an, bn\n"
        ^ "    K = 0.5 * np.pi / a\n"
        ^ "    return K, K * (1.0 - 0.5 * s)\n\n"
        ^ "def fpsi(R, Z, Rc, Zc):\n"
        ^ "    k2 = (4.0 * Rc * R) / ((Rc + R) ** 2 + (Z - Zc) ** 2 + 1e-8)\n"
        ^ "    K, E = agm_KE(k2)\n"
        ^ "    k = np.sqrt(np.maximum(k2, 1e-12))\n"
        ^ "    return 0.4 * np.sqrt(Rc * R) * ((2.0 / k - k) * K - (2.0 / k) * E)\n\n"
        ^ "ptsR = np.concatenate([[axR], lcfs[:, 0], [rx + 2e-3, rx - 2e-3, rx, rx]])\n"
        ^ "ptsZ = np.concatenate([[axZ], lcfs[:, 1], [zx, zx, zx + 2e-3, zx - 2e-3]])\n"
        ^ "G = np.zeros((len(ptsR), 8))\n"
        ^ "psi_fix = Ip * fpsi(ptsR, ptsZ, axR, axZ)\n"
        ^ "for ic, c in enumerate(coils):\n"
        ^ "    g = fpsi(ptsR, ptsZ, float(c['r']), float(c['z']))\n"
        ^ "    if 14 <= ic < 22:\n"
        ^ "        G[:, ic - 14] = g\n"
        ^ "    else:\n"
        ^ "        psi_fix += float(c['current_ma']) * g\n\n"
        ^ "def sparc_score(xm):\n"
        ^ "    I = np.asarray(xm, dtype=float)\n"
        ^ "    psi = psi_fix[None, :] + I @ G.T\n"
        ^ "    lc = psi[:, 1:1 + len(lcfs)]\n"
        ^ "    sh = np.mean((lc - np.mean(lc, axis=1, keepdims=True)) ** 2, axis=1)\n"
        ^ "    dR = (psi[:, -4] - psi[:, -3]) / 4e-3\n"
        ^ "    dZ = (psi[:, -2] - psi[:, -1]) / 4e-3\n"
        ^ "    return -(np.sqrt(sh + 1e-12) + 0.35 * np.sqrt(dR ** 2 + dZ ** 2 + 1e-12) + 1e-3 * np.mean(I ** 2, axis=1))[:, None]\n\n"
        ^ "bounds = np.array([[-13.0, 13.0]] * 8)\n"
        ^ "def run_eq(cfg, n_evals=96):\n"
        ^ "    st = baux.init(bounds, cfg, seed=1)\n"
        ^ "    while st.num_observations < n_evals:\n"
        ^ "        st, bat = baux.ask(st, 4)\n"
        ^ "        st, _ = baux.tell(st, bat, sparc_score(bat.points))\n"
        ^ "    return st\n\n"
        ^ "state = run_eq(baux.turbo_enn(num_init=12, candidates=CandidateConfig(num_candidates=300)))\n"
        ^ "st_zero = run_eq(baux.turbo_zero(num_init=12, num_candidates=300))\n"
        ^ "st_lhd = run_eq(baux.lhd_only(num_init=96))\n"
        ^ "nom_s = float(sparc_score(nom_I[None, :])[0, 0])\n"
        ^ "print(f'SPARC PF equilibrium (96 evals): ENN={state.best_y:.4f} | Zero={st_zero.best_y:.4f} | LHD={st_lhd.best_y:.4f} | FUSE-nom={nom_s:.4f}')\n"
        ^ "print('ENN coil currents [MA]:', np.round(state.best_x, 3))\n\n"
        ^ "nr, nz = 28, 28\n"
        ^ "rr, zz = np.meshgrid(np.linspace(0.45, 4.25, nr), np.linspace(-2.45, 2.45, nz))\n"
        ^ "def grid_psi(I_pf):\n"
        ^ "    val = Ip * fpsi(rr, zz, axR, axZ)\n"
        ^ "    for ic, c in enumerate(coils):\n"
        ^ "        cur = float(I_pf[ic - 14]) if 14 <= ic < 22 else float(c['current_ma'])\n"
        ^ "        val = val + cur * fpsi(rr, zz, float(c['r']), float(c['z']))\n"
        ^ "    return val\n\n"
        ^ "init_I = state.x_obs[0]\n"
        ^ "def lcfs_mean(I_pf):\n"
        ^ "    return float(np.mean((psi_fix[None, :] + I_pf[None, :] @ G.T)[0, 1:1 + len(lcfs)]))\n\n"
        ^ "coil_viz = [[float(c['r']), float(c['z']), float(c['w']), float(c['h']),\n"
        ^ "             float(state.best_x[ic - 14]) / 13.0 if 14 <= ic < 22 else float(c['current_ma']) / 13.0]\n"
        ^ "            for ic, c in enumerate(coils)]\n"
        ^ "fig = {\n"
        ^ "    'kind': 'tokamak',\n"
        ^ "    'title': 'SPARC 8-PF Shaping Separatrix vs FUSE Target LCFS',\n"
        ^ "    'subtitle': 'AGM Biot-Savart flux contours matched to TEQUILA equilibrium & lower X-point null',\n"
        ^ "    'r_range': [0.45, 4.25], 'z_range': [-2.45, 2.45],\n"
        ^ "    'first_wall': sparc['first_wall'], 'lcfs': sparc['lcfs'], 'x_point': [rx, zx],\n"
        ^ "    'coils': coil_viz,\n"
        ^ "    'flux_grid': {\n"
        ^ "        'nr': nr, 'nz': nz,\n"
        ^ "        'psi_init': grid_psi(init_I).ravel().tolist(),\n"
        ^ "        'psi_opt': grid_psi(state.best_x).ravel().tolist(),\n"
        ^ "        'psi_init_target': lcfs_mean(init_I),\n"
        ^ "        'psi_opt_target': lcfs_mean(state.best_x),\n"
        ^ "    },\n"
        ^ "    'curves': {\n"
        ^ "        'title': 'PF Shaping Objective Trajectory (96 evals)',\n"
        ^ "        'y_label': 'Equilibrium Score', 'x_label': 'Evaluations',\n"
        ^ "        'series': [\n"
        ^ "            {'label': 'TuRBO-ENN', 'y': np.maximum.accumulate(state.y_obs[:, 0]).tolist()},\n"
        ^ "            {'label': 'TuRBO-Zero', 'y': np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist()},\n"
        ^ "            {'label': 'LHD-only', 'y': np.maximum.accumulate(st_lhd.y_obs[:, 0]).tolist(), 'dashed': True},\n"
        ^ "        ],\n"
        ^ "    },\n"
        ^ "}\n";
    };
    {
      id = "kh_pde";
      label = "2D Kelvin-Helmholtz Shear PDE";
      short_label = "KH Shear PDE";
      description =
        "Pseudo-spectral 2D Navier-Stokes double shear layer control suppressing vortex rollup enstrophy (ENN vs Zero).";
      code =
        "import time\n"
        ^ "import numpy as np\n"
        ^ "import baux\n"
        ^ "from baux.config import CandidateConfig, AcquisitionKind, ENNSurrogate, SurrogateFit\n\n"
        ^ "N = 24\n"
        ^ "k1 = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi\n"
        ^ "kx, kz = np.meshgrid(k1, k1)\n"
        ^ "k2 = kx ** 2 + kz ** 2\n"
        ^ "k2_inv = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 > 0)\n"
        ^ "dealias = (np.abs(kx) < (2.0 / 3.0) * N * np.pi) & (np.abs(kz) < (2.0 / 3.0) * N * np.pi)\n"
        ^ "grid1d = np.linspace(0.0, 1.0, N, endpoint=False)\n"
        ^ "X, Z = np.meshgrid(grid1d, grid1d)\n\n"
        ^ "w_bar = 4.0 / np.cosh((Z - 0.25) / 0.05) ** 2 - 4.0 / np.cosh((Z - 0.75) / 0.05) ** 2\n"
        ^ "w0 = w_bar + 1.2 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Z)\n"
        ^ "modes = [(1, 0), (1, 1), (1, -1), (2, 0), (2, 1)]\n"
        ^ "basis = np.stack([f(2.0 * np.pi * (mx * X + mz * Z))\n"
        ^ "                  for mx, mz in modes for f in (np.sin, np.cos)], axis=0)\n"
        ^ "drive = 5.0 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Z) + 3.0 * np.cos(4.0 * np.pi * X) * np.sin(2.0 * np.pi * Z)\n\n"
        ^ "def sim_kh(ctrl, nsteps=18, dt=0.025):\n"
        ^ "    force = np.tensordot(ctrl, basis, axes=(0, 0)) + drive\n"
        ^ "    w = w0.copy()\n"
        ^ "    for _ in range(nsteps):\n"
        ^ "        wh = np.fft.fft2(w)\n"
        ^ "        psih = wh * k2_inv\n"
        ^ "        u = np.fft.ifft2(1j * kz * psih).real\n"
        ^ "        v = np.fft.ifft2(-1j * kx * psih).real\n"
        ^ "        wx = np.fft.ifft2(1j * kx * wh * dealias).real\n"
        ^ "        wz = np.fft.ifft2(1j * kz * wh * dealias).real\n"
        ^ "        rhs = -(u * wx + v * wz) + force\n"
        ^ "        w = np.fft.ifft2((wh + dt * np.fft.fft2(rhs) * dealias) / (1.0 + 3e-3 * k2 * dt)).real\n"
        ^ "    return w, -(float(np.mean((w - w_bar) ** 2)) + 0.002 * float(np.sum(ctrl ** 2)))\n\n"
        ^ "w_unc, y_unc = sim_kh(np.zeros(10))\n"
        ^ "bounds = np.array([[-6.0, 6.0]] * 10)\n"
        ^ "def run_kh(cfg):\n"
        ^ "    st = baux.init(bounds, cfg, seed=3)\n"
        ^ "    while st.num_observations < 72:\n"
        ^ "        st, bat = baux.ask(st, 4)\n"
        ^ "        st, _ = baux.tell(st, bat, np.array([[sim_kh(r)[1]] for r in bat.points]))\n"
        ^ "    return st\n\n"
        ^ "t0 = time.perf_counter()\n"
        ^ "state = run_kh(baux.turbo_enn(num_init=12, enn=ENNSurrogate(fit=SurrogateFit(num_samples=150)),\n"
        ^ "                              acquisition=AcquisitionKind.UCB, candidates=CandidateConfig(num_candidates=300)))\n"
        ^ "t_enn = (time.perf_counter() - t0) * 1e3\n"
        ^ "st_zero = run_kh(baux.turbo_zero(num_init=12, num_candidates=300))\n"
        ^ "w_opt, _ = sim_kh(state.best_x)\n"
        ^ "vmax = float(max(np.max(np.abs(w_unc)), np.max(np.abs(w_opt))))\n\n"
        ^ "print(f'2D KH shear (72 evals): uncontrolled={y_unc:.3f} -> ENN={state.best_y:.3f} ({t_enn:.0f}ms) vs Zero={st_zero.best_y:.3f}')\n"
        ^ "fig = {\n"
        ^ "    'kind': 'field2d',\n"
        ^ "    'title': '2D Incompressible Shear Vorticity w(x, z)',\n"
        ^ "    'subtitle': '10-mode spectral actuator control suppressing Kelvin-Helmholtz vortex rollup',\n"
        ^ "    'panels': [\n"
        ^ "        {'label': 'Uncontrolled blowup', 'nx': N, 'nz': N, 'vals': w_unc.ravel().tolist(), 'vmin': -vmax, 'vmax': vmax},\n"
        ^ "        {'label': 'TuRBO-ENN damped', 'nx': N, 'nz': N, 'vals': w_opt.ravel().tolist(), 'vmin': -vmax, 'vmax': vmax},\n"
        ^ "    ],\n"
        ^ "    'curves': {\n"
        ^ "        'title': 'Perturbation Enstrophy Suppression (72 evals)',\n"
        ^ "        'y_label': 'Negative Cost', 'x_label': 'Evaluations',\n"
        ^ "        'series': [\n"
        ^ "            {'label': 'TuRBO-ENN (UCB)', 'y': np.maximum.accumulate(state.y_obs[:, 0]).tolist()},\n"
        ^ "            {'label': 'TuRBO-Zero', 'y': np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist()},\n"
        ^ "            {'label': 'Uncontrolled', 'y': [y_unc] * state.num_observations, 'dashed': True},\n"
        ^ "        ],\n"
        ^ "    },\n"
        ^ "}\n";
    };
    {
      id = "oxcaml_24d";
      label = "OxCaml 24D Levy: ENN vs GP vs Zero";
      short_label = "OxCaml 24D";
      description =
        "Head-to-head 24D multimodal scaling comparison on equal 360-candidate budget: TuRBO-ENN vs Exact GP vs Zero.";
      code =
        "import time\n"
        ^ "import numpy as np\n"
        ^ "import baux\n"
        ^ "from baux.config import CandidateConfig, AcquisitionKind, ENNSurrogate, SurrogateFit\n"
        ^ "from baux.benchmarks import oxcaml\n\n"
        ^ "desc = oxcaml.describe('levy', 24)\n"
        ^ "bounds = np.array(desc['bounds'], dtype=float)\n\n"
        ^ "def bench(cfg):\n"
        ^ "    t0 = time.perf_counter()\n"
        ^ "    st = baux.init(bounds, cfg, seed=0)\n"
        ^ "    while st.num_observations < 128:\n"
        ^ "        st, bat = baux.ask(st, 4)\n"
        ^ "        st, _ = baux.tell(st, bat, -oxcaml.eval_batch('levy', bat.points).reshape(-1, 1))\n"
        ^ "    return st, (time.perf_counter() - t0) * 1e3\n\n"
        ^ "state, ms_enn = bench(baux.turbo_enn(num_init=16, enn=ENNSurrogate(fit=SurrogateFit(num_samples=150)),\n"
        ^ "                                     acquisition=AcquisitionKind.UCB, candidates=CandidateConfig(num_candidates=360)))\n"
        ^ "st_gp, ms_gp = bench(baux.turbo_one(num_init=16, num_candidates=360))\n"
        ^ "st_zero, ms_zero = bench(baux.turbo_zero(num_init=16, num_candidates=360))\n\n"
        ^ "print(f'OxCaml catalogue size: {len(oxcaml.catalogue())} functions')\n"
        ^ "print(f'24D Levy @ 128 evals (max -f, opt=0.0):')\n"
        ^ "print(f'  TuRBO-ENN (UCB) : best={state.best_y:8.2f}  wall={ms_enn:5.0f} ms')\n"
        ^ "print(f'  TuRBO-1 ExactGP : best={st_gp.best_y:8.2f}  wall={ms_gp:5.0f} ms')\n"
        ^ "print(f'  TuRBO-Zero      : best={st_zero.best_y:8.2f}  wall={ms_zero:5.0f} ms')\n\n"
        ^ "fig = {\n"
        ^ "    'kind': 'curves',\n"
        ^ "    'title': '24D Levy Scaling (Equal 360-Candidate Budget)',\n"
        ^ "    'subtitle': 'Higher is better (-Levy loss; 0.0 is global optimum)',\n"
        ^ "    'x_label': 'Evaluations', 'y_label': 'Best Objective',\n"
        ^ "    'series': [\n"
        ^ "        {'label': 'TuRBO-ENN', 'y': np.maximum.accumulate(state.y_obs[:, 0]).tolist()},\n"
        ^ "        {'label': 'TuRBO-1 GP', 'y': np.maximum.accumulate(st_gp.y_obs[:, 0]).tolist()},\n"
        ^ "        {'label': 'TuRBO-Zero', 'y': np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist(), 'dashed': True},\n"
        ^ "    ],\n"
        ^ "}\n";
    };
    {
      id = "metric_cal";
      label = "Non-Euclidean ENN & McShane-Whitney";
      short_label = "Metric ENN";
      description =
        "Compare euclidean, conformal, and hyperbolic ENN uncertainty calibration against exact McShane-Whitney Lipschitz bounds.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "rng = np.random.default_rng(19)\n"
        ^ "x_tr = np.concatenate([\n"
        ^ "    np.linspace(-1.6, 0.25, 22),\n"
        ^ "    np.linspace(1.35, 2.2, 16),\n"
        ^ "    np.array([0.82]),\n"
        ^ "])[:, None]\n"
        ^ "f_true = lambda x: np.sin(4.2 * x) * np.exp(-0.22 * x ** 2) + 0.35 * x\n"
        ^ "y_tr = f_true(x_tr) + 0.03 * rng.normal(size=x_tr.shape)\n"
        ^ "y_var = np.full_like(y_tr, 0.03 ** 2)\n"
        ^ "xg = np.linspace(-1.6, 2.2, 81)[:, None]\n"
        ^ "yg = f_true(xg).ravel()\n"
        ^ "kfit = baux.key(29)\n"
        ^ "flags = baux.enn.PosteriorFlags(observation_noise=False)\n\n"
        ^ "posts = {}\n"
        ^ "for geom in ['euclidean', 'conformal', 'hyperbolic']:\n"
        ^ "    es = baux.with_geometry(baux.enn_init(x_tr, y_tr, y_var, scale_x=True), geom, local_k=4)\n"
        ^ "    pr = baux.enn_fit(es, k=6, infer_aleatoric=False, key=kfit)\n"
        ^ "    po = baux.enn_posterior(es, xg, pr, flags=flags)\n"
        ^ "    rmse = float(np.sqrt(np.mean((po.mu.ravel() - yg) ** 2)))\n"
        ^ "    cov95 = float(np.mean(np.abs(yg - po.mu.ravel()) <= 2.0 * po.se.ravel()))\n"
        ^ "    posts[geom] = (po.mu.ravel(), po.se.ravel())\n"
        ^ "    print(f'{geom:10s} | RMSE={rmse:.4f} | empirical 95% cov={100.0 * cov95:5.1f}%')\n\n"
        ^ "dx = np.abs(x_tr[:, 0:1] - x_tr[:, 0:1].T)\n"
        ^ "dy = np.abs(y_tr[:, 0:1] - y_tr[:, 0:1].T)\n"
        ^ "L_est = float(np.max(np.divide(dy, dx, out=np.zeros_like(dy), where=dx > 0.08)))\n"
        ^ "dxg = np.abs(xg[:, 0:1] - x_tr[:, 0].T)\n"
        ^ "mw_hi = np.min(y_tr[:, 0][None, :] + L_est * dxg, axis=1)\n"
        ^ "mw_lo = np.max(y_tr[:, 0][None, :] - L_est * dxg, axis=1)\n"
        ^ "print(f'McShane-Whitney L_est={L_est:.2f} | envelope cov={100.0 * np.mean((yg >= mw_lo) & (yg <= mw_hi)):.1f}%')\n\n"
        ^ "mu_c, se_c = posts['conformal']\n"
        ^ "mu_e, _ = posts['euclidean']\n"
        ^ "fig = {\n"
        ^ "    'kind': 'band1d',\n"
        ^ "    'title': 'Conformal ENN Posterior vs McShane-Whitney Envelope',\n"
        ^ "    'subtitle': 'Sparse sampling gap in [0.25, 1.35] tests epistemic expansion and Lipschitz bounds',\n"
        ^ "    'x_grid': xg.ravel().tolist(),\n"
        ^ "    'y_true': yg.tolist(),\n"
        ^ "    'mu': mu_c.tolist(),\n"
        ^ "    'lo_2se': (mu_c - 2.0 * se_c).tolist(),\n"
        ^ "    'hi_2se': (mu_c + 2.0 * se_c).tolist(),\n"
        ^ "    'mw_lo': mw_lo.tolist(), 'mw_hi': mw_hi.tolist(),\n"
        ^ "    'train_x': x_tr.ravel().tolist(), 'train_y': y_tr.ravel().tolist(),\n"
        ^ "    'curves': {\n"
        ^ "        'title': 'Pointwise Uncertainty Width (2 * se) Across Spatial Gap',\n"
        ^ "        'x_label': 'Domain x', 'y_label': '2 * se(x)',\n"
        ^ "        'series': [\n"
        ^ "            {'label': 'Conformal 2*se', 'x': xg.ravel().tolist(), 'y': (2.0 * se_c).tolist()},\n"
        ^ "            {'label': 'Euclidean |err|', 'x': xg.ravel().tolist(), 'y': np.abs(mu_e - yg).tolist(), 'dashed': True},\n"
        ^ "        ],\n"
        ^ "    },\n"
        ^ "}\n";
    };
    {
      id = "quant_deadzone";
      label = "FP16 Quantization Deadzone on SO(4)";
      short_label = "FP16 SO(4)";
      description =
        "Observe IEEE float16 rounding deadzones stalling SO(4) Cayley Procrustes descent near bias=32.0 vs float32.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "v_star = np.array([0.08, -0.06, 0.09, -0.05, 0.07, -0.04])\n"
        ^ "pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]\n"
        ^ "def cayley(v):\n"
        ^ "    A = np.zeros((4, 4))\n"
        ^ "    for k, (i, j) in enumerate(pairs):\n"
        ^ "        A[i, j], A[j, i] = v[k], -v[k]\n"
        ^ "    I = np.eye(4)\n"
        ^ "    return np.linalg.solve(I + A, I - A)\n\n"
        ^ "R_star = cayley(v_star)\n"
        ^ "bounds = np.array([[-0.35, 0.35]] * 6)\n"
        ^ "def run_so4(quant):\n"
        ^ "    cfg = (baux.lhd(8) | baux.enn() | baux.ucb()\n"
        ^ "           | baux.regions(1, success_tolerance=2, failure_tolerance=2,\n"
        ^ "                          length=baux.TrustRegionBounds(length_init=0.14, length_min=1e-4, length_max=0.4)))\n"
        ^ "    st = baux.init(bounds, cfg, seed=21)\n"
        ^ "    while st.num_observations < 96:\n"
        ^ "        st, bat = baux.ask(st, 4)\n"
        ^ "        ys = []\n"
        ^ "        for row in bat.points:\n"
        ^ "            w = np.float16(32.0 + row).astype(np.float64) - 32.0 if quant else row\n"
        ^ "            ys.append([-float(np.sum((cayley(w) - R_star) ** 2))])\n"
        ^ "        st, _ = baux.tell(st, bat, np.array(ys))\n"
        ^ "    return st\n\n"
        ^ "state = run_so4(False)\n"
        ^ "st_fp16 = run_so4(True)\n"
        ^ "print(f'SO(4) Procrustes @ bias=32.0 (ULP ~ 0.03125 in float16):')\n"
        ^ "print(f'  FP32 weights : best_err={-state.best_y:.3e} | tr_length={state.tr_length:.4f}')\n"
        ^ "print(f'  FP16 weights : best_err={-st_fp16.best_y:.3e} | tr_length={st_fp16.tr_length:.4f}')\n\n"
        ^ "fig = {\n"
        ^ "    'kind': 'curves',\n"
        ^ "    'title': 'SO(4) Procrustes Frobenius Error (log scale)',\n"
        ^ "    'subtitle': 'float16 storage around bias=32 hits rounding floor while float32 descends smoothly',\n"
        ^ "    'x_label': 'Evaluations', 'y_label': 'Frobenius Err', 'log_y': True,\n"
        ^ "    'series': [\n"
        ^ "        {'label': 'FP32 weights', 'y': (-np.maximum.accumulate(state.y_obs[:, 0])).tolist()},\n"
        ^ "        {'label': 'FP16 @ bias=32', 'y': (-np.maximum.accumulate(st_fp16.y_obs[:, 0])).tolist()},\n"
        ^ "    ],\n"
        ^ "}\n";
    };
    {
      id = "stiefel_fir";
      label = "Mixed Manifold + Stein Shrinkage";
      short_label = "Stiefel FIR";
      description =
        "Synthesize Simplex + Periodic + Log + Categorical FIR filter under observation noise comparing Stein shrinkage vs unshrunk ENN.";
      code =
        "import math\n"
        ^ "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "spec = baux.pytree_bounds({\n"
        ^ "    'weights': baux.Simplex(4),\n"
        ^ "    'phase': baux.Periodic(0.0, 2.0 * np.pi),\n"
        ^ "    'decay': baux.Log(1e-3, 1.0),\n"
        ^ "    'window': baux.Categorical(['hann', 'kaiser', 'blackman']),\n"
        ^ "})\n"
        ^ "w_star = np.array([0.52, 0.28, 0.14, 0.06])\n"
        ^ "def clean_fir(p):\n"
        ^ "    w = np.asarray(p['weights'])\n"
        ^ "    win = {'hann': 0.08, 'kaiser': 0.14, 'blackman': 0.05}[p['window']]\n"
        ^ "    return -(float(np.sum((w - w_star) ** 2)) + 0.08 * (1.0 - math.cos(float(p['phase']) - 1.1))\n"
        ^ "             + 0.03 * (math.log10(float(p['decay'])) + 1.5) ** 2 - win)\n\n"
        ^ "def run_fir(cfg, seed_noise=13):\n"
        ^ "    rng = np.random.default_rng(seed_noise)\n"
        ^ "    st = baux.init(spec, cfg, seed=8)\n"
        ^ "    clean_trace = []\n"
        ^ "    while st.num_observations < 44:\n"
        ^ "        st, bat = baux.ask(st, 4)\n"
        ^ "        ys = np.array([[clean_fir(p) + 0.04 * rng.normal()] for p in bat.params])\n"
        ^ "        st, _ = baux.tell(st, bat, ys)\n"
        ^ "        clean_trace.append(clean_fir(st.best_params))\n"
        ^ "    return st, clean_trace\n\n"
        ^ "st_unshrunk, tr_un = run_fir(baux.turbo_enn(num_init=8))\n"
        ^ "frame = np.eye(st_unshrunk.dim, 2)\n"
        ^ "state, tr_stein = run_fir(baux.lhd(8) | baux.enn() | baux.stein(frame=frame, rank=2) | baux.ucb() | baux.tr_bounds())\n\n"
        ^ "print(f'Embedded dim={state.dim} (max clean score = 0.14000)')\n"
        ^ "print(f'Unshrunk ENN : noiseless={clean_fir(st_unshrunk.best_params):+.5f}')\n"
        ^ "print(f'Stein(rank=2): noiseless={clean_fir(state.best_params):+.5f}')\n"
        ^ "print('Stein best_params:', state.best_params)\n\n"
        ^ "fig = {\n"
        ^ "    'kind': 'curves',\n"
        ^ "    'title': 'Noiseless FIR Filter Quality at Selected Incumbent',\n"
        ^ "    'subtitle': 'Stein subspace shrinkage suppresses false spikes from 0.04 observation noise',\n"
        ^ "    'x_label': 'Batch Step', 'y_label': 'Noiseless Score',\n"
        ^ "    'series': [\n"
        ^ "        {'label': 'Stein(frame, rank=2)', 'y': tr_stein},\n"
        ^ "        {'label': 'Unshrunk ENN', 'y': tr_un, 'dashed': True},\n"
        ^ "    ],\n"
        ^ "}\n";
    };
    {
      id = "async_hartmann6";
      label = "Async Hartmann-6D + Freeze/Digest";
      short_label = "Async H6";
      description =
        "Heterogeneous worker queue cancelling slow stragglers, out-of-order complete(), and mid-run BLAKE3 freeze/thaw.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "c_h = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]], dtype=float)\n"
        ^ "a_h = np.array([1.0, 1.2, 3.0, 3.2])\n"
        ^ "p_h = 1e-4 * np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])\n"
        ^ "def hart6(x):\n"
        ^ "    ex = np.sum(c_h[None, :, :] * (x[:, None, :] - p_h[None, :, :]) ** 2, axis=-1)\n"
        ^ "    return np.sum(a_h[None, :] * np.exp(-ex), axis=-1)[:, None]\n\n"
        ^ "bounds = np.array([[0.0, 1.0]] * 6)\n"
        ^ "cfg = baux.lhd(12) | baux.enn(num_samples=150) | baux.ucb() | baux.raasp(240) | baux.regions(2)\n"
        ^ "blank = baux.init(bounds, cfg, seed=31)\n"
        ^ "st = blank\n"
        ^ "st, b_fast = baux.ask(st, 3)\n"
        ^ "st, b_slow = baux.ask(st, 3)\n"
        ^ "st = baux.cancel(st, b_slow.ids)\n"
        ^ "st = baux.complete(st, b_fast.ids, hart6(b_fast.points))\n\n"
        ^ "blob = baux.freeze(st)\n"
        ^ "d_pre = baux.digest(st)\n"
        ^ "state = baux.thaw(blank, blob)\n"
        ^ "assert baux.digest(state) == d_pre\n\n"
        ^ "while state.num_observations < 84:\n"
        ^ "    state, bf = baux.ask(state, 4)\n"
        ^ "    state, bs = baux.ask(state, 2)\n"
        ^ "    state = baux.cancel(state, bs.ids)\n"
        ^ "    state = baux.complete(state, bf.ids, hart6(bf.points))\n\n"
        ^ "st_zero = baux.init(bounds, baux.turbo_zero(num_init=12, num_candidates=240), seed=31)\n"
        ^ "while st_zero.num_observations < 84:\n"
        ^ "    st_zero, bz = baux.ask(st_zero, 4)\n"
        ^ "    st_zero, _ = baux.tell(st_zero, bz, hart6(bz.points))\n\n"
        ^ "print(f'Checkpoint bytes={len(blob)} BLAKE3={d_pre[:16]}... pending_end={len(state.pending_ids or [])}')\n"
        ^ "print(f'Hartmann-6D (opt=3.3224): Async ENN={state.best_y:.4f} vs Zero={st_zero.best_y:.4f}')\n\n"
        ^ "fig = {\n"
        ^ "    'kind': 'curves',\n"
        ^ "    'title': 'Hartmann-6D Async Straggler Cancellation vs Zero',\n"
        ^ "    'subtitle': '2-region async complete() + cancel() with freeze/thaw BLAKE3 continuity',\n"
        ^ "    'x_label': 'Completed Evals', 'y_label': 'Best Hartmann-6D',\n"
        ^ "    'series': [\n"
        ^ "        {'label': 'Async TuRBO-ENN', 'y': np.maximum.accumulate(state.y_obs[:, 0]).tolist()},\n"
        ^ "        {'label': 'TuRBO-Zero', 'y': np.maximum.accumulate(st_zero.y_obs[:, 0]).tolist(), 'dashed': True},\n"
        ^ "    ],\n"
        ^ "}\n";
    };
  ]

let find_preset id =
  List.find_opt (fun p -> p.id = id) presets

let trim_s s = String.trim s

let contains_sub hay needle =
  let hl = String.length hay in
  let nl = String.length needle in
  if nl = 0 then true
  else if nl > hl then false
  else
    let rec loop i =
      if i > hl - nl then false
      else if String.sub hay i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let get_base_url () =
  match Js.Opt.to_option (document##.documentElement##getAttribute (js "data-base-url")) with
  | Some v -> to_s v
  | None -> "./"

let mk ?cls ?txt tag =
  let e = document##createElement (js tag) in
  (match cls with Some c -> e##.className := js c | None -> ());
  (match txt with Some t -> e##.textContent := Js.some (js t) | None -> ());
  e

let set_text (e : #Dom.node Js.t) (s : string) =
  e##.textContent := Js.some (js s)

let svg_el tag attrs =
  let e : element Js.t =
    Js.Unsafe.meth_call document "createElementNS"
      [| Js.Unsafe.inject (js svg_ns); Js.Unsafe.inject (js tag) |]
  in
  List.iter (fun (k, v) -> e##setAttribute (js k) (js v)) attrs;
  e

let clear_children (parent : #Dom.node Js.t) =
  let rec loop () =
    match Js.Opt.to_option parent##.firstChild with
    | None -> ()
    | Some ch ->
        ignore (parent##removeChild ch);
        loop ()
  in
  loop ()

(* ---------------------------------------------------------------------------
   Singleton lazy Web Worker kernel controller
   --------------------------------------------------------------------------- *)

type kernel_state = Idle | Booting of string | Ready | Running | Interrupted of string

let worker_ref : Js.Unsafe.any option ref = ref None
let kernel_state_ref : kernel_state ref = ref Idle
let status_listeners : (kernel_state -> unit) list ref = ref []
let pending_reqs : (int, Js.Unsafe.any -> unit) Hashtbl.t = Hashtbl.create 16
let next_req_id = ref 1

let notify_status st =
  kernel_state_ref := st;
  List.iter (fun fn -> fn st) !status_listeners

let add_status_listener fn =
  status_listeners := fn :: !status_listeners;
  fn !kernel_state_ref

let interrupt_kernel () =
  (match !worker_ref with
  | Some w ->
      ignore (Js.Unsafe.meth_call w "terminate" [||]);
      worker_ref := None
  | None -> ());
  let callbacks = Hashtbl.fold (fun _ v acc -> v :: acc) pending_reqs [] in
  Hashtbl.clear pending_reqs;
  List.iter
    (fun cb ->
      let err_obj =
        Js.Unsafe.obj
          [|
            ("ok", Js.Unsafe.inject Js._false);
            ("stdout", Js.Unsafe.inject (js ""));
            ("stderr", Js.Unsafe.inject (js "Kernel interrupted by user. Click Run to start a fresh kernel."));
            ("incomplete", Js.Unsafe.inject Js._false);
          |]
      in
      cb err_obj)
    callbacks;
  notify_status (Interrupted "Kernel interrupted")

let ensure_worker () =
  match !worker_ref with
  | Some w -> w
  | None ->
      let worker_ctor = Js.Unsafe.get Dom_html.window (js "Worker") in
      let url = get_base_url () ^ "assets/repl_worker.js" in
      let w = Js.Unsafe.new_obj worker_ctor [| Js.Unsafe.inject (js url) |] in
      notify_status (Booting "Starting Web Worker...");
      let onmsg =
        Dom.handler (fun ev ->
            let data = Js.Unsafe.get ev (js "data") in
            let typ =
              if Js.Optdef.test (Js.Unsafe.get data (js "type")) then
                to_s (Js.Unsafe.get data (js "type"))
              else ""
            in
            if typ = "status" then (
              let detail = to_s (Js.Unsafe.get data (js "detail")) in
              let stage = to_s (Js.Unsafe.get data (js "stage")) in
              if stage = "ready" then notify_status Ready
              else notify_status (Booting detail))
            else (
              let id_any = Js.Unsafe.get data (js "id") in
              if Js.Opt.test id_any then (
                let id_int = int_of_float (Js.float_of_number id_any) in
                match Hashtbl.find_opt pending_reqs id_int with
                | Some cb ->
                    Hashtbl.remove pending_reqs id_int;
                    if !kernel_state_ref = Running then notify_status Ready;
                    cb data
                | None -> ()));
            Js._true)
      in
      let onerr =
        Dom.handler (fun ev ->
            let msg =
              if Js.Optdef.test (Js.Unsafe.get ev (js "message")) then
                to_s (Js.Unsafe.get ev (js "message"))
              else "Worker error"
            in
            notify_status (Interrupted ("Worker failed: " ^ msg));
            Js._true)
      in
      Js.Unsafe.set w (js "onmessage") onmsg;
      Js.Unsafe.set w (js "onerror") onerr;
      worker_ref := Some w;
      w

let send_worker typ fields cb =
  let w = ensure_worker () in
  let req_id = !next_req_id in
  next_req_id := req_id + 1;
  Hashtbl.add pending_reqs req_id cb;
  if !kernel_state_ref = Ready then notify_status Running;
  let payload =
    Js.Unsafe.obj
      (Array.of_list
         (("type", Js.Unsafe.inject (js typ))
         :: ("id", Js.Unsafe.inject req_id)
         :: fields))
  in
  ignore (Js.Unsafe.meth_call w "postMessage" [| Js.Unsafe.inject payload |])

let warm_kernel () =
  send_worker "init" [] (fun _ -> ())

let status_text = function
  | Idle -> "Idle (loads ~16 MB WASM on first Run)"
  | Booting d -> d
  | Ready -> "Python 3.12 + NumPy + baux ready"
  | Running -> "Executing Python in Web Worker..."
  | Interrupted r -> r

(* ---------------------------------------------------------------------------
   Live OptimizerState Summary & Multi-View Scientific SVG Renderers
   --------------------------------------------------------------------------- *)

let is_defined (v : Js.Unsafe.any) : bool =
  Js.Optdef.test (Obj.magic v : _ Js.optdef) && Js.Opt.test (Obj.magic v : _ Js.opt)

let is_finite (v : float) : bool =
  match classify_float v with
  | FP_nan | FP_infinite -> false
  | _ -> true

let json_num_list (raw : Js.Unsafe.any) =
  if not (is_defined raw) then []
  else
    let arr : Js.Unsafe.any Js.js_array Js.t = Obj.magic raw in
    let len = arr##.length in
    let out = ref [] in
    for i = 0 to len - 1 do
      let item = Js.array_get arr i in
      match Js.Optdef.to_option item with
      | Some v when is_defined v ->
          let fv = Js.float_of_number (Obj.magic v) in
          if is_finite fv then out := fv :: !out
      | _ -> ()
    done;
    List.rev !out

let json_obj_list (raw : Js.Unsafe.any) : Js.Unsafe.any list =
  if not (is_defined raw) then []
  else
    let arr : Js.Unsafe.any Js.js_array Js.t = Obj.magic raw in
    let len = arr##.length in
    let out = ref [] in
    for i = 0 to len - 1 do
      match Js.Optdef.to_option (Js.array_get arr i) with
      | Some v when is_defined v -> out := v :: !out
      | _ -> ()
    done;
    List.rev !out

let json_pair_list (raw : Js.Unsafe.any) : (float * float) list =
  List.filter_map
    (fun row ->
      let xs = json_num_list row in
      match xs with a :: b :: _ -> Some (a, b) | _ -> None)
    (json_obj_list raw)

let opt_str obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (to_s (Obj.magic v)) else None

let opt_int obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (int_of_float (Js.float_of_number (Obj.magic v))) else None

let opt_float obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then
    let fv = Js.float_of_number (Obj.magic v) in
    if is_finite fv then Some fv else None
  else None

let opt_bool obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Js.to_bool (Obj.magic v : bool Js.t) else false

let make_fig_pane title subtitle =
  let card = mk ~cls:"py-fig-pane" "div" in
  let hdr = mk ~cls:"py-fig-hdr" "div" in
  Dom.appendChild hdr (mk ~cls:"py-fig-title" ~txt:title "div");
  if trim_s subtitle <> "" then
    Dom.appendChild hdr (mk ~cls:"py-fig-subtitle" ~txt:subtitle "div");
  let stage = mk ~cls:"py-fig-stage" "div" in
  Dom.appendChild card hdr;
  Dom.appendChild card stage;
  (card, stage)

let append_legend_badge legend swatch_cls label =
  let item = mk ~cls:"py-fig-legend-item" "span" in
  Dom.appendChild item (mk ~cls:("py-fig-swatch " ^ swatch_cls) "span");
  Dom.appendChild item (mk ~cls:"py-fig-legend-label" ~txt:label "span");
  Dom.appendChild legend item

type curve_series = {
  c_label : string;
  c_xs : float list;
  c_ys : float list;
  c_dashed : bool;
}

let render_curves_card ~title ~subtitle ~x_label ~y_label ~log_y (series : curve_series list) =
  let card, stage = make_fig_pane title subtitle in
  let non_empty = List.filter (fun s -> s.c_ys <> [] && s.c_xs <> []) series in
  if non_empty = [] then card
  else
    let all_y_raw = List.concat_map (fun s -> s.c_ys) non_empty in
    let use_log = log_y && List.for_all (fun v -> v > 0.0) all_y_raw in
    let tr_y v = if use_log then log10 (max 1e-12 v) else v in
    let all_x = List.concat_map (fun s -> s.c_xs) non_empty in
    let all_y = List.map tr_y all_y_raw in
    let x_lo_r = List.fold_left min (List.hd all_x) all_x in
    let x_hi_r = List.fold_left max (List.hd all_x) all_x in
    let x_sp = if x_hi_r -. x_lo_r < 1e-9 then 1.0 else x_hi_r -. x_lo_r in
    let y_lo_r = List.fold_left min (List.hd all_y) all_y in
    let y_hi_r = List.fold_left max (List.hd all_y) all_y in
    let y_sp_r = if y_hi_r -. y_lo_r < 1e-9 then 1.0 else y_hi_r -. y_lo_r in
    let y_lo = y_lo_r -. 0.07 *. y_sp_r in
    let y_hi = y_hi_r +. 0.07 *. y_sp_r in
    let y_sp = y_hi -. y_lo in
    let legend = mk ~cls:"py-fig-legend" "div" in
    let xl_lower = String.lowercase_ascii x_label in
    let show_end_val = contains_sub xl_lower "eval" || contains_sub xl_lower "step" in
    List.iteri
      (fun idx s ->
        let slot = idx mod 5 in
        let last_y = List.nth s.c_ys (List.length s.c_ys - 1) in
        let lbl_txt =
          if show_end_val then Printf.sprintf "%s (%.4g)" s.c_label last_y else s.c_label
        in
        let sw_cls =
          Printf.sprintf "py-swatch-%d%s" slot (if s.c_dashed then " is-dashed" else "")
        in
        append_legend_badge legend sw_cls lbl_txt)
      non_empty;
    Dom.appendChild stage legend;
    let w = 460.0 and h = 176.0 in
    let pl = 56.0 and pr = 16.0 and pt = 18.0 and pb = 32.0 in
    let pw = w -. pl -. pr in
    let ph = h -. pt -. pb in
    let x_px x = pl +. ((x -. x_lo_r) /. x_sp) *. pw in
    let y_px y = pt +. (1.0 -. ((tr_y y -. y_lo) /. y_sp)) *. ph in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let yy = pt +. (1.0 -. frac) *. ph in
      let v_t = y_lo +. frac *. y_sp in
      let v_disp = if use_log then 10.0 ** v_t else v_t in
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.1f" pl);
             ("x2", Printf.sprintf "%.1f" (w -. pr));
             ("y1", Printf.sprintf "%.1f" yy);
             ("y2", Printf.sprintf "%.1f" yy);
             ("class", "py-svg-grid");
           ]);
      let lbl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" (pl -. 6.0));
            ("y", Printf.sprintf "%.1f" (yy +. 3.5));
            ("text-anchor", "end");
            ("class", "py-svg-tick");
          ]
      in
      set_text lbl (Printf.sprintf "%.3g" v_disp);
      Dom.appendChild svg lbl
    done;
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let xx = pl +. frac *. pw in
      let xv = x_lo_r +. frac *. x_sp in
      let xtl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" xx);
            ("y", Printf.sprintf "%.1f" (h -. pb +. 12.0));
            ("text-anchor", "middle");
            ("class", "py-svg-tick");
          ]
      in
      set_text xtl (Printf.sprintf "%.3g" xv);
      Dom.appendChild svg xtl
    done;
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" (w -. pr));
           ("y1", Printf.sprintf "%.1f" (h -. pb));
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" pl);
           ("y1", Printf.sprintf "%.1f" pt);
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    let ylbl =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (pl +. 4.0));
          ("y", "12.0");
          ("text-anchor", "start");
          ("class", "py-svg-axis-lbl");
        ]
    in
    set_text ylbl (if use_log then y_label ^ " (log)" else y_label);
    Dom.appendChild svg ylbl;
    List.iteri
      (fun idx s ->
        let slot = idx mod 5 in
        let rec zip_pts xs ys acc =
          match (xs, ys) with
          | x :: xtl, y :: ytl ->
              zip_pts xtl ytl (Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y) :: acc)
          | _ -> List.rev acc
        in
        let pts_list = zip_pts s.c_xs s.c_ys [] in
        let pts_str = String.concat " " pts_list in
        let line_cls =
          Printf.sprintf "py-svg-series py-svg-series-%d%s" slot
            (if s.c_dashed then " is-dashed" else "")
        in
        Dom.appendChild svg
          (svg_el "polyline" [ ("fill", "none"); ("points", pts_str); ("class", line_cls) ]);
        let npts = min (List.length s.c_xs) (List.length s.c_ys) in
        if npts > 0 then
          let lx = x_px (List.nth s.c_xs (npts - 1)) in
          let ly = y_px (List.nth s.c_ys (npts - 1)) in
          Dom.appendChild svg
            (svg_el "circle"
               [
                 ("cx", Printf.sprintf "%.2f" lx);
                 ("cy", Printf.sprintf "%.2f" ly);
                 ("r", "3.4");
                 ("class", Printf.sprintf "py-svg-series-dot py-svg-series-dot-%d" slot);
               ]))
      non_empty;
    let xlbl =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (pl +. 0.5 *. pw));
          ("y", Printf.sprintf "%.1f" (h -. 4.0));
          ("text-anchor", "middle");
          ("class", "py-svg-axis-lbl");
        ]
    in
    set_text xlbl x_label;
    Dom.appendChild svg xlbl;
    Dom.appendChild stage svg;
    card

let curves_from_js (raw : Js.Unsafe.any) : element Js.t option =
  if not (is_defined raw) then None
  else
    let raw_series = json_obj_list (Js.Unsafe.get raw (js "series")) in
    let series =
      List.filter_map
        (fun item ->
          let ys = json_num_list (Js.Unsafe.get item (js "y")) in
          if ys = [] then None
          else
            let xs0 = json_num_list (Js.Unsafe.get item (js "x")) in
            let xs =
              if List.length xs0 = List.length ys then xs0
              else List.mapi (fun i _ -> float_of_int (i + 1)) ys
            in
            Some
              {
                c_label = Option.value (opt_str item "label") ~default:"series";
                c_xs = xs;
                c_ys = ys;
                c_dashed = opt_bool item "dashed";
              })
        raw_series
    in
    if series = [] then None
    else
      Some
        (render_curves_card
           ~title:(Option.value (opt_str raw "title") ~default:"Convergence Comparison")
           ~subtitle:(Option.value (opt_str raw "subtitle") ~default:"")
           ~x_label:(Option.value (opt_str raw "x_label") ~default:"Evaluations")
           ~y_label:(Option.value (opt_str raw "y_label") ~default:"Objective")
           ~log_y:(opt_bool raw "log_y")
           series)

let marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px (vals : float array) (level : float) :
    string =
  let len = Array.length vals in
  if nr < 2 || nz < 2 || len <> nr * nz then ""
  else
    let dr = (r1 -. r0) /. float_of_int (nr - 1) in
    let dz = (z1 -. z0) /. float_of_int (nz - 1) in
    let interp a b va vb =
      let dv = vb -. va in
      if abs_float dv < 1e-12 then 0.5 *. (a +. b)
      else
        let t = max 0.0 (min 1.0 ((level -. va) /. dv)) in
        a +. t *. (b -. a)
    in
    let buf = Buffer.create 512 in
    let emit (ax, ay) (bx, by) =
      Buffer.add_string buf
        (Printf.sprintf "M%.2f %.2fL%.2f %.2f" (x_px ax) (y_px ay) (x_px bx) (y_px by))
    in
    for iz = 0 to nz - 2 do
      let za = z0 +. float_of_int iz *. dz in
      let zb = za +. dz in
      let row0 = iz * nr in
      let row1 = (iz + 1) * nr in
      for ir = 0 to nr - 2 do
        let ra = r0 +. float_of_int ir *. dr in
        let rb = ra +. dr in
        let v00 = vals.(row0 + ir) in
        let v10 = vals.(row0 + ir + 1) in
        let v11 = vals.(row1 + ir + 1) in
        let v01 = vals.(row1 + ir) in
        let b0 = v00 >= level in
        let b1 = v10 >= level in
        let b2 = v11 >= level in
        let b3 = v01 >= level in
        let c_b = if b0 <> b1 then Some (interp ra rb v00 v10, za) else None in
        let c_r = if b1 <> b2 then Some (rb, interp za zb v10 v11) else None in
        let c_t = if b2 <> b3 then Some (interp ra rb v01 v11, zb) else None in
        let c_l = if b3 <> b0 then Some (ra, interp za zb v00 v01) else None in
        match (c_b, c_r, c_t, c_l) with
        | Some pb, Some pr, Some pt, Some pl ->
            let vc = 0.25 *. (v00 +. v10 +. v11 +. v01) in
            if vc >= level then (
              emit pb pr;
              emit pl pt)
            else (
              emit pb pl;
              emit pr pt)
        | _ ->
            let hits = List.filter_map (fun x -> x) [ c_b; c_r; c_t; c_l ] in
            (match hits with p :: q :: _ -> emit p q | _ -> ())
      done
    done;
    Buffer.contents buf

let render_tokamak_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"Tokamak Equilibrium" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, stage = make_fig_pane title subtitle in
  let split = mk ~cls:"py-tokamak-split" "div" in
  let aside = mk ~cls:"py-tokamak-aside" "div" in
  let legend = mk ~cls:"py-fig-legend py-fig-legend-col" "div" in
  append_legend_badge legend "py-swatch-target" "Target LCFS";
  append_legend_badge legend "py-swatch-init" "Initial guess";
  append_legend_badge legend "py-swatch-enn" "ENN separatrix";
  append_legend_badge legend "py-swatch-coil-pos" "PF coil (+)";
  append_legend_badge legend "py-swatch-coil-neg" "PF coil (-)";
  Dom.appendChild aside legend;
  let r_rng = json_num_list (Js.Unsafe.get fig (js "r_range")) in
  let z_rng = json_num_list (Js.Unsafe.get fig (js "z_range")) in
  let r0, r1 = match r_rng with a :: b :: _ when b > a -> (a, b) | _ -> (0.4, 3.0) in
  let z0, z1 = match z_rng with a :: b :: _ when b > a -> (a, b) | _ -> (-1.6, 1.6) in
  let coils_n = List.length (json_obj_list (Js.Unsafe.get fig (js "coils"))) in
  let meta_list = mk ~cls:"py-tokamak-meta" "div" in
  Dom.appendChild meta_list (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "R [%.2f, %.2f] m" r0 r1) "div");
  Dom.appendChild meta_list (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "Z [%.2f, %.2f] m" z0 z1) "div");
  if coils_n > 0 then (
    Dom.appendChild meta_list
      (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "%d PF/CS coils" coils_n) "div");
    Dom.appendChild meta_list
      (mk ~cls:"py-tokamak-meta-row" ~txt:"× X-point null" "div"));
  Dom.appendChild aside meta_list;
  let w = 276.0 and h = 260.0 in
  let pl = 44.0 and pr = 12.0 and pt = 10.0 and pb = 28.0 in
  let pw = w -. pl -. pr and ph = h -. pt -. pb in
  let scale = min (pw /. (r1 -. r0)) (ph /. (z1 -. z0)) in
  let rw = scale *. (r1 -. r0) and zh = scale *. (z1 -. z0) in
  let ox = pl +. 0.5 *. (pw -. rw) in
  let oy = pt +. 0.5 *. (ph -. zh) in
  let x_px r = ox +. (r -. r0) *. scale in
  let y_px z = oy +. (z1 -. z) *. scale in
  let svg =
    svg_el "svg"
      [
        ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
        ("class", "py-traj-svg py-tokamak-svg");
        ("role", "img");
        ("aria-label", title);
      ]
  in
  (* Frame and R/Z ticks *)
  Dom.appendChild svg
    (svg_el "rect"
       [
         ("x", Printf.sprintf "%.1f" ox);
         ("y", Printf.sprintf "%.1f" oy);
         ("width", Printf.sprintf "%.1f" rw);
         ("height", Printf.sprintf "%.1f" zh);
         ("class", "py-svg-frame");
       ]);
  for i = 0 to 3 do
    let f = float_of_int i /. 3.0 in
    let rv = r0 +. f *. (r1 -. r0) in
    let zv = z0 +. f *. (z1 -. z0) in
    let rx = x_px rv in
    let zy = y_px zv in
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" rx);
           ("x2", Printf.sprintf "%.1f" rx);
           ("y1", Printf.sprintf "%.1f" oy);
           ("y2", Printf.sprintf "%.1f" (oy +. zh));
           ("class", "py-svg-grid");
         ]);
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" ox);
           ("x2", Printf.sprintf "%.1f" (ox +. rw));
           ("y1", Printf.sprintf "%.1f" zy);
           ("y2", Printf.sprintf "%.1f" zy);
           ("class", "py-svg-grid");
         ]);
    let tx =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" rx);
          ("y", Printf.sprintf "%.1f" (oy +. zh +. 13.5));
          ("text-anchor", "middle");
          ("class", "py-svg-tick");
        ]
    in
    set_text tx (Printf.sprintf "%.2f" rv);
    Dom.appendChild svg tx;
    let ty =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (ox -. 6.0));
          ("y", Printf.sprintf "%.1f" (zy +. 2.5));
          ("text-anchor", "end");
          ("class", "py-svg-tick");
        ]
    in
    set_text ty (Printf.sprintf "%.2f" zv);
    Dom.appendChild svg ty
  done;
  (* Flux contours *)
  let fg = Js.Unsafe.get fig (js "flux_grid") in
  (if is_defined fg then
     let nr = Option.value (opt_int fg "nr") ~default:0 in
     let nz = Option.value (opt_int fg "nz") ~default:0 in
     let arr_of field = Array.of_list (json_num_list (Js.Unsafe.get fg (js field))) in
     let psi_init = arr_of "psi_init" in
     let psi_opt = arr_of "psi_opt" in
     (match (opt_float fg "psi_init_target", Array.length psi_init = nr * nz) with
     | Some lvl, true ->
         let d = marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_init lvl in
         if d <> "" then
           Dom.appendChild svg (svg_el "path" [ ("d", d); ("class", "py-svg-sep-init") ])
     | _ -> ());
     (match (opt_float fg "psi_opt_target", Array.length psi_opt = nr * nz) with
     | Some lvl, true ->
         let lo_v = Array.fold_left min psi_opt.(0) psi_opt in
         let hi_v = Array.fold_left max psi_opt.(0) psi_opt in
         let span = max 1e-6 (hi_v -. lo_v) in
         List.iter
           (fun df ->
             let d_sub =
               marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_opt
                 (lvl +. df *. span)
             in
             if d_sub <> "" then
               Dom.appendChild svg (svg_el "path" [ ("d", d_sub); ("class", "py-svg-flux-minor") ]))
           [ -0.14; -0.07; 0.07; 0.14 ];
         let d = marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_opt lvl in
         if d <> "" then
           Dom.appendChild svg (svg_el "path" [ ("d", d); ("class", "py-svg-sep-opt") ])
     | _ -> ()));
  (* First wall & Target LCFS *)
  let wall = json_pair_list (Js.Unsafe.get fig (js "first_wall")) in
  if wall <> [] then (
    let pts = String.concat " " (List.map (fun (r, z) -> Printf.sprintf "%.2f,%.2f" (x_px r) (y_px z)) wall) in
    Dom.appendChild svg (svg_el "polygon" [ ("points", pts); ("class", "py-svg-first-wall") ]));
  let lcfs = json_pair_list (Js.Unsafe.get fig (js "lcfs")) in
  if lcfs <> [] then (
    let pts = String.concat " " (List.map (fun (r, z) -> Printf.sprintf "%.2f,%.2f" (x_px r) (y_px z)) lcfs) in
    Dom.appendChild svg (svg_el "polygon" [ ("points", pts); ("class", "py-svg-lcfs-target") ]));
  (* PF Coils *)
  List.iter
    (fun row ->
      match json_num_list row with
      | rc :: zc :: drc :: dzc :: curr :: _ ->
          let w_px = max 4.0 (drc *. scale) in
          let h_px = max 4.0 (dzc *. scale) in
          let cx0 = x_px rc -. 0.5 *. w_px in
          let cy0 = y_px zc -. 0.5 *. h_px in
          let cls = if curr >= 0.0 then "py-svg-coil-pos" else "py-svg-coil-neg" in
          let op = Printf.sprintf "%.2f" (0.28 +. 0.68 *. min 1.0 (abs_float curr)) in
          Dom.appendChild svg
            (svg_el "rect"
               [
                 ("x", Printf.sprintf "%.2f" cx0);
                 ("y", Printf.sprintf "%.2f" cy0);
                 ("width", Printf.sprintf "%.2f" w_px);
                 ("height", Printf.sprintf "%.2f" h_px);
                 ("rx", "1.5");
                 ("fill-opacity", op);
                 ("class", cls);
               ])
      | _ -> ())
    (json_obj_list (Js.Unsafe.get fig (js "coils")));
  (* X-point crosshair *)
  (match json_num_list (Js.Unsafe.get fig (js "x_point")) with
  | rx :: zx :: _ ->
      let xx = x_px rx and yy = y_px zx in
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.2f" (xx -. 5.0));
             ("x2", Printf.sprintf "%.2f" (xx +. 5.0));
             ("y1", Printf.sprintf "%.2f" (yy -. 5.0));
             ("y2", Printf.sprintf "%.2f" (yy +. 5.0));
             ("class", "py-svg-xpt");
           ]);
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.2f" (xx -. 5.0));
             ("x2", Printf.sprintf "%.2f" (xx +. 5.0));
             ("y1", Printf.sprintf "%.2f" (yy +. 5.0));
             ("y2", Printf.sprintf "%.2f" (yy -. 5.0));
             ("class", "py-svg-xpt");
           ])
  | _ -> ());
  let axlbl =
    svg_el "text"
      [
        ("x", Printf.sprintf "%.1f" (ox +. 0.5 *. rw));
        ("y", Printf.sprintf "%.1f" (h -. 3.0));
        ("text-anchor", "middle");
        ("class", "py-svg-axis-lbl");
      ]
  in
  set_text axlbl "Major Radius R [m] vs Vertical Z [m]";
  Dom.appendChild svg axlbl;
  Dom.appendChild split svg;
  Dom.appendChild split aside;
  Dom.appendChild stage split;
  card

let render_field2d_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"2D Field" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, stage = make_fig_pane title subtitle in
  let panels = json_obj_list (Js.Unsafe.get fig (js "panels")) in
  if panels = [] then card
  else
    let n_p = min 2 (List.length panels) in
    let w = 460.0 and h = 212.0 in
    let gap = 18.0 and pl = 10.0 and pr = 10.0 and pt = 22.0 and pb = 26.0 in
    let avail_w = w -. pl -. pr -. float_of_int (n_p - 1) *. gap in
    let box_s = min (avail_w /. float_of_int n_p) (h -. pt -. pb) in
    let total_w = float_of_int n_p *. box_s +. float_of_int (n_p - 1) *. gap in
    let start_x = 0.5 *. (w -. total_w) in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    List.iteri
      (fun p_idx p ->
        if p_idx < 2 then
          let nx = Option.value (opt_int p "nx") ~default:0 in
          let nz = Option.value (opt_int p "nz") ~default:0 in
          let vals = Array.of_list (json_num_list (Js.Unsafe.get p (js "vals"))) in
          if nx >= 2 && nz >= 2 && Array.length vals = nx * nz then (
            let vmin =
              match opt_float p "vmin" with
              | Some v -> v
              | None -> Array.fold_left min vals.(0) vals
            in
            let vmax =
              match opt_float p "vmax" with
              | Some v -> v
              | None -> Array.fold_left max vals.(0) vals
            in
            let span = if vmax -. vmin < 1e-12 then 1.0 else vmax -. vmin in
            let bx = start_x +. float_of_int p_idx *. (box_s +. gap) in
            let by = pt in
            let cw = box_s /. float_of_int nx in
            let ch = box_s /. float_of_int nz in
            let p_lbl =
              svg_el "text"
                [
                  ("x", Printf.sprintf "%.1f" (bx +. 0.5 *. box_s));
                  ("y", Printf.sprintf "%.1f" (pt -. 6.0));
                  ("text-anchor", "middle");
                  ("class", "py-svg-axis-lbl");
                ]
            in
            set_text p_lbl (Option.value (opt_str p "label") ~default:"Field");
            Dom.appendChild svg p_lbl;
            for iz = 0 to nz - 1 do
              for ix = 0 to nx - 1 do
                let v = vals.(iz * nx + ix) in
                let norm = max 0.0 (min 1.0 ((v -. vmin) /. span)) in
                let bucket = int_of_float (floor (norm *. 15.999)) in
                Dom.appendChild svg
                  (svg_el "rect"
                     [
                       ("x", Printf.sprintf "%.2f" (bx +. float_of_int ix *. cw));
                       ("y", Printf.sprintf "%.2f" (by +. float_of_int (nz - 1 - iz) *. ch));
                       ("width", Printf.sprintf "%.2f" (cw +. 0.4));
                       ("height", Printf.sprintf "%.2f" (ch +. 0.4));
                       ("class", Printf.sprintf "py-heat-%d" bucket);
                     ])
              done
            done;
            Dom.appendChild svg
              (svg_el "rect"
                 [
                   ("x", Printf.sprintf "%.2f" bx);
                   ("y", Printf.sprintf "%.2f" by);
                   ("width", Printf.sprintf "%.2f" box_s);
                   ("height", Printf.sprintf "%.2f" box_s);
                   ("class", "py-svg-frame");
                 ])))
      panels;
    (* Diverging colorbar strip at bottom *)
    let bar_w = 180.0 and bar_x = 0.5 *. (w -. 180.0) and bar_y = h -. 15.0 in
    for b = 0 to 15 do
      let bw = bar_w /. 16.0 in
      Dom.appendChild svg
        (svg_el "rect"
           [
             ("x", Printf.sprintf "%.2f" (bar_x +. float_of_int b *. bw));
             ("y", Printf.sprintf "%.2f" bar_y);
             ("width", Printf.sprintf "%.2f" (bw +. 0.4));
             ("height", "7.0");
             ("class", Printf.sprintf "py-heat-%d" b);
           ])
    done;
    let first_p = List.hd panels in
    let cmin = Option.value (opt_float first_p "vmin") ~default:(-1.0) in
    let cmax = Option.value (opt_float first_p "vmax") ~default:1.0 in
    let t_lo =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (bar_x -. 5.0));
          ("y", Printf.sprintf "%.1f" (bar_y +. 6.5));
          ("text-anchor", "end");
          ("class", "py-svg-tick");
        ]
    in
    set_text t_lo (Printf.sprintf "%.2g" cmin);
    Dom.appendChild svg t_lo;
    let t_hi =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (bar_x +. bar_w +. 5.0));
          ("y", Printf.sprintf "%.1f" (bar_y +. 6.5));
          ("text-anchor", "start");
          ("class", "py-svg-tick");
        ]
    in
    set_text t_hi (Printf.sprintf "%.2g" cmax);
    Dom.appendChild svg t_hi;
    Dom.appendChild stage svg;
    card

let render_band1d_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"1D Posterior & Uncertainty" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, stage = make_fig_pane title subtitle in
  let xg = json_num_list (Js.Unsafe.get fig (js "x_grid")) in
  let mu = json_num_list (Js.Unsafe.get fig (js "mu")) in
  if xg = [] || List.length xg <> List.length mu then card
  else
    let y_true = json_num_list (Js.Unsafe.get fig (js "y_true")) in
    let lo_2se = json_num_list (Js.Unsafe.get fig (js "lo_2se")) in
    let hi_2se = json_num_list (Js.Unsafe.get fig (js "hi_2se")) in
    let mw_lo = json_num_list (Js.Unsafe.get fig (js "mw_lo")) in
    let mw_hi = json_num_list (Js.Unsafe.get fig (js "mw_hi")) in
    let tr_x = json_num_list (Js.Unsafe.get fig (js "train_x")) in
    let tr_y = json_num_list (Js.Unsafe.get fig (js "train_y")) in
    let legend = mk ~cls:"py-fig-legend" "div" in
    append_legend_badge legend "py-swatch-0" "ENN mean";
    if y_true <> [] then append_legend_badge legend "py-swatch-1 is-dashed" "True f(x)";
    if lo_2se <> [] && hi_2se <> [] then append_legend_badge legend "py-swatch-band" "mu +/- 2*se";
    if mw_lo <> [] && mw_hi <> [] then append_legend_badge legend "py-swatch-2 is-dashed" "McShane-Whitney";
    Dom.appendChild stage legend;
    let all_y = mu @ y_true @ lo_2se @ hi_2se @ mw_lo @ mw_hi @ tr_y in
    let x_lo = List.fold_left min (List.hd xg) xg in
    let x_hi = List.fold_left max (List.hd xg) xg in
    let x_sp = if x_hi -. x_lo < 1e-9 then 1.0 else x_hi -. x_lo in
    let y_lo_r = List.fold_left min (List.hd all_y) all_y in
    let y_hi_r = List.fold_left max (List.hd all_y) all_y in
    let y_sp_r = if y_hi_r -. y_lo_r < 1e-9 then 1.0 else y_hi_r -. y_lo_r in
    let y_lo = y_lo_r -. 0.06 *. y_sp_r in
    let y_hi = y_hi_r +. 0.06 *. y_sp_r in
    let y_sp = y_hi -. y_lo in
    let w = 460.0 and h = 198.0 in
    let pl = 52.0 and pr = 16.0 and pt = 14.0 and pb = 28.0 in
    let pw = w -. pl -. pr and ph = h -. pt -. pb in
    let x_px x = pl +. ((x -. x_lo) /. x_sp) *. pw in
    let y_px y = pt +. (1.0 -. ((y -. y_lo) /. y_sp)) *. ph in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let yy = pt +. (1.0 -. frac) *. ph in
      let v = y_lo +. frac *. y_sp in
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.1f" pl);
             ("x2", Printf.sprintf "%.1f" (w -. pr));
             ("y1", Printf.sprintf "%.1f" yy);
             ("y2", Printf.sprintf "%.1f" yy);
             ("class", "py-svg-grid");
           ]);
      let lbl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" (pl -. 6.0));
            ("y", Printf.sprintf "%.1f" (yy +. 3.5));
            ("text-anchor", "end");
            ("class", "py-svg-tick");
          ]
      in
      set_text lbl (Printf.sprintf "%.2g" v);
      Dom.appendChild svg lbl
    done;
    let pts_of xs ys =
      let rec go ax ay acc =
        match (ax, ay) with
        | x :: xt, y :: yt -> go xt yt (Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y) :: acc)
        | _ -> List.rev acc
      in
      String.concat " " (go xs ys [])
    in
    if List.length lo_2se = List.length xg && List.length hi_2se = List.length xg then (
      let top =
        List.map2 (fun x y -> Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y)) xg hi_2se
      in
      let bot =
        List.rev
          (List.map2 (fun x y -> Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y)) xg lo_2se)
      in
      Dom.appendChild svg
        (svg_el "polygon"
           [ ("points", String.concat " " (top @ bot)); ("class", "py-svg-band-fill") ]));
    if List.length mw_hi = List.length xg && List.length mw_lo = List.length xg then (
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg mw_hi); ("class", "py-svg-mw-line") ]);
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg mw_lo); ("class", "py-svg-mw-line") ]));
    if List.length y_true = List.length xg then
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg y_true); ("class", "py-svg-band-true") ]);
    Dom.appendChild svg
      (svg_el "polyline"
         [ ("fill", "none"); ("points", pts_of xg mu); ("class", "py-svg-series py-svg-series-0") ]);
    let rec iter_pairs f xs ys =
      match (xs, ys) with
      | x :: xt, y :: yt ->
          f x y;
          iter_pairs f xt yt
      | _ -> ()
    in
    iter_pairs
      (fun x y ->
        Dom.appendChild svg
          (svg_el "circle"
             [
               ("cx", Printf.sprintf "%.2f" (x_px x));
               ("cy", Printf.sprintf "%.2f" (y_px y));
               ("r", "2.6");
               ("class", "py-svg-obs-dot-strong");
             ]))
      tr_x tr_y;
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" (w -. pr));
           ("y1", Printf.sprintf "%.1f" (h -. pb));
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    Dom.appendChild stage svg;
    card

let append_telemetry_band (panel : element Js.t) (state_raw : Js.Unsafe.any) =
  let band = mk ~cls:"py-telemetry-band" "div" in
  let stats = mk ~cls:"py-telemetry-stats" "div" in
  let vname = Option.value (opt_str state_raw "var") ~default:"state" in
  Dom.appendChild stats (mk ~cls:"py-telemetry-var" ~txt:vname "span");
  let add_stat label value =
    let item = mk ~cls:"py-telemetry-item" "span" in
    Dom.appendChild item (mk ~cls:"py-telemetry-lbl" ~txt:label "span");
    Dom.appendChild item (mk ~cls:"py-telemetry-val" ~txt:value "strong");
    Dom.appendChild stats item
  in
  add_stat "obs" (string_of_int (Option.value (opt_int state_raw "num_observations") ~default:0));
  add_stat "epoch" (string_of_int (Option.value (opt_int state_raw "epoch") ~default:0));
  add_stat "regions" (string_of_int (Option.value (opt_int state_raw "num_regions") ~default:1));
  add_stat "tr" (match opt_float state_raw "tr_length" with Some f -> Printf.sprintf "%.4g" f | None -> "-");
  add_stat "best" (match opt_float state_raw "best_y" with Some f -> Printf.sprintf "%.5f" f | None -> "-");
  let pend = Option.value (opt_int state_raw "pending_count") ~default:0 in
  if pend > 0 then add_stat "pending" (string_of_int pend);
  Dom.appendChild band stats;
  let best_vec_opt =
    match opt_str state_raw "best_params" with
    | Some bp -> Some ("best_params", bp)
    | None -> (
        let bx = json_num_list (Js.Unsafe.get state_raw (js "best_x")) in
        if bx <> [] then
          Some ("best_x", "[" ^ String.concat ", " (List.map (Printf.sprintf "%.4f") bx) ^ "]")
        else None)
  in
  (match best_vec_opt with
  | Some (lbl, txt) ->
      let row = mk ~cls:"py-best-row" "div" in
      Dom.appendChild row (mk ~cls:"py-telemetry-lbl" ~txt:lbl "span");
      let code_el = mk ~cls:"py-best-inline" ~txt "code" in
      let exp_btn = mk ~cls:"py-best-toggle" ~txt:"Expand" "button" in
      exp_btn##setAttribute (js "type") (js "button");
      exp_btn##.onclick :=
        Dom.handler (fun _ ->
            let cur = to_s code_el##.className in
            if contains_sub cur "is-expanded" then (
              code_el##.className := js "py-best-inline";
              set_text exp_btn "Expand")
            else (
              code_el##.className := js "py-best-inline is-expanded";
              set_text exp_btn "Collapse");
            Js._true);
      let cpy_btn = mk ~cls:"py-best-toggle" ~txt:"Copy" "button" in
      cpy_btn##setAttribute (js "type") (js "button");
      cpy_btn##.onclick :=
        Dom.handler (fun _ ->
            let clip = Js.Unsafe.get Dom_html.window##.navigator (js "clipboard") in
            if is_defined clip then (
              ignore (Js.Unsafe.meth_call clip "writeText" [| Js.Unsafe.inject (js txt) |]);
              set_text cpy_btn "Copied";
              ignore
                (Dom_html.window##setTimeout
                   (Js.wrap_callback (fun () -> set_text cpy_btn "Copy"))
                   (Js.float 1100.0)));
            Js._true);
      Dom.appendChild row code_el;
      Dom.appendChild row exp_btn;
      Dom.appendChild row cpy_btn;
      Dom.appendChild band row
  | None -> ());
  Dom.appendChild panel band

let render_fig_split (panel : element Js.t) (state_raw : Js.Unsafe.any) (fig : Js.Unsafe.any) =
  let has_opt = is_defined (Js.Unsafe.get state_raw (js "dim")) in
  let kind = Option.value (opt_str fig "kind") ~default:"" in
  let append_subcurves () =
    match curves_from_js (Js.Unsafe.get fig (js "curves")) with
    | Some c -> Dom.appendChild panel c
    | None -> ()
  in
  match kind with
  | "curves" ->
      (match curves_from_js fig with Some c -> Dom.appendChild panel c | None -> ());
      if has_opt then append_telemetry_band panel state_raw
  | "tokamak" ->
      Dom.appendChild panel (render_tokamak_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | "field2d" ->
      Dom.appendChild panel (render_field2d_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | "band1d" ->
      Dom.appendChild panel (render_band1d_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | _ -> if has_opt then append_telemetry_band panel state_raw

let render_state_panel ?empty_node (panel : element Js.t) (state_raw : Js.Unsafe.any) =
  clear_children (panel :> Dom.node Js.t);
  if not (is_defined state_raw) then (
    match empty_node with
    | Some node -> Dom.appendChild panel node
    | None ->
        let hint =
          mk ~cls:"py-inspector-empty"
            ~txt:
              "Run a script that creates OptimizerState variable(s) or a `fig` diagnostic dict to inspect live convergence trajectories, tokamak equilibrium fluxes, PDE fields, and ENN uncertainty bands."
            "p"
        in
        Dom.appendChild panel hint)
  else
    let fig_raw = Js.Unsafe.get state_raw (js "fig") in
    let has_opt = is_defined (Js.Unsafe.get state_raw (js "dim")) in
    (if is_defined fig_raw then render_fig_split panel state_raw fig_raw
     else if has_opt then (
       let y_hist = json_num_list (Js.Unsafe.get state_raw (js "y_history")) in
       let best_hist = json_num_list (Js.Unsafe.get state_raw (js "best_history")) in
       if best_hist <> [] then
         let xs = List.mapi (fun i _ -> float_of_int (i + 1)) best_hist in
         let series =
           [ { c_label = "Best y"; c_xs = xs; c_ys = best_hist; c_dashed = false } ]
           @
           if List.length y_hist = List.length xs then
             [ { c_label = "Raw eval y"; c_xs = xs; c_ys = y_hist; c_dashed = true } ]
           else []
         in
         Dom.appendChild panel
           (render_curves_card ~title:"Cumulative Best Objective"
              ~subtitle:"Best-so-far vs raw observations" ~x_label:"Evaluations"
              ~y_label:"Objective" ~log_y:false series);
       append_telemetry_band panel state_raw));

    let recent_raw = Js.Unsafe.get state_raw (js "recent") in
    if Js.Optdef.test recent_raw then (
      let rows : Js.Unsafe.any Js.js_array Js.t = Obj.magic recent_raw in
      if rows##.length > 0 then (
        let det = mk ~cls:"py-recent-details" "details" in
        let sum =
          mk ~cls:"py-recent-summary"
            ~txt:(Printf.sprintf "Recent evaluations (last %d)" rows##.length)
            "summary"
        in
        Dom.appendChild det sum;
        let tbl_wrap = mk ~cls:"py-recent-wrap" "div" in
        let tbl = mk ~cls:"py-recent-table" "table" in
        let thead = mk "thead" in
        let hr = mk "tr" in
        List.iter (fun h -> Dom.appendChild hr (mk ~txt:h "th")) [ "#"; "y"; "Best"; "x[:4]" ];
        Dom.appendChild thead hr;
        Dom.appendChild tbl thead;
        let tbody = mk "tbody" in
        for i = rows##.length - 1 downto 0 do
          match Js.Optdef.to_option (Js.array_get rows i) with
          | None -> ()
          | Some r ->
              let tr = mk "tr" in
              let obs_n = string_of_int (Option.value (opt_int r "obs") ~default:0) in
              let yv = match opt_float r "y" with Some f -> Printf.sprintf "%.4f" f | None -> "-" in
              let bv = match opt_float r "best_so_far" with Some f -> Printf.sprintf "%.4f" f | None -> "-" in
              let xs = json_num_list (Js.Unsafe.get r (js "x")) in
              let xs_cut =
                if List.length xs <= 4 then xs
                else [ List.nth xs 0; List.nth xs 1; List.nth xs 2; List.nth xs 3 ]
              in
              let xstr = "[" ^ String.concat ", " (List.map (Printf.sprintf "%.2f") xs_cut) ^ "]" in
              List.iter (fun cell -> Dom.appendChild tr (mk ~txt:cell "td")) [ obs_n; yv; bv; xstr ];
              Dom.appendChild tbody tr
        done;
        Dom.appendChild tbl tbody;
        Dom.appendChild tbl_wrap tbl;
        Dom.appendChild det tbl_wrap;
        Dom.appendChild panel det))

(* ---------------------------------------------------------------------------
   Pure-OCaml Python Syntax Tokenizer & Synchronized Line-Number Gutter
   --------------------------------------------------------------------------- *)

let is_ident_start c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'

let is_ident_char c =
  is_ident_start c || (c >= '0' && c <= '9')

let is_digit c = c >= '0' && c <= '9'

let is_py_keyword = function
  | "and" | "as" | "assert" | "async" | "await" | "break" | "class"
  | "continue" | "def" | "del" | "elif" | "else" | "except" | "False"
  | "finally" | "for" | "from" | "global" | "if" | "import" | "in"
  | "is" | "lambda" | "None" | "nonlocal" | "not" | "or" | "pass"
  | "raise" | "return" | "True" | "try" | "while" | "with" | "yield" ->
      true
  | _ -> false

let render_py_highlight (code_el : element Js.t) (src : string) =
  clear_children (code_el :> Dom.node Js.t);
  let len = String.length src in
  let plain_buf = Buffer.create 128 in
  let flush_plain () =
    if Buffer.length plain_buf > 0 then (
      let tn = document##createTextNode (js (Buffer.contents plain_buf)) in
      Dom.appendChild code_el tn;
      Buffer.clear plain_buf)
  in
  let emit_span cls txt =
    flush_plain ();
    Dom.appendChild code_el (mk ~cls ~txt "span")
  in
  let rec scan i =
    if i >= len then ()
    else
      let c = src.[i] in
      if c = '#' then (
        let j = ref (i + 1) in
        while !j < len && src.[!j] <> '\n' do
          incr j
        done;
        emit_span "code-comment" (String.sub src i (!j - i));
        scan !j)
      else if c = '\'' || c = '"' then
        let q = c in
        let is_triple = i + 2 < len && src.[i + 1] = q && src.[i + 2] = q in
        let j = ref (i + if is_triple then 3 else 1) in
        let done_s = ref false in
        while !j < len && not !done_s do
          if src.[!j] = '\\' then j := min len (!j + 2)
          else if is_triple then
            if !j + 2 < len && src.[!j] = q && src.[!j + 1] = q && src.[!j + 2] = q then (
              j := !j + 3;
              done_s := true)
            else incr j
          else if src.[!j] = q || src.[!j] = '\n' then (
            if src.[!j] = q then incr j;
            done_s := true)
          else incr j
        done;
        emit_span "code-string" (String.sub src i (!j - i));
        scan !j
      else if
        (c = 'f' || c = 'r' || c = 'b' || c = 'u' || c = 'F' || c = 'R' || c = 'B' || c = 'U')
        && i + 1 < len
        && (src.[i + 1] = '\'' || src.[i + 1] = '"')
      then (
        Buffer.add_char plain_buf c;
        scan (i + 1))
      else if is_digit c || (c = '.' && i + 1 < len && is_digit src.[i + 1]) then (
        let j = ref i in
        if
          c = '0'
          && i + 1 < len
          && (src.[i + 1] = 'x' || src.[i + 1] = 'X' || src.[i + 1] = 'b' || src.[i + 1] = 'o')
        then (
          j := i + 2;
          while !j < len && is_ident_char src.[!j] do
            incr j
          done)
        else (
          while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
            incr j
          done;
          if !j < len && src.[!j] = '.' then (
            incr j;
            while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
              incr j
            done);
          if !j < len && (src.[!j] = 'e' || src.[!j] = 'E') then (
            let e_pos = !j in
            incr j;
            if !j < len && (src.[!j] = '+' || src.[!j] = '-') then incr j;
            if !j < len && is_digit src.[!j] then
              while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
                incr j
              done
            else j := e_pos));
        emit_span "code-number" (String.sub src i (!j - i));
        scan !j)
      else if is_ident_start c then (
        let j = ref (i + 1) in
        while !j < len && is_ident_char src.[!j] do
          incr j
        done;
        let word = String.sub src i (!j - i) in
        if is_py_keyword word then emit_span "code-keyword" word else Buffer.add_string plain_buf word;
        scan !j)
      else (
        Buffer.add_char plain_buf c;
        scan (i + 1))
  in
  scan 0;
  flush_plain ();
  Dom.appendChild code_el (document##createTextNode (js "\n"))

let count_lines s =
  let n = ref 1 in
  String.iter (fun ch -> if ch = '\n' then incr n) s;
  !n

let render_line_numbers (gutter : element Js.t) prev_ref n =
  if n <> !prev_ref then (
    prev_ref := n;
    let buf = Buffer.create (n * 4) in
    for k = 1 to n do
      if k > 1 then Buffer.add_char buf '\n';
      Buffer.add_string buf (string_of_int k)
    done;
    Buffer.add_char buf '\n';
    set_text gutter (Buffer.contents buf))

(* ---------------------------------------------------------------------------
   Full Playground Workbench UI (#py-playground-root)
   --------------------------------------------------------------------------- *)

let append_console_chunk (out_pre : element Js.t) cls text =
  if String.length text > 0 then (
    let sp = mk ~cls ~txt:text "span" in
    Dom.appendChild out_pre sp;
    out_pre##.scrollTop := Js.float (float_of_int out_pre##.scrollHeight))

let initial_playground_code () =
  let stored =
    try
      let ss = Js.Unsafe.get Dom_html.window (js "sessionStorage") in
      let raw = Js.Unsafe.meth_call ss "getItem" [| Js.Unsafe.inject (js "baux_playground_code") |] in
      if Js.Opt.test raw then (
        ignore (Js.Unsafe.meth_call ss "removeItem" [| Js.Unsafe.inject (js "baux_playground_code") |]);
        Some (to_s raw))
      else None
    with _ -> None
  in
  match stored with
  | Some code -> (None, code)
  | None ->
      let first = List.hd presets in
      (Some first.id, first.code)

let mount_playground (root : element Js.t) =
  clear_children (root :> Dom.node Js.t);
  let shell = mk ~cls:"py-pg-app" "div" in
  shell##setAttribute (js "data-mob-tab") (js "script");

  (* Band 1: Unified Top Bar (Preset Tabs + Kernel Status + Primary Controls) *)
  let topbar = mk ~cls:"py-pg-topbar" "div" in
  let preset_rail = mk ~cls:"py-preset-rail" "div" in
  preset_rail##setAttribute (js "role") (js "tablist");
  preset_rail##setAttribute (js "aria-label") (js "Playground presets");

  let controls = mk ~cls:"py-pg-controls" "div" in
  let status_pill = mk ~cls:"py-kernel-pill" "div" in
  let status_dot = mk ~cls:"py-status-dot" "span" in
  let status_lbl = mk ~cls:"py-status-text" ~txt:"Idle" "span" in
  let timing_lbl = mk ~cls:"py-timing-badge" ~txt:"" "span" in
  Dom.appendChild status_pill status_dot;
  Dom.appendChild status_pill status_lbl;
  Dom.appendChild status_pill timing_lbl;

  let run_btn = mk ~cls:"py-btn py-btn-primary" ~txt:"▶ Run" "button" in
  run_btn##setAttribute (js "type") (js "button");
  run_btn##setAttribute (js "title") (js "Run script (Ctrl/Cmd + Enter)");
  let reset_btn = mk ~cls:"py-btn" ~txt:"Reset" "button" in
  reset_btn##setAttribute (js "type") (js "button");
  reset_btn##setAttribute (js "title") (js "Reset Python namespace / interrupt execution");
  Dom.appendChild controls status_pill;
  Dom.appendChild controls run_btn;
  Dom.appendChild controls reset_btn;
  Dom.appendChild topbar preset_rail;
  Dom.appendChild topbar controls;

  (* Band 2: Editorial Caption Bar *)
  let cap_bar = mk ~cls:"py-pg-caption" "div" in
  let cap_title = mk ~cls:"py-cap-title" ~txt:(List.hd presets).label "span" in
  let cap_desc = mk ~cls:"py-cap-desc" ~txt:(List.hd presets).description "span" in
  Dom.appendChild cap_bar cap_title;
  Dom.appendChild cap_bar cap_desc;

  (* Band 2b: Mobile (<900px) Segmented Switcher *)
  let mob_tabs = mk ~cls:"py-mob-tabs" "div" in
  let mk_mob_tab tab_id label_txt =
    let b = mk ~cls:"py-mob-tab" ~txt:label_txt "button" in
    b##setAttribute (js "type") (js "button");
    b##setAttribute (js "data-tab") (js tab_id);
    (tab_id, b)
  in
  let mob_pairs =
    [
      mk_mob_tab "script" "Script";
      mk_mob_tab "plots" "Plots & State";
      mk_mob_tab "console" "Console";
    ]
  in
  List.iter (fun (_, b) -> Dom.appendChild mob_tabs b) mob_pairs;

  let set_mob_tab target =
    shell##setAttribute (js "data-mob-tab") (js target);
    List.iter
      (fun (tid, b) ->
        let base = "py-mob-tab" in
        b##.className := js (if tid = target then base ^ " active" else base))
      mob_pairs
  in
  set_mob_tab "script";
  List.iter
    (fun (tid, b) ->
      b##.onclick :=
        Dom.handler (fun _ ->
            set_mob_tab tid;
            Js._true))
    mob_pairs;

  (* Band 3: Asymmetric Editorial Grid *)
  let grid = mk ~cls:"py-pg-grid" "div" in
  let left_col = mk ~cls:"py-pg-left" "div" in
  let right_col = mk ~cls:"py-pg-right" "div" in

  (* Editor Sub-pane *)
  let ed_pane = mk ~cls:"py-ed-pane" "div" in
  let ed_hdr = mk ~cls:"py-subbar" "div" in
  let ed_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild ed_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Script" "span");
  let ed_meta = mk ~cls:"py-subbar-meta" ~txt:"sparc_pf.py" "span" in
  Dom.appendChild ed_hdr_l ed_meta;
  let ed_hdr_r = mk ~cls:"py-subbar-right" "div" in
  Dom.appendChild ed_hdr_r (mk ~cls:"py-key-hint" ~txt:"⌘↵ run" "span");
  let copy_btn = mk ~cls:"py-ghost-btn" ~txt:"Copy" "button" in
  copy_btn##setAttribute (js "type") (js "button");
  Dom.appendChild ed_hdr_r copy_btn;
  Dom.appendChild ed_hdr ed_hdr_l;
  Dom.appendChild ed_hdr ed_hdr_r;

  let ed_shell = mk ~cls:"py-ed-shell" "div" in
  let ed_gutter = mk ~cls:"py-ed-gutter" "div" in
  ed_gutter##setAttribute (js "aria-hidden") (js "true");
  let ed_stage = mk ~cls:"py-ed-stage" "div" in
  let highlight_pre = mk ~cls:"py-ed-highlight" "pre" in
  highlight_pre##setAttribute (js "aria-hidden") (js "true");
  let highlight_code = mk "code" in
  Dom.appendChild highlight_pre highlight_code;
  let editor = createTextarea document in
  editor##.className := js "py-ed-textarea";
  editor##setAttribute (js "wrap") (js "off");
  editor##setAttribute (js "spellcheck") (js "false");
  editor##setAttribute (js "autocomplete") (js "off");
  editor##setAttribute (js "autocorrect") (js "off");
  editor##setAttribute (js "autocapitalize") (js "off");
  editor##setAttribute (js "aria-label") (js "Python script editor");
  Dom.appendChild ed_stage highlight_pre;
  Dom.appendChild ed_stage editor;
  Dom.appendChild ed_shell ed_gutter;
  Dom.appendChild ed_shell ed_stage;
  Dom.appendChild ed_pane ed_hdr;
  Dom.appendChild ed_pane ed_shell;

  let active_preset_ref : preset option ref = ref None in
  let prev_lines_ref = ref 0 in
  let user_dirty_ref = ref false in
  let sync_editor_view () =
    let src = to_s editor##.value in
    render_py_highlight highlight_code src;
    let n = count_lines src in
    render_line_numbers ed_gutter prev_lines_ref n;
    let stem =
      match !active_preset_ref with
      | Some p -> p.id ^ ".py" ^ if !user_dirty_ref then " *" else ""
      | None -> "snippet.py"
    in
    set_text ed_meta (Printf.sprintf "%s · %d lines" stem n);
    highlight_pre##.scrollTop := editor##.scrollTop;
    highlight_pre##.scrollLeft := editor##.scrollLeft;
    ed_gutter##.scrollTop := editor##.scrollTop
  in

  (* Console + Unified REPL Pane *)
  let con_pane = mk ~cls:"py-con-pane" "div" in
  let con_hdr = mk ~cls:"py-subbar" "div" in
  let con_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild con_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Console" "span");
  Dom.appendChild con_hdr_l (mk ~cls:"py-subbar-meta" ~txt:"stdout · python 3.12" "span");
  let con_hdr_r = mk ~cls:"py-subbar-right" "div" in
  let clear_btn = mk ~cls:"py-ghost-btn" ~txt:"Clear" "button" in
  clear_btn##setAttribute (js "type") (js "button");
  Dom.appendChild con_hdr_r clear_btn;
  Dom.appendChild con_hdr con_hdr_l;
  Dom.appendChild con_hdr con_hdr_r;

  let console_pre = mk ~cls:"py-pg-console" "pre" in
  append_console_chunk console_pre "py-out-muted"
    "Warming CPython 3.12 WebAssembly kernel...\n";

  let prompt_row = createForm document in
  prompt_row##.className := js "py-repl-prompt-row";
  let prompt_glyph = mk ~cls:"py-repl-glyph" ~txt:">>>" "span" in
  let repl_input = createInput ~_type:(js "text") document in
  repl_input##.className := js "py-repl-input";
  repl_input##setAttribute (js "placeholder") (js "Inspect state (e.g. state.best_y, dir(baux))");
  repl_input##setAttribute (js "spellcheck") (js "false");
  repl_input##setAttribute (js "autocomplete") (js "off");
  repl_input##setAttribute (js "aria-label") (js "Interactive Python prompt");
  let repl_send = mk ~cls:"py-ghost-btn py-repl-send" ~txt:"Eval ↵" "button" in
  repl_send##setAttribute (js "type") (js "submit");
  Dom.appendChild prompt_row prompt_glyph;
  Dom.appendChild prompt_row repl_input;
  Dom.appendChild prompt_row repl_send;

  Dom.appendChild con_pane con_hdr;
  Dom.appendChild con_pane console_pre;
  Dom.appendChild con_pane prompt_row;

  Dom.appendChild left_col ed_pane;
  Dom.appendChild left_col con_pane;

  (* Right Output & Figures Stage *)
  let insp_hdr = mk ~cls:"py-subbar" "div" in
  let insp_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild insp_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Figures & State" "span");
  let insp_chip = mk ~cls:"py-subbar-meta py-state-chip" ~txt:"awaiting run" "span" in
  let insp_hdr_r = mk ~cls:"py-subbar-right" "div" in
  Dom.appendChild insp_hdr_r insp_chip;
  Dom.appendChild insp_hdr insp_hdr_l;
  Dom.appendChild insp_hdr insp_hdr_r;

  let insp_body = mk ~cls:"py-inspector-body" "div" in
  Dom.appendChild right_col insp_hdr;
  Dom.appendChild right_col insp_body;

  Dom.appendChild grid left_col;
  Dom.appendChild grid right_col;

  Dom.appendChild shell topbar;
  Dom.appendChild shell cap_bar;
  Dom.appendChild shell mob_tabs;
  Dom.appendChild shell grid;
  Dom.appendChild root shell;

  (* Build Empty Stage helper *)
  let initial_ran_ref = ref false in
  let manual_abort_ref = ref false in
  let run_script_ref : (unit -> unit) ref = ref (fun () -> ()) in

  let make_empty_stage () =
    let box = mk ~cls:"py-empty-stage" "div" in
    let ttl =
      match !active_preset_ref with
      | Some p -> p.label
      | None -> "Documentation Snippet"
    in
    let dsc =
      match !active_preset_ref with
      | Some p -> p.description
      | None -> "Execute the loaded documentation example to visualize trajectories and state."
    in
    Dom.appendChild box (mk ~cls:"py-empty-title" ~txt:ttl "div");
    Dom.appendChild box (mk ~cls:"py-empty-desc" ~txt:dsc "p");
    let run_cta = mk ~cls:"py-btn py-btn-primary" ~txt:"▶ Execute now" "button" in
    run_cta##setAttribute (js "type") (js "button");
    run_cta##.onclick :=
      Dom.handler (fun _ ->
          !run_script_ref ();
          Js._true);
    Dom.appendChild box run_cta;
    box
  in

  let sync_state_chip st_raw =
    if not (is_defined st_raw) then set_text insp_chip "awaiting run"
    else
      let has_opt = is_defined (Js.Unsafe.get st_raw (js "dim")) in
      let has_fig = is_defined (Js.Unsafe.get st_raw (js "fig")) in
      if has_opt then
        let vn = Option.value (opt_str st_raw "var") ~default:"state" in
        let dm = Option.value (opt_int st_raw "dim") ~default:0 in
        set_text insp_chip (Printf.sprintf "var %s · %dD" vn dm)
      else if has_fig then set_text insp_chip "diagnostic figure"
      else set_text insp_chip "ready"
  in

  let (init_preset_id, init_code) = initial_playground_code () in
  active_preset_ref :=
    (match init_preset_id with Some pid -> find_preset pid | None -> None);
  editor##.value := js init_code;
  (match !active_preset_ref with
  | Some p ->
      set_text cap_title p.label;
      set_text cap_desc p.description
  | None ->
      set_text cap_title "Custom documentation snippet";
      set_text cap_desc "Loaded directly from docs—edit or run in WebAssembly Python.");
  sync_editor_view ();
  render_state_panel ~empty_node:(make_empty_stage ()) insp_body (Js.Unsafe.inject Js.undefined);

  let preset_btns : (string * element Js.t) list =
    List.map
      (fun p ->
        let btn = mk ~cls:"py-preset-tab" ~txt:p.short_label "button" in
        btn##setAttribute (js "type") (js "button");
        btn##setAttribute (js "role") (js "tab");
        btn##setAttribute (js "title") (js p.label);
        Dom.appendChild preset_rail btn;
        (p.id, btn))
      presets
  in

  let sync_preset_active () =
    let active_id = match !active_preset_ref with Some p -> p.id | None -> "" in
    List.iter
      (fun (pid, b) ->
        let sel_flag = pid = active_id in
        b##.className := js (if sel_flag then "py-preset-tab active" else "py-preset-tab");
        b##setAttribute (js "aria-selected") (js (if sel_flag then "true" else "false")))
      preset_btns
  in
  sync_preset_active ();

  let mark_mob_updates () =
    List.iter
      (fun (tid, b) ->
        if tid = "plots" then (
          let cur = to_s b##.className in
          if not (contains_sub cur "has-update") then b##.className := js (cur ^ " has-update")))
      mob_pairs
  in

  let run_script () =
    initial_ran_ref := true;
    manual_abort_ref := false;
    let src = to_s editor##.value in
    clear_children (console_pre :> Dom.node Js.t);
    append_console_chunk console_pre "py-out-muted" ">>> Executing script...\n";
    set_text timing_lbl "";
    send_worker "exec"
      [
        ("code", Js.Unsafe.inject (js src));
        ("resetNamespace", Js.Unsafe.inject Js._true);
      ]
      (fun res ->
        let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
        let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
        let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
        let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
        let dt_s =
          if Js.Optdef.test dt_raw then Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw) else ""
        in
        set_text timing_lbl (if dt_s <> "" then "· " ^ dt_s else "");
        clear_children (console_pre :> Dom.node Js.t);
        if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
        if err_s <> "" then append_console_chunk console_pre "py-out-stderr" err_s;
        if out_s = "" && err_s = "" && ok then
          append_console_chunk console_pre "py-out-muted" "(completed with no printed output)\n";
        set_text prompt_glyph ">>>";
        let st_raw = Js.Unsafe.get res (js "state") in
        sync_state_chip st_raw;
        render_state_panel ~empty_node:(make_empty_stage ()) insp_body st_raw;
        mark_mob_updates ())
  in
  run_script_ref := run_script;

  add_status_listener (fun st ->
      let short_s =
        match st with
        | Idle -> "Idle"
        | Booting _ -> "Booting WASM..."
        | Ready -> "Py 3.12 · Ready"
        | Running -> "Running..."
        | Interrupted _ -> "Stopped"
      in
      set_text status_lbl short_s;
      status_pill##setAttribute (js "title") (js (status_text st));
      let cls =
        match st with
        | Ready -> "py-status-dot is-ready"
        | Booting _ | Running -> "py-status-dot is-busy"
        | Interrupted _ -> "py-status-dot is-err"
        | Idle -> "py-status-dot"
      in
      status_dot##.className := js cls;
      set_text reset_btn (match st with Running | Booting _ -> "Interrupt" | _ -> "Reset");
      if st = Ready && (not !initial_ran_ref) && (not !user_dirty_ref) && (not !manual_abort_ref) then
        run_script ());

  List.iter
    (fun (pid, btn) ->
      btn##.onclick :=
        Dom.handler (fun _ ->
            (match find_preset pid with
            | Some p ->
                active_preset_ref := Some p;
                user_dirty_ref := false;
                editor##.value := js p.code;
                editor##.scrollTop := Js.float 0.0;
                editor##.scrollLeft := Js.float 0.0;
                set_text cap_title p.label;
                set_text cap_desc p.description;
                sync_preset_active ();
                sync_editor_view ();
                render_state_panel ~empty_node:(make_empty_stage ()) insp_body
                  (Js.Unsafe.inject Js.undefined);
                sync_state_chip (Js.Unsafe.inject Js.undefined);
                if !kernel_state_ref = Ready then run_script ()
            | None -> ());
            Js._true))
    preset_btns;

  let repl_history : string list ref = ref [] in
  let hist_idx : int ref = ref (-1) in

  let push_repl_line line =
    let glyph_now = to_s (Js.Opt.get prompt_glyph##.textContent (fun () -> js ">>>")) in
    append_console_chunk console_pre "py-out-cmd" (glyph_now ^ " " ^ line ^ "\n");
    if trim_s line <> "" then (
      repl_history := line :: !repl_history;
      hist_idx := -1);
    send_worker "repl" [ ("line", Js.Unsafe.inject (js line)) ] (fun res ->
        let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
        let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
        let inc = Js.to_bool (Js.Unsafe.get res (js "incomplete")) in
        let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
        if Js.Optdef.test dt_raw then
          set_text timing_lbl (Printf.sprintf "· %.1f ms" (Js.float_of_number dt_raw));
        if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
        if err_s <> "" then append_console_chunk console_pre "py-out-stderr" err_s;
        set_text prompt_glyph (if inc then "..." else ">>>");
        let st_raw = Js.Unsafe.get res (js "state") in
        sync_state_chip st_raw;
        render_state_panel ~empty_node:(make_empty_stage ()) insp_body st_raw)
  in

  run_btn##.onclick :=
    Dom.handler (fun _ ->
        run_script ();
        (if Dom_html.window##.innerWidth <= 900 then set_mob_tab "plots");
        Js._true);

  reset_btn##.onclick :=
    Dom.handler (fun _ ->
        manual_abort_ref := true;
        interrupt_kernel ();
        set_text prompt_glyph ">>>";
        set_text timing_lbl "";
        warm_kernel ();
        Js._true);

  clear_btn##.onclick :=
    Dom.handler (fun _ ->
        clear_children (console_pre :> Dom.node Js.t);
        append_console_chunk console_pre "py-out-muted" "Console cleared.\n";
        Js._true);

  copy_btn##.onclick :=
    Dom.handler (fun _ ->
        let clip = Js.Unsafe.get Dom_html.window##.navigator (js "clipboard") in
        if is_defined clip then (
          ignore (Js.Unsafe.meth_call clip "writeText" [| Js.Unsafe.inject editor##.value |]);
          set_text copy_btn "Copied";
          ignore
            (Dom_html.window##setTimeout
               (Js.wrap_callback (fun () -> set_text copy_btn "Copy"))
               (Js.float 1200.0)));
        Js._true);

  editor##.oninput :=
    Dom.handler (fun _ ->
        user_dirty_ref := true;
        sync_editor_view ();
        Js._true);

  editor##.onscroll :=
    Dom.handler (fun _ ->
        highlight_pre##.scrollTop := editor##.scrollTop;
        highlight_pre##.scrollLeft := editor##.scrollLeft;
        ed_gutter##.scrollTop := editor##.scrollTop;
        Js._true);

  editor##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let ctrl_or_meta = Js.to_bool ev##.ctrlKey || Js.to_bool ev##.metaKey in
        if ctrl_or_meta && key = "Enter" then (
          Dom.preventDefault ev;
          run_script ();
          (if Dom_html.window##.innerWidth <= 900 then set_mob_tab "plots");
          Js._false)
        else if key = "Tab" then (
          Dom.preventDefault ev;
          let st = editor##.selectionStart in
          let en = editor##.selectionEnd in
          let v = to_s editor##.value in
          let left = String.sub v 0 st in
          let right = String.sub v en (String.length v - en) in
          editor##.value := js (left ^ "    " ^ right);
          editor##.selectionStart := st + 4;
          editor##.selectionEnd := st + 4;
          user_dirty_ref := true;
          sync_editor_view ();
          Js._false)
        else Js._true);

  prompt_row##.onsubmit :=
    Dom.handler (fun ev ->
        Dom.preventDefault ev;
        let line = to_s repl_input##.value in
        repl_input##.value := js "";
        push_repl_line line;
        Js._false);

  repl_input##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let hlen = List.length !repl_history in
        if key = "ArrowUp" && hlen > 0 then (
          Dom.preventDefault ev;
          let next_i = min (hlen - 1) (!hist_idx + 1) in
          hist_idx := next_i;
          repl_input##.value := js (List.nth !repl_history next_i);
          Js._false)
        else if key = "ArrowDown" then (
          Dom.preventDefault ev;
          if !hist_idx <= 0 then (
            hist_idx := -1;
            repl_input##.value := js "")
          else (
            let next_i = !hist_idx - 1 in
            hist_idx := next_i;
            repl_input##.value := js (List.nth !repl_history next_i));
          Js._false)
        else Js._true);

  (* Automatically warm runtime on the dedicated Playground page *)
  warm_kernel ()

(* ---------------------------------------------------------------------------
   Inline Doc Code Block Runner (.code-wrap inside article)
   --------------------------------------------------------------------------- *)

let is_runnable_baux_code code =
  (contains_sub code "import baux" || contains_sub code "from baux")
  && (not (contains_sub code "..."))
  && (not (contains_sub code "your_"))
  && (not (contains_sub code "oxcaml_step"))

let enhance_inline_blocks () =
  let wraps = document##querySelectorAll (js "article .code-wrap") in
  let len = wraps##.length in
  for i = 0 to len - 1 do
    match Js.Opt.to_option (wraps##item i) with
    | None -> ()
    | Some wrap ->
        let inside_pg =
          Js.Opt.test
            (Js.Unsafe.meth_call wrap "closest" [| Js.Unsafe.inject (js "#py-playground-root") |])
        in
        if (not inside_pg)
           && not (Js.Opt.test (wrap##querySelector (js ".code-run")))
        then
          match Js.Opt.to_option (wrap##querySelector (js "pre")) with
          | None -> ()
          | Some pre ->
              let orig_code = trim_s (to_s (Js.Opt.get pre##.textContent (fun () -> js ""))) in
              if is_runnable_baux_code orig_code then (
                wrap##.className := js (to_s wrap##.className ^ " has-py-run");
                let actions = mk ~cls:"code-actions" "div" in
                let run_btn = mk ~cls:"code-run" ~txt:"Run" "button" in
                run_btn##setAttribute (js "type") (js "button");
                run_btn##setAttribute (js "title") (js "Run live in browser Python REPL");
                let pg_btn = mk ~cls:"code-pg" ~txt:"Playground" "button" in
                pg_btn##setAttribute (js "type") (js "button");
                pg_btn##setAttribute (js "title") (js "Open in interactive Playground");
                Dom.appendChild actions run_btn;
                Dom.appendChild actions pg_btn;
                (match Js.Opt.to_option (wrap##querySelector (js ".code-copy")) with
                | Some copy_btn -> Dom.appendChild actions copy_btn
                | None -> ());
                Dom.appendChild wrap actions;

                let drawer_ref : element Js.t option ref = ref None in
                let editor_ref : textAreaElement Js.t option ref = ref None in
                let out_ref : element Js.t option ref = ref None in
                let meta_ref : element Js.t option ref = ref None in
                let chart_ref : element Js.t option ref = ref None in

                let ensure_drawer () =
                  match !drawer_ref with
                  | Some d ->
                      d##removeAttribute (js "hidden");
                      d
                  | None ->
                      let d = mk ~cls:"py-inline-drawer" "div" in
                      let bar = mk ~cls:"py-inline-bar" "div" in
                      let ed_toggle = mk ~cls:"py-btn py-btn-sm" ~txt:"Edit" "button" in
                      ed_toggle##setAttribute (js "type") (js "button");
                      let rerun_btn = mk ~cls:"py-btn py-btn-sm py-btn-primary" ~txt:"Run" "button" in
                      rerun_btn##setAttribute (js "type") (js "button");
                      let close_btn = mk ~cls:"py-btn py-btn-sm" ~txt:"Close" "button" in
                      close_btn##setAttribute (js "type") (js "button");
                      let meta = mk ~cls:"py-inline-meta" ~txt:"" "span" in
                      List.iter (fun x -> Dom.appendChild bar x) [ rerun_btn; ed_toggle; close_btn; meta ];

                      let ed = createTextarea document in
                      ed##.className := js "py-inline-editor";
                      ed##setAttribute (js "spellcheck") (js "false");
                      ed##setAttribute (js "hidden") (js "hidden");
                      ed##.rows := max 5 (min 14 (List.length (String.split_on_char '\n' orig_code) + 1));
                      ed##.value := js (orig_code ^ "\n");

                      let out = mk ~cls:"py-inline-out" "pre" in
                      let ch = mk ~cls:"py-inline-state" "div" in
                      Dom.appendChild d bar;
                      Dom.appendChild d ed;
                      Dom.appendChild d out;
                      Dom.appendChild d ch;
                      Dom.appendChild wrap d;
                      drawer_ref := Some d;
                      editor_ref := Some ed;
                      out_ref := Some out;
                      meta_ref := Some meta;
                      chart_ref := Some ch;

                      ed_toggle##.onclick :=
                        Dom.handler (fun _ ->
                            if Js.to_bool (ed##hasAttribute (js "hidden")) then (
                              ed##removeAttribute (js "hidden");
                              set_text ed_toggle "Hide Editor")
                            else (
                              ed##setAttribute (js "hidden") (js "hidden");
                              set_text ed_toggle "Edit");
                            Js._true);
                      close_btn##.onclick :=
                        Dom.handler (fun _ ->
                            d##setAttribute (js "hidden") (js "hidden");
                            Js._true);
                      rerun_btn##.onclick :=
                        Dom.handler (fun _ ->
                            let code = to_s ed##.value in
                            set_text meta "Running...";
                            clear_children (out :> Dom.node Js.t);
                            send_worker "exec"
                              [
                                ("code", Js.Unsafe.inject (js code));
                                ("resetNamespace", Js.Unsafe.inject Js._true);
                              ]
                              (fun res ->
                                let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
                                let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
                                let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
                                let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
                                let dt_s =
                                  if Js.Optdef.test dt_raw then
                                    Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw)
                                  else ""
                                in
                                set_text meta
                                  ((if ok then "OK" else "Error")
                                  ^ if dt_s <> "" then " (" ^ dt_s ^ ")" else "");
                                clear_children (out :> Dom.node Js.t);
                                if out_s <> "" then append_console_chunk out "py-out-stdout" out_s;
                                if err_s <> "" then append_console_chunk out "py-out-stderr" err_s;
                                if out_s = "" && err_s = "" && ok then
                                  append_console_chunk out "py-out-muted"
                                    "Assertions passed (no stdout). Try Edit -> print(...) to inspect values.\n";
                                render_state_panel ch (Js.Unsafe.get res (js "state")));
                            Js._true);
                      d
                in

                let trigger_run () =
                  let _d = ensure_drawer () in
                  match (!editor_ref, !out_ref, !meta_ref, !chart_ref) with
                  | Some ed, Some out, Some meta, Some ch ->
                      let src = to_s ed##.value in
                      set_text meta (status_text !kernel_state_ref);
                      clear_children (out :> Dom.node Js.t);
                      append_console_chunk out "py-out-muted" "Running in WebAssembly Python worker...\n";
                      let unsub_ref = ref false in
                      add_status_listener (fun st ->
                          if (not !unsub_ref) && !kernel_state_ref <> Ready then
                            set_text meta (status_text st));
                      send_worker "exec"
                        [
                          ("code", Js.Unsafe.inject (js src));
                          ("resetNamespace", Js.Unsafe.inject Js._true);
                        ]
                        (fun res ->
                          unsub_ref := true;
                          let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
                          let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
                          let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
                          let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
                          let dt_s =
                            if Js.Optdef.test dt_raw then
                              Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw)
                            else ""
                          in
                          set_text meta
                            ((if ok then "OK" else "Error")
                            ^ if dt_s <> "" then " (" ^ dt_s ^ ")" else "");
                          clear_children (out :> Dom.node Js.t);
                          if out_s <> "" then append_console_chunk out "py-out-stdout" out_s;
                          if err_s <> "" then append_console_chunk out "py-out-stderr" err_s;
                          if out_s = "" && err_s = "" && ok then
                            append_console_chunk out "py-out-muted"
                              "Assertions passed (no stdout). Click Edit -> print(...) to inspect values.\n";
                          render_state_panel ch (Js.Unsafe.get res (js "state")))
                  | _ -> ()
                in

                run_btn##.onclick :=
                  Dom.handler (fun _ ->
                      trigger_run ();
                      Js._true);

                pg_btn##.onclick :=
                  Dom.handler (fun _ ->
                      let src =
                        match !editor_ref with
                        | Some ed -> to_s ed##.value
                        | None -> orig_code
                      in
                      (try
                         let ss = Js.Unsafe.get Dom_html.window (js "sessionStorage") in
                         ignore
                           (Js.Unsafe.meth_call ss "setItem"
                              [|
                                Js.Unsafe.inject (js "baux_playground_code");
                                Js.Unsafe.inject (js (src ^ "\n"));
                              |])
                       with _ -> ());
                      Dom_html.window##.location##.href := js (get_base_url () ^ "playground.html");
                      Js._true))
  done

let init () : unit =
  (match Js.Opt.to_option (document##getElementById (js "py-playground-root")) with
  | Some root -> mount_playground root
  | None -> ());
  enhance_inline_blocks ()
