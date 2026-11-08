"""Stateful JIT-compiled closed-loop TORAX 2.0.0 2D MEQ + 4-PDE rollout for the 93->19 TCV policy.

Uses ``torax.experimental.make_step_fn`` with 2D ``MEQConfig`` (``GeometryType.MEQ``)
and ``RuntimeParamsProvider.update_provider`` under ``jax.lax.scan`` to evolve
coupled 2D Miller/MEQ flux-surface shaping (kappa, delta_u, delta_l, z_axis, R_in, R_out)
and 4-channel radial transport PDEs (T_i, T_e, n_e, psi) driven by all 19 policy actuators.
"""

import copy
import os
import struct
import sys
from pathlib import Path

try:
    import baux
except ImportError:
    baux = None
import numpy as np

os.environ.setdefault("JAX_PRECISION", "f32")

TOTAL_PARAMS = 160531
TORAX_STEPS = 20
WARMUP_STEPS = 15
MLP_LAYERS = ((93, 256), (256, 256), (256, 256), (256, 19))


def _meq2d():
    from torax._src import constants
    from torax._src.geometry import chease, geometry_loader

    raw = geometry_loader.load_geo_data(
        None, "iterhybrid.mat2cols", geometry_loader.GeometrySource.CHEASE
    )
    cd = chease.CheaseData.model_validate(raw)
    r_maj, b_0 = 6.2, 5.3
    psi_scale = (r_maj**2) * b_0 * 2.0 * np.pi
    psi = cd.psi_chease * psi_scale
    phi = ((cd.rho_tor * r_maj) ** 2) * b_0 * np.pi
    r_in = cd.r_inboard * r_maj
    r_out = cd.r_outboard * r_maj
    rgeom = 0.5 * (r_out + r_in)
    aminor = 0.5 * (r_out - r_in)
    dl_bp = cd.int_dl_over_bp * r_maj / b_0
    l_dict = {"pQ": np.sqrt(np.maximum(0.0, (psi - psi[0]) / (psi[-1] - psi[0])))}
    ly_dict = {
        "rBt": float(rgeom[-1] * b_0),
        "aminor": aminor,
        "rgeom": rgeom,
        "epsilon": aminor / rgeom,
        "TQ": cd.t_rbphi * r_maj * b_0,
        "FB": float(psi[-1]),
        "FA": float(psi[0]),
        "Q0Q": cd.inv_r / r_maj,
        "Q1Q": np.where(dl_bp > 0.0, 1.0 / np.maximum(dl_bp, 1e-12), 0.0),
        "Q2Q": cd.inv_r2 / (r_maj**2),
        "Q3Q": cd.bp2 * (b_0**2) * (4.0 * np.pi**2),
        "Q4Q": cd.grad_psi2 * (psi_scale**2) / (r_maj**2),
        "Q5Q": (cd.grad_psi * psi_scale / r_maj) / (2.0 * np.pi),
        "ItQ": cd.ip_profile / constants.CONSTANTS.mu_0 * r_maj * b_0,
        "deltau": cd.delta_upper,
        "deltal": cd.delta_bottom,
        "kappa": cd.elongation,
        "FtPVQ": phi,
        "zA": 0.0,
        "t": 0.0,
        "lX": 0,
    }
    return l_dict, ly_dict


def _toraxcfg():
    from torax.examples.basic_config import CONFIG

    l_dict, ly_dict = _meq2d()
    cfg = copy.deepcopy(CONFIG)
    cfg["geometry"] = {"model_name": "meq", "L": l_dict, "LY": ly_dict}
    cfg["numerics"]["evolve_ion_heat"] = True
    cfg["numerics"]["evolve_electron_heat"] = True
    cfg["numerics"]["evolve_density"] = True
    cfg["numerics"]["evolve_current"] = True
    cfg["profile_conditions"]["T_e_right_bc"] = 0.5
    cfg["profile_conditions"]["T_i_right_bc"] = 0.5
    cfg["transport"]["core_transport_models"]["prescribed"]["chi_i"] = 1.75
    cfg["transport"]["core_transport_models"]["prescribed"]["chi_e"] = 1.75
    return cfg


def _actnodes(x):
    g = x.geometry._precomputed_geometry
    return [
        x.profile_conditions.I_p,
        x.profile_conditions.T_e_right_bc,
        x.profile_conditions.T_i_right_bc,
        x.sources.generic_heat.P_total,
        x.sources.generic_heat.gaussian_location,
        x.sources.generic_heat.gaussian_width,
        x.sources.generic_heat.electron_heat_fraction,
        x.sources.generic_current.fraction_of_total_current,
        x.sources.generic_current.gaussian_location,
        x.sources.generic_current.gaussian_width,
        x.sources.gas_puff.S_total,
        x.sources.gas_puff.puff_decay_length,
        x.sources.pellet.S_total,
        x.sources.pellet.pellet_deposition_location,
        x.sources.pellet.pellet_width,
        x.sources.generic_particle.S_total,
        x.sources.generic_particle.deposition_location,
        x.sources.generic_particle.particle_width,
        g.elongation,
        g.elongation_face,
        g.delta_upper_face,
        g.delta_lower_face,
        g.delta_face,
        g._z_magnetic_axis,
    ]


def _mapacts(jnp, tx, geo0, a):
    vals = [
        (11.5 + 4.0 * a[0]) * 1e6,
        0.55 + 0.30 * a[1],
        0.55 + 0.30 * a[2],
        (50.0 + 45.0 * a[4]) * 1e6,
        0.25 + 0.18 * a[5],
        0.15 + 0.08 * a[6],
        0.50 + 0.35 * a[7],
        0.20 + 0.16 * a[8],
        0.35 + 0.18 * a[9],
        0.10 + 0.05 * a[10],
        1.0e22 * (1.0 + 0.8 * a[11]),
        0.08 + 0.04 * a[12],
        1.5e22 * (1.0 + 0.8 * a[13]),
        0.80 + 0.10 * a[14],
        0.10 + 0.04 * a[15],
        1.0e22 * (1.0 + 0.8 * a[16]),
        0.25 + 0.15 * a[17],
        0.20 + 0.08 * a[18],
    ]
    su = [tx.TimeVaryingScalarUpdate(value=jnp.atleast_1d(v)) for v in vals]
    k_sc = 1.0 + 0.08 * a[3]
    du_f = jnp.clip(
        geo0.delta_upper_face + 0.10 * a[6] * geo0.rho_face_norm, -0.85, 0.85
    )
    dl_f = jnp.clip(
        geo0.delta_lower_face + 0.10 * a[10] * geo0.rho_face_norm, -0.85, 0.85
    )
    d_f = 0.5 * (du_f + dl_f)
    z_ax = geo0._z_magnetic_axis + 0.06 * a[12]
    return su + [
        geo0.elongation * k_sc,
        geo0.elongation_face * k_sc,
        du_f,
        dl_f,
        d_f,
        z_ax,
    ]


def _lcfs2d(jnp, geo0, a, post):
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 32, endpoint=False, dtype=jnp.float32)
    sin_th = jnp.sin(theta)
    up = (sin_th >= 0.0).astype(jnp.float32)
    k_sc = 1.0 + 0.08 * a[3]
    du = jnp.clip(geo0.delta_upper_face[-1] + 0.10 * a[6], -0.85, 0.85)
    dl = jnp.clip(geo0.delta_lower_face[-1] + 0.10 * a[10], -0.85, 0.85)
    z_ax = geo0._z_magnetic_axis + 0.06 * a[12]
    r_sh = 0.04 * a[15] + 0.015 * (post.beta_N - 1.2) + 0.010 * (post.li3 - 0.76)
    a_sc = 1.0 + 0.03 * a[18]
    r_geo = 0.5 * (geo0.R_out_face[-1] + geo0.R_in_face[-1]) + r_sh
    a_min = 0.5 * (geo0.R_out_face[-1] - geo0.R_in_face[-1]) * a_sc
    kappa = geo0.elongation_face[-1] * k_sc
    d_th = up * du + (1.0 - up) * dl
    r_pts = r_geo + a_min * jnp.cos(theta + jnp.arcsin(d_th) * sin_th)
    z_pts = z_ax + kappa * a_min * sin_th
    sh6 = jnp.array([kappa, du, dl, z_ax, r_geo, a_min], dtype=jnp.float32)
    return r_pts, z_pts, sh6


def _softplus(jnp, val, good: float, bad: float):
    c_good = np.log(0.95 / 0.05)
    c_bad = np.log(0.10 / 0.90)
    t = (val - good) / (bad - good)
    z = c_good + t * (c_bad - c_good)
    return 1.0 / (1.0 + jnp.exp(-jnp.clip(z, -50.0, 50.0)))


def _smoothmax(jnp, comps, alpha: float = -3.0):
    arr = jnp.stack(comps)
    m = jnp.min(arr)
    w = jnp.exp(alpha * (arr - m))
    return jnp.sum(arr * w) / jnp.sum(w)


from python.baux.pytree import flat_mlp


def _mlplogits(jax, jnp, obs, weights):
    return flat_mlp(weights, MLP_LAYERS, obs, act=jax.nn.elu)


def _extractobs(jnp, carry, ref, tau):
    state, post, sh6 = carry
    cp = state.core_profiles
    te_err = (cp.kinetic.T_e.value - ref["te"]) / 2.0
    ti_err = (cp.kinetic.T_i.value - ref["ti"]) / 2.0
    ne_err = (cp.kinetic.n_e.value - ref["ne"]) / 0.2e20
    sh6_scale = jnp.array([0.15, 0.10, 0.10, 0.06, 0.05, 0.05], dtype=te_err.dtype)
    sh6_err = (sh6 - ref["sh6"]) / sh6_scale
    psi_sh = cp.poloidal_flux.psi.value[:11] / 10.0 + jnp.pad(sh6_err, (5, 0))
    scalars = jnp.array(
        [
            (cp.v_loop_lcfs - ref["vloop"]) / 0.2,
            (post.q95 - ref["q95"]) / 0.5,
            (post.q_min - 1.1) / 0.5,
            (post.li3 - ref["li3"]) / 0.1,
            (post.beta_N - ref["beta"]) / 0.2,
            post.f_bootstrap,
            tau,
        ],
        dtype=te_err.dtype,
    )
    return jnp.concatenate([te_err, ti_err, ne_err, psi_sh, scalars])


def _stepreward(jnp, geo0, nxt, a, ref):
    cp = nxt[0].core_profiles
    p = nxt[1]
    r_pts, z_pts, sh6 = _lcfs2d(jnp, geo0, a, p)
    lcfs_rms = jnp.sqrt(
        jnp.mean((r_pts - ref["r_lcfs"]) ** 2 + (z_pts - ref["z_lcfs"]) ** 2)
    )
    prof_rms = jnp.sqrt(
        jnp.mean(((cp.kinetic.T_e.value - ref["te"]) / 1.2) ** 2)
        + jnp.mean(((cp.kinetic.T_i.value - ref["ti"]) / 1.2) ** 2)
        + jnp.mean(((cp.kinetic.n_e.value - ref["ne"]) / 0.12e20) ** 2)
    )
    c_lcfs = _softplus(jnp, 0.15 * lcfs_rms + 0.012 * prof_rms, 0.003, 0.03)
    ip_err = 18.0 * jnp.sqrt(
        ((p.q95 - ref["q95"]) / 0.18) ** 2 + ((p.li3 - ref["li3"]) / 0.08) ** 2
    )
    c_ip = _softplus(jnp, ip_err, 5.0, 30.0)
    c_pf = _softplus(jnp, 500.0 * jnp.abs(p.beta_N - ref["beta"]) / 0.15, 150.0, 720.0)
    u_abs = jnp.abs(a)
    c_volt = _softplus(
        jnp, jnp.mean(jnp.maximum(0.0, u_abs - 0.35) + 0.12 * u_abs * u_abs), 0.05, 0.30
    )
    c_oh = _softplus(
        jnp, 1400.0 * jnp.abs(cp.v_loop_lcfs - ref["vloop"]) / 0.22, 200.0, 2000.0
    )
    return _smoothmax(jnp, [c_lcfs, c_ip, c_pf, c_volt, c_oh], -3.0), sh6


def _refaction(jnp, t: int):
    ramp = min(1.0, (t + 1) / 8.0)
    osc = float(np.sin(t * 0.35))
    return jnp.array(
        [
            -0.20 - 0.20 * ramp + 0.08 * osc,
            0.35 * ramp,
            0.35 * ramp,
            0.25 * ramp - 0.06 * osc,
            -0.40 + 0.95 * ramp + 0.10 * osc,
            -0.15 * ramp,
            0.0,
            0.25 * ramp,
            0.45 * ramp,
            -0.10 * ramp,
            0.0,
            -0.45 * ramp,
            0.0,
            -0.45 * ramp,
            0.0,
            0.0,
            -0.35 * ramp,
            0.0,
            0.0,
        ],
        dtype=jnp.float32,
    )


def _buildref(jnp, geo0, step_one, s_init, p_init):
    s_cur, p_cur = s_init, p_init
    keys = (
        "te",
        "ti",
        "ne",
        "psi5",
        "sh6",
        "r_lcfs",
        "z_lcfs",
        "q95",
        "li3",
        "beta",
        "vloop",
    )
    rows = {k: [] for k in keys}
    for t in range(TORAX_STEPS):
        a_ref = _refaction(jnp, t)
        s_cur, p_cur = step_one(s_cur, p_cur, a_ref)
        r_pts, z_pts, sh6 = _lcfs2d(jnp, geo0, a_ref, p_cur)
        cp = s_cur.core_profiles
        for k, v in (
            ("te", cp.kinetic.T_e.value),
            ("ti", cp.kinetic.T_i.value),
            ("ne", cp.kinetic.n_e.value),
            ("psi5", cp.poloidal_flux.psi.value[::5]),
            ("sh6", sh6),
            ("r_lcfs", r_pts),
            ("z_lcfs", z_pts),
            ("q95", p_cur.q95),
            ("li3", p_cur.li3),
            ("beta", p_cur.beta_N),
            ("vloop", cp.v_loop_lcfs),
        ):
            rows[k].append(v)
    return {k: jnp.stack(v) for k, v in rows.items()}


def _trajstep(ctx, carry, scan_in):
    ref_slice, tau, eps_t = scan_in
    obs = _extractobs(ctx[1], carry, ref_slice, tau)
    u = _mlplogits(ctx[0], ctx[1], obs, ctx[4]) + eps_t
    nxt_s, nxt_p = ctx[3](carry[0], carry[1], ctx[1].tanh(u))
    r, nxt_sh6 = _stepreward(ctx[1], ctx[2], (nxt_s, nxt_p), ctx[1].tanh(u), ref_slice)
    return (nxt_s, nxt_p, nxt_sh6), (obs, u, r)


def _f32geo(jax, jnp, geo):
    return jax.tree.map(
        lambda x: (
            jnp.asarray(x, dtype=jnp.float32)
            if isinstance(x, (np.ndarray, jax.Array, float, np.floating))
            and not isinstance(x, bool)
            and not np.issubdtype(np.asarray(x).dtype, np.bool_)
            else x
        ),
        geo,
    )


class ToraxClosedLoopEnv:
    """Compiled closed-loop TORAX 2.0.0 2D MEQ + 4-PDE trajectory evaluator."""

    _instance = None

    @classmethod
    def shared(cls) -> "ToraxClosedLoopEnv":
        """Return the singleton compiled closed-loop TORAX environment."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import torax
        import torax.experimental as tx

        if hasattr(baux, "verify_upstream_lock"):
            baux.verify_upstream_lock(
                "torax", str(Path(torax.__file__).resolve().parent)
            )

        jax.config.update("jax_enable_x64", False)
        torax_cfg = torax.ToraxConfig.from_dict(_toraxcfg())
        torax_cfg.geometry.__dict__["_precomputed_geometry"] = _f32geo(
            jax, jnp, torax_cfg.geometry._precomputed_geometry
        )
        step_fn = tx.make_step_fn(torax_config=torax_cfg)
        prov = step_fn.runtime_params_provider
        geo0 = prov.geometry._precomputed_geometry
        s0, p0 = tx.get_initial_state_and_post_processed_outputs(step_fn)

        @jax.jit
        def step_one(state, post, a):
            upd = prov.update_provider(_actnodes, _mapacts(jnp, tx, geo0, a))
            return step_fn(state, post, runtime_params_overrides=upd)

        a_warm = jnp.zeros(19, dtype=jnp.float32).at[0].set(-0.2).at[4].set(-0.4)
        s_init, p_init = s0, p0
        for _ in range(WARMUP_STEPS):
            s_init, p_init = step_one(s_init, p_init, a_warm)
        _, _, sh6_init = _lcfs2d(jnp, geo0, a_warm, p_init)

        refs = _buildref(jnp, geo0, step_one, s_init, p_init)
        taus = jnp.linspace(1.0 / TORAX_STEPS, 1.0, TORAX_STEPS, dtype=jnp.float32)
        self._jax = jax
        self._jnp = jnp
        self._geo0 = geo0
        self._step_one = step_one
        self._init_carry = (s_init, p_init, sh6_init)
        self._scan_xs = (refs, taus)
        self._zero_eps = jnp.zeros((TORAX_STEPS, 19), dtype=jnp.float32)
        self._traj = jax.jit(self._rolltraj)
        self._batch = jax.jit(jax.vmap(lambda w: self._rolltraj(w, self._zero_eps)[3]))

    def _rolltraj(self, weights, eps):
        ctx = (self._jax, self._jnp, self._geo0, self._step_one, weights)
        xs = (self._scan_xs[0], self._scan_xs[1], eps)
        (_, _, _), (obs, u, rewards) = self._jax.lax.scan(
            lambda c, x: _trajstep(ctx, c, x), self._init_carry, xs
        )
        return obs, u, rewards, self._jnp.mean(rewards)

    def eval_mean(self, weights: np.ndarray) -> float:
        """Evaluate the mean per-step reward for policy `weights`."""
        w = self._jnp.asarray(weights, dtype=self._jnp.float32)
        return float(self._traj(w, self._zero_eps)[3])

    def eval_return(self, weights: np.ndarray) -> float:
        """Evaluate the 100-step scaled episode return for policy `weights`."""
        return 100.0 * self.eval_mean(weights)

    def eval_batch(self, batch: np.ndarray) -> np.ndarray:
        """Evaluate 100-step scaled episode returns for a batch of policy weights."""
        w = self._jnp.asarray(batch, dtype=self._jnp.float32)
        means = np.asarray(self._batch(w), dtype=np.float64)
        return 100.0 * means

    def eval_traj(self, weights: np.ndarray, eps: np.ndarray):
        """Evaluate a perturbed rollout trajectory and return score, observations, controls, and rewards."""
        w = self._jnp.asarray(weights, dtype=self._jnp.float32)
        e = self._jnp.asarray(eps, dtype=self._jnp.float32).reshape((TORAX_STEPS, 19))
        obs, u, r, mean_r = self._traj(w, e)
        return 100.0 * float(mean_r), np.asarray(obs), np.asarray(u), np.asarray(r)


def _trajreq(env, stdin, stdout) -> bool:
    nbytes = (TOTAL_PARAMS + TORAX_STEPS * 19) * 4
    raw = stdin.read(nbytes)
    if len(raw) < nbytes:
        return False
    buf = np.frombuffer(raw, dtype=np.float32)
    score, obs, u, r = env.eval_traj(buf[:TOTAL_PARAMS], buf[TOTAL_PARAMS:])
    stdout.write(struct.pack("<d", score) + obs.tobytes() + u.tobytes() + r.tobytes())
    stdout.flush()
    return True


def run_worker() -> None:
    """Binary stdin/stdout worker for Rust `baux tune` closed-loop TORAX evaluation."""
    env = ToraxClosedLoopEnv.shared()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        hdr = stdin.read(4)
        if not hdr or len(hdr) < 4:
            break
        (count,) = struct.unpack("<I", hdr)
        if count == 0:
            break
        if count == 0x80000001:
            if not _trajreq(env, stdin, stdout):
                break
            continue
        nbytes = count * TOTAL_PARAMS * 4
        raw = stdin.read(nbytes)
        if len(raw) < nbytes:
            break
        batch = np.frombuffer(raw, dtype=np.float32).reshape((count, TOTAL_PARAMS))
        scores = env.eval_batch(batch)
        stdout.write(struct.pack(f"<{count}d", *scores))
        stdout.flush()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker()
