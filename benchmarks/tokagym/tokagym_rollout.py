"""Stateful JIT-compiled closed-loop TokaGym TCV rollout for the 11->8 policy.

Uses the unmodified upstream EPFL/MIT ``toka_gym`` simulation environment
(``build_gym``, ``TCVGym``, ``TCVEnv``) and evaluation harness (``RewardModel``,
``ESFitnessModel``) verified against ``research/upstream/tokagym.lock.toml``,
while defining the neural network policy (``11 -> 256 -> 256 -> 256 -> 8``) on
the BAUX side.

Academic & Open-Source Attribution:
- Christian Theiler et al., "Progress and innovations in the TCV tokamak
  research programme", Nuclear Fusion 66 (2026) 116007,
  https://doi.org/10.1088/1741-4326/ae6d12 (Section 8, Figure 13, Ref. [124]).
- Allen M. Wang et al., "Learning plasma dynamics and robust rampdown
  trajectories with predict-first experiments at TCV", Nature Communications
  16 (2025) 8877, https://doi.org/10.1038/s41467-025-63917-3,
  Zenodo archive https://doi.org/10.5281/zenodo.16621120.
"""

import copy
import os
import struct
import sys
import types
from pathlib import Path
import numpy as np

os.environ.setdefault("JAX_PRECISION", "f32")

OBS_DIM = 11
ACTION_DIM = 8
HIDDEN_DIM = 256
TOTAL_PARAMS = 136712
TOKAGYM_STEPS = 20
MLP_LAYERS = (
    (OBS_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, ACTION_DIM),
)
OBS_KEYS = (
    "kappa",
    "a_minor",
    "delta",
    "fir_lads_vol",
    "Wtot",
    "li",
    "IP",
    "vc_minus_vb",
    "beta_p",
    "greenwald_frac",
    "vgr",
)


class _Stubitem:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return self

    def __getattr__(self, name):
        return _Stubitem


class _Stubmod(types.ModuleType):
    def __getattr__(self, name):
        return _Stubitem


STUB_MODULES = (
    "jax_dataloader",
    "panel",
    "typer",
    "wandb",
    "holoviews",
    "hvplot",
    "hvplot.xarray",
    "plotly",
    "plotly.graph_objects",
    "plotly.colors",
    "plotly.subplots",
    "ipdb",
)


def _stubmods() -> None:
    os.environ["ML_DATA_DUMP"] = "/tmp"
    import functools
    import equinox as eqx
    import jax

    eqx.static_field = functools.partial(eqx.field, static=True)
    jax.devices("cpu")
    try:
        import jax_plugins.mps.ops as mpsops

        orig = getattr(mpsops, "_gelu_original", None)
    except Exception:
        orig = None
    if orig is not None:
        jax.nn.gelu = orig
        import jax._src.nn.functions as jfunctions

        jfunctions.gelu = orig
    for modname in STUB_MODULES:
        sys.modules.setdefault(modname, _Stubmod(modname))


def _ensuretokagym() -> Path:
    try:
        import baux
    except ImportError:
        baux = None
    if baux is not None and hasattr(baux, "verify_upstream_lock"):
        baux.verify_upstream_lock("tokagym")
    root = Path(__file__).resolve().parents[2] / ".cache/baux/upstream/tokagym/TokaGym"
    _stubmods()
    cpath = str(root / "submodules/contrax")
    rpath = str(root)
    if cpath not in sys.path:
        sys.path.insert(0, cpath)
    if rpath not in sys.path:
        sys.path.insert(0, rpath)
    return root


_ensure_tokagym = _ensuretokagym


def kaiming_bounds() -> np.ndarray:
    """Per-parameter Kaiming uniform intervals [-bound, +bound] for 11->256->256->256->8."""
    blocks = []
    for fan_in, fan_out in MLP_LAYERS:
        wb = float(np.sqrt(3.0 / fan_in))
        bb = float(1.0 / np.sqrt(fan_in))
        blocks.append(np.tile([-wb, wb], (fan_in * fan_out, 1)))
        blocks.append(np.tile([-bb, bb], (fan_out, 1)))
    return np.vstack(blocks).astype(np.float64)


def _buildnorm(root):
    import xarray as xr
    from toka_gym.types import EpisodeData, build_standardizers_np

    ds = xr.open_dataset(root / "data/initial_training_validation.nc")
    hl = xr.open_dataset(root / "data/cached_high_level.nc")
    obs_dict = {
        k.removeprefix("targ."): ds[k].values
        for k in ds.data_vars
        if k.startswith("targ.")
    }
    ep = EpisodeData(
        observations=obs_dict,
        controls={"NBI": hl["NBI"].values},
        time=ds["time"].values,
        episode_label=ds["shot"].values,
    )
    return build_standardizers_np(ep)


def _fitcfg(config: dict) -> dict:
    fit_cfg = copy.deepcopy(config["fitness"])
    fit_cfg.setdefault(
        "deenergize",
        {"existence_is_pain_cost": 5.0, "ip_weight": 1.0, "wtot_weight": 1.0},
    )
    return fit_cfg


from python.baux.pytree import flat_mlp


def _mlplogits(jax, jnp, obs, weights):
    return flat_mlp(weights, MLP_LAYERS, obs, act=jax.nn.elu)


def _obsvec(jnp, env, stateobs):
    obs_pol, _ = env.get_obs(stateobs.state, stateobs.obs)
    return jnp.stack(
        [jnp.asarray(obs_pol[k], dtype=jnp.float32).reshape(()) for k in OBS_KEYS]
    )


def _trajstep(ctx, stateobs, eps_t):
    jax, jnp, env, key, weights = ctx
    obs = _obsvec(jnp, env, stateobs)
    u = _mlplogits(jax, jnp, obs, weights) + eps_t
    step_out = env.step_env(key, stateobs, jnp.tanh(u))
    nxt_stateobs = step_out[2]
    reward = jnp.asarray(step_out[4], dtype=jnp.float32).reshape(())
    return nxt_stateobs, (obs, u, reward)


class TokaGymClosedLoopEnv:
    """Compiled closed-loop TokaGym TCV trajectory evaluator."""

    _instance = None

    @classmethod
    def shared(cls, scenario: str = "170kA") -> "TokaGymClosedLoopEnv":
        if cls._instance is None or cls._instance.scenario != scenario:
            cls._instance = cls(scenario=scenario)
        return cls._instance

    def __init__(self, scenario: str = "170kA"):
        root = _ensuretokagym()
        import jax
        import jax.numpy as jnp
        from toka_gym.es_fitness_model import ESFitnessModel
        from toka_gym.tcv_env import TCVEnv
        import toka_gym.tcv_gym as tg

        jax.config.update("jax_enable_x64", False)
        cpu = jax.devices("cpu")[0]
        cfg_path = str(root / f"config/{scenario}.yaml")
        with jax.default_device(cpu):
            gym = tg.build_gym(cfg_path)
            obs_norm, state_norm = _buildnorm(root)
            env = TCVEnv(gym, obs_norm, state_norm)
            fitness_model = ESFitnessModel(_fitcfg(gym.config))
            init_stateobs, _ = gym.reset(jax.random.PRNGKey(0))
            key = jax.random.PRNGKey(1)
            zero_eps = jnp.zeros((TOKAGYM_STEPS, ACTION_DIM), dtype=jnp.float32)

        self.scenario = scenario
        self.gym = gym
        self.env = env
        self.fitness_model = fitness_model
        self._jax = jax
        self._jnp = jnp
        self._cpu = cpu
        self._key = key
        self._init_stateobs = init_stateobs
        self._zero_eps = zero_eps
        self._traj = jax.jit(self._rolltraj)
        self._batch = jax.jit(jax.vmap(lambda w: self._rolltraj(w, self._zero_eps)[3]))

    def _rolltraj(self, weights, eps):
        ctx = (self._jax, self._jnp, self.env, self._key, weights)
        _, (obs, u, rewards) = self._jax.lax.scan(
            lambda s, e: _trajstep(ctx, s, e), self._init_stateobs, eps
        )
        return obs, u, rewards, self._jnp.mean(rewards)

    def eval_mean(self, weights: np.ndarray) -> float:
        with self._jax.default_device(self._cpu):
            w = self._jnp.asarray(weights, dtype=self._jnp.float32)
            return float(self._traj(w, self._zero_eps)[3])

    def eval_return(self, weights: np.ndarray) -> float:
        return float(TOKAGYM_STEPS) * self.eval_mean(weights)

    def eval_batch(self, batch: np.ndarray) -> np.ndarray:
        with self._jax.default_device(self._cpu):
            w = self._jnp.asarray(batch, dtype=self._jnp.float32)
            means = np.asarray(self._batch(w), dtype=np.float64)
        return float(TOKAGYM_STEPS) * means

    def eval_traj(self, weights: np.ndarray, eps: np.ndarray):
        with self._jax.default_device(self._cpu):
            w = self._jnp.asarray(weights, dtype=self._jnp.float32)
            e = self._jnp.asarray(eps, dtype=self._jnp.float32).reshape(
                (TOKAGYM_STEPS, ACTION_DIM)
            )
            obs, u, r, mean_r = self._traj(w, e)
        return (
            float(TOKAGYM_STEPS) * float(mean_r),
            np.asarray(obs),
            np.asarray(u),
            np.asarray(r),
        )


def _trajreq(env, stdin, stdout) -> bool:
    nbytes = (TOTAL_PARAMS + TOKAGYM_STEPS * ACTION_DIM) * 4
    raw = stdin.read(nbytes)
    if len(raw) < nbytes:
        return False
    buf = np.frombuffer(raw, dtype=np.float32)
    score, obs, u, r = env.eval_traj(buf[:TOTAL_PARAMS], buf[TOTAL_PARAMS:])
    stdout.write(struct.pack("<d", score) + obs.tobytes() + u.tobytes() + r.tobytes())
    stdout.flush()
    return True


def run_worker() -> None:
    """Binary stdin/stdout worker for closed-loop TokaGym evaluation."""
    env = TokaGymClosedLoopEnv.shared()
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
