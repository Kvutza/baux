"""Couples the 19 TCV coil actions and plasma current to the TORAX core transport simulator.

Academic & Open-Source Attribution:
- David Pfau et al., "Wasserstein Policy Optimization", ICML 2025, arXiv:2505.00663
  (Appendix B.3, Table 1, Table 6, Equation 11), https://github.com/google-deepmind/acme
  (acme/agents/jax/wpo/, commit 2e09e3ddecf4a1ba783688c92aa493c596a2beec).
- Brendan D. Tracey et al., "Towards practical reinforcement learning for tokamak
  magnetic control", Fusion Engineering and Design 200 (2024) 114161, arXiv:2307.11546
  (Section 3, Equations 1-3).
- Jonathan Citrin et al., "TORAX: A Fast and Differentiable Tokamak Transport Simulator
  in JAX", arXiv:2406.06718, https://github.com/google-deepmind/torax (v2.0.0).
"""

import copy
from dataclasses import dataclass
import os
import numpy as np

os.environ.setdefault("JAX_PRECISION", "f32")

OBS_DIM = 93
ACTION_DIM = 19
HIDDEN_DIM = 256
TOTAL_PARAMS = 160531
STEPS = 100
MLP_LAYERS = (
    (OBS_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, HIDDEN_DIM),
    (HIDDEN_DIM, ACTION_DIM),
)

ZETA = -2.9444389791664403  # -ln(19), so Bad maps to 0.1
LCFS_GOOD = 0.001
LCFS_BAD = 0.01
IP_GOOD = 100.0
IP_BAD = 2000.0
PF_GOOD = 100.0
PF_BAD = 50.0
VOLT_GOOD = 0.0
VOLT_BAD = 1.0
OH_GOOD = 50.0
OH_BAD = 1050.0
COMBINER_ALPHA = -3.0


def softplus_component(val: float, good: float, bad: float) -> float:
    """WPO Table 6 SoftPlus component mapping good -> ~0.95 and bad -> 0.1."""
    scale = ZETA * (val - good) / (bad - good)
    sigma = 1.0 / (1.0 + float(np.exp(-scale)))
    return float(np.clip(2.0 * sigma, 0.0, 1.0))


def smooth_max(comps, alpha: float = COMBINER_ALPHA) -> float:
    """WPO Table 6 SmoothMax combiner with negative alpha weighting."""
    arr = np.asarray(comps, dtype=float)
    weights = np.exp(alpha * arr)
    return float(np.sum(arr * weights) / np.sum(weights))


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.exp(x) - 1.0)


@dataclass(frozen=True)
class PolicyLayers:
    """160,531-parameter MLP weights for 93->256->256->256->19 TCV policy."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    w4: np.ndarray
    b4: np.ndarray

    @classmethod
    def kaiming_bounds(cls) -> np.ndarray:
        """Per-parameter Kaiming uniform intervals [-bound, +bound] derived from MLP_LAYERS."""
        bounds = np.empty((TOTAL_PARAMS, 2), dtype=np.float64)
        offset = 0
        for fan_in, fan_out in MLP_LAYERS:
            w_bound = np.sqrt(3.0 / float(fan_in))
            b_bound = 1.0 / np.sqrt(float(fan_in))
            w_len = fan_in * fan_out
            bounds[offset : offset + w_len, 0] = -w_bound
            bounds[offset : offset + w_len, 1] = w_bound
            offset += w_len
            bounds[offset : offset + fan_out, 0] = -b_bound
            bounds[offset : offset + fan_out, 1] = b_bound
            offset += fan_out
        return bounds

    @classmethod
    def sample_kaiming(cls, rng: np.random.Generator) -> np.ndarray:
        """Sample a flat parameter vector from the layer-wise Kaiming uniform initializer."""
        bounds = cls.kaiming_bounds()
        return rng.uniform(bounds[:, 0], bounds[:, 1])

    @classmethod
    def unpack(cls, weights: np.ndarray) -> "PolicyLayers":
        assert len(weights) == TOTAL_PARAMS, (
            f"Expected {TOTAL_PARAMS} params, got {len(weights)}"
        )
        idx = 0
        w1 = weights[idx : idx + 93 * 256].reshape((93, 256))
        idx += 93 * 256
        b1 = weights[idx : idx + 256]
        idx += 256
        w2 = weights[idx : idx + 256 * 256].reshape((256, 256))
        idx += 256 * 256
        b2 = weights[idx : idx + 256]
        idx += 256
        w3 = weights[idx : idx + 256 * 256].reshape((256, 256))
        idx += 256 * 256
        b3 = weights[idx : idx + 256]
        idx += 256
        w4 = weights[idx : idx + 256 * 19].reshape((256, 19))
        idx += 256 * 19
        b4 = weights[idx : idx + 19]
        return cls(w1, b1, w2, b2, w3, b3, w4, b4)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        h1 = _elu(np.dot(obs, self.w1) + self.b1)
        h2 = _elu(np.dot(h1, self.w2) + self.b2)
        h3 = _elu(np.dot(h2, self.w3) + self.b3)
        raw_out = np.dot(h3, self.w4) + self.b4
        return np.tanh(raw_out)


def _initstate() -> np.ndarray:
    state = np.zeros(OBS_DIM, dtype=np.float64)
    idx72 = np.arange(1, 73, dtype=np.float64)
    state[:72] = np.sin(idx72 * 0.17) * 0.06
    idx16 = np.arange(16, dtype=np.float64)
    state[72:88] = 0.12 + 0.02 * np.cos(idx16 * 0.5)
    state[88] = 0.25
    state[89] = 0.10
    state[90] = 0.0
    state[91] = 0.35
    state[92] = 0.35
    return state


def _phasemat() -> tuple[np.ndarray, np.ndarray]:
    ii = np.arange(72, dtype=np.float64)[:, None]
    jj = np.arange(ACTION_DIM, dtype=np.float64)[None, :]
    phase = np.cos((ii * 7.0 + jj * 13.0) * 0.11) / float(ACTION_DIM)
    mode = np.sin(np.arange(1, 73, dtype=np.float64) * 0.17)
    return phase, mode


def _stepreward(state: np.ndarray, action: np.ndarray) -> float:
    lcfs_dist = 0.05 * float(np.sqrt(np.mean(state[:72] ** 2)))
    c_lcfs = softplus_component(lcfs_dist, LCFS_GOOD, LCFS_BAD)
    ip_err = 1000.0 * abs(float(state[91]))
    c_ip = softplus_component(ip_err, IP_GOOD, IP_BAD)
    pf_val = 1000.0 * float(np.mean(np.abs(state[72:88])))
    c_pf = softplus_component(pf_val, PF_GOOD, PF_BAD)
    u_abs = np.abs(action)
    volt_level = float(np.mean(np.maximum(0.0, u_abs - 0.15) + 0.5 * u_abs * u_abs))
    c_volt = softplus_component(volt_level, VOLT_GOOD, VOLT_BAD)
    oh_diff = 1000.0 * abs(float(state[88] - state[89]))
    c_oh = softplus_component(oh_diff, OH_GOOD, OH_BAD)
    return smooth_max([c_lcfs, c_ip, c_pf, c_volt, c_oh], COMBINER_ALPHA)


def _stepdyn(
    state: np.ndarray,
    action: np.ndarray,
    step: int,
    phase: np.ndarray,
    mode: np.ndarray,
) -> np.ndarray:
    nxt = state.copy()
    pf_mean = float(np.mean(action[:16]))
    harmonic = phase @ action
    nxt[:72] = 1.018 * state[:72] + mode * (0.0045 - 0.045 * pf_mean) - 0.04 * harmonic
    nxt[72:88] = 0.965 * state[72:88] + 0.001 + 0.045 * action[:16]
    nxt[88] = 1.008 * state[88] + 0.006 - 0.06 * action[16]
    nxt[89] = 0.985 * state[89] - 0.003 + 0.06 * action[17]
    nxt[90] = 0.92 * state[90] + 0.10 * action[18]
    oh_drive = 0.5 * (action[16] + action[17]) + 0.4 * action[18]
    nxt[91] = 1.015 * state[91] + 0.015 - 0.16 * oh_drive
    nxt[92] = (float(step + 1) * state[92] + nxt[91]) / float(step + 2)
    return nxt


def _evalnumpy(weights: np.ndarray) -> float:
    policy = PolicyLayers.unpack(np.asarray(weights, dtype=np.float64))
    state = _initstate()
    phase, mode = _phasemat()
    total = 0.0
    for step in range(STEPS):
        action = policy.forward(state)
        total += _stepreward(state, action)
        state = _stepdyn(state, action, step, phase, mode)
    return float(total)


def _torchfwd(policy: PolicyLayers, state: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    obs = torch.from_numpy(state)
    h1 = F.elu(obs @ torch.from_numpy(policy.w1) + torch.from_numpy(policy.b1))
    h2 = F.elu(h1 @ torch.from_numpy(policy.w2) + torch.from_numpy(policy.b2))
    h3 = F.elu(h2 @ torch.from_numpy(policy.w3) + torch.from_numpy(policy.b3))
    raw = h3 @ torch.from_numpy(policy.w4) + torch.from_numpy(policy.b4)
    act = torch.tanh(raw)
    return act.numpy()


def _evaltorch(weights: np.ndarray) -> float:
    policy = PolicyLayers.unpack(np.asarray(weights, dtype=np.float64))
    state = _initstate()
    phase, mode = _phasemat()
    total = 0.0
    for step in range(STEPS):
        action = _torchfwd(policy, state)
        total += _stepreward(state, action)
        state = _stepdyn(state, action, step, phase, mode)
    return float(total)


def _jaxfwd(policy: PolicyLayers, state: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    obs = jnp.asarray(state, dtype=jnp.float32)
    h1 = jax.nn.elu(jnp.dot(obs, jnp.asarray(policy.w1)) + jnp.asarray(policy.b1))
    h2 = jax.nn.elu(jnp.dot(h1, jnp.asarray(policy.w2)) + jnp.asarray(policy.b2))
    h3 = jax.nn.elu(jnp.dot(h2, jnp.asarray(policy.w3)) + jnp.asarray(policy.b3))
    act = jnp.tanh(jnp.dot(h3, jnp.asarray(policy.w4)) + jnp.asarray(policy.b4))
    return np.asarray(act, dtype=np.float64)


def _evaljax(weights: np.ndarray) -> float:
    import jax

    policy = PolicyLayers.unpack(np.asarray(weights, dtype=np.float32))
    state = _initstate()
    phase, sinterm = _phasemat()
    total = 0.0
    with jax.default_device(jax.devices("cpu")[0]):
        for step in range(STEPS):
            action = _jaxfwd(policy, state)
            total += _stepreward(state, action)
            state = _stepdyn(state, action, step, phase, sinterm)
    return float(total)


def eval_table6(weights: np.ndarray, backend: str = "numpy") -> float:
    """Evaluates the 100-step WPO Table 6 TCV trajectory across numpy/torch/jax."""
    if backend == "numpy":
        return _evalnumpy(weights)
    if backend == "torch":
        return _evaltorch(weights)
    if backend == "jax":
        return _evaljax(weights)
    raise ValueError(f"Unsupported backend: {backend}")


eval_wpo_table6 = eval_table6


def make_obs(profile, prev_action: np.ndarray) -> np.ndarray:
    """Builds 93 continuous tokamak observations from TORAX core profile."""
    te_vals = np.asarray(profile.kinetic.T_e.value, dtype=float) / 15.0
    ti_vals = np.asarray(profile.kinetic.T_i.value, dtype=float) / 12.0
    ne_vals = np.asarray(profile.kinetic.n_e.value, dtype=float) / 1e20
    v_loop = np.array([float(profile.v_loop_lcfs) * 0.1], dtype=float)
    psi_val = np.array([float(profile.psi_norm[-1])], dtype=float)
    act_feed = np.asarray(prev_action[:16], dtype=float)
    return np.concatenate([te_vals, ti_vals, ne_vals, v_loop, psi_val, act_feed])


class ToraxTcvBridge:
    """Bridges the 93->19 TCV policy to TORAX PDE core transport."""

    def __init__(self, target_te: float = 15.0, target_ti: float = 12.0):
        self.target_te = target_te
        self.target_ti = target_ti

    def build_config(self, action: np.ndarray):
        from torax.examples.basic_config import CONFIG

        delta_ip = float(action[0]) * 1e5
        base_ip = 15000000.0
        i_p = float(np.clip(base_ip + delta_ip, 1e6, 2e7))
        heat_power = float(np.clip(80.0 + float(action[1]) * 20.0, 10.0, 150.0)) * 1e6
        heat_loc = float(np.clip(0.2 + float(action[2]) * 0.1, 0.0, 0.8))

        cfg_dict = copy.deepcopy(CONFIG)
        cfg_dict["profile_conditions"]["I_p"] = i_p
        cfg_dict["sources"]["generic_heat"]["P_total"] = heat_power
        cfg_dict["sources"]["generic_heat"]["gaussian_location"] = heat_loc
        return cfg_dict

    def calc_reward(
        self, central_te: float, central_ti: float, action: np.ndarray
    ) -> float:
        lcfs_dist = 0.002 * abs(central_te - self.target_te)
        c_lcfs = softplus_component(lcfs_dist, LCFS_GOOD, LCFS_BAD)

        ip_err = 250.0 * abs(central_ti - self.target_ti) + 100.0 * abs(
            float(action[0])
        )
        c_ip = softplus_component(ip_err, IP_GOOD, IP_BAD)

        pf_val = 120.0 + 50.0 * float(np.mean(np.abs(action[:16])))
        c_pf = softplus_component(pf_val, PF_GOOD, PF_BAD)

        u_abs = np.abs(action)
        volt_level = float(np.mean(np.maximum(0.0, u_abs - 0.15) + 0.5 * u_abs * u_abs))
        c_volt = softplus_component(volt_level, VOLT_GOOD, VOLT_BAD)

        oh_diff = 150.0 + 1000.0 * abs(float(action[16] - action[17]))
        c_oh = softplus_component(oh_diff, OH_GOOD, OH_BAD)

        return smooth_max([c_lcfs, c_ip, c_pf, c_volt, c_oh], COMBINER_ALPHA)

    def eval_rollout(self, weights: np.ndarray, rounds: int = 20) -> float:
        from benchmarks.torax.torax_rollout import ToraxClosedLoopEnv

        _ = rounds
        return ToraxClosedLoopEnv.shared().eval_mean(weights)

    def eval_policy(self, weights: np.ndarray, steps: int = 20) -> float:
        return self.eval_rollout(weights, rounds=steps)
