import copy
import os
import numpy as np

# Configure JAX precision before torax/jax are imported anywhere
os.environ.setdefault("JAX_PRECISION", "f32")


class ToraxRunner:
    """Tokamak core transport runner backed by Google DeepMind TORAX."""

    def __init__(
        self,
        target_te: float = 15.0,
        target_ti: float = 12.0,
        max_steps: int = 1,
    ):
        self.target_te = target_te
        self.target_ti = target_ti
        self.max_steps = max_steps
        # Bounds: [P_heat (MW), gaussian_location (normalized radius)]
        self.bounds = np.array([[10.0, 150.0], [0.0, 0.8]], dtype=float)

    def evaluate(self, p_heat_mw: float, gaussian_loc: float) -> float:
        import torax
        from torax.examples.basic_config import CONFIG

        cfg_dict = copy.deepcopy(CONFIG)
        cfg_dict["sources"]["generic_heat"]["P_total"] = float(p_heat_mw * 1e6)
        cfg_dict["sources"]["generic_heat"]["gaussian_location"] = float(gaussian_loc)
        torax_cfg = torax.ToraxConfig.from_dict(cfg_dict)
        history = torax.run_simulation(
            torax_cfg, progress_bar=False, max_steps=self.max_steps
        )
        last_profile = history.core_profiles[-1]
        central_te = float(last_profile.kinetic.T_e.value[0])
        central_ti = float(last_profile.kinetic.T_i.value[0])

        loss = (
            (central_te - self.target_te) ** 2
            + 0.5 * (central_ti - self.target_ti) ** 2
            + 0.001 * (p_heat_mw**2)
        )
        return float(-loss)
