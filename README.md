<p align="center">
  <a href="https://github.com/Kvutza/baux/blob/assets/baux.pdf">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Kvutza/baux/assets/readme-dark.svg">
      <img src="https://raw.githubusercontent.com/Kvutza/baux/assets/readme-light.svg" width="100%" alt="baux">
    </picture>
  </a>
</p>

<details>
<summary><b>Quickstart &amp; CLI Commands</b></summary>

```python
import numpy as np
import baux

# Stateful ask / tell loop
bounds = np.array([[-5.0, 5.0]] * 16)
opt = baux.create_optimizer(
    bounds=bounds,
    config=baux.turbo_enn(num_init=8),
    rng=np.random.default_rng(0),
)
for _ in range(25):
    x = opt.ask(4)
    opt.tell(x, -np.sum(x**2, axis=1))

# Composable pipeline & pure functional state transitions over PyTrees
spec = baux.pytree_bounds({"w": (4, 4), "b": (4,)}, low=-2.0, high=2.0)
cfg = (
    baux.lhd(8)
    | baux.raasp(12, num_candidates=64)
    | baux.enn(k=5, num_samples=32)
    | baux.ucb(beta=1.5)
    | baux.tr_bounds(length_init=0.8, length_min=0.05, length_max=1.6)
    | baux.regions(2, sharing="ellipsoid_ball", allocation="least_trials")
)
state = baux.init(spec, cfg, seed=0)
state, hist = baux.scan(
    state,
    lambda t: -np.sum(t["w"] ** 2, axis=(-2, -1)) - np.sum(t["b"] ** 2, axis=-1),
    steps=25,
    num_arms=2,
    use_pytree=True,
)

# Standalone ENN surrogate (init / append / fit / posterior / sample)
st = baux.enn.init(x, -np.sum(x**2, axis=1), scale_x=True)
params = baux.enn.fit(st, k=2, num_samples=16, seed=0)
post = baux.enn.posterior(st, x, params)          # post.mu, post.se, post.se_epi, post.se_ale
draws, idx = baux.enn.sample(st, x, params, seeds=[0, 1, 2])
```

```bash
./baux build
./baux test
./baux check
```

</details>
