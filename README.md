<p align="center">
  <a href="https://github.com/Kvutza/baux/blob/assets/baux.pdf">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Kvutza/baux/assets/readme-dark.svg">
      <img src="https://raw.githubusercontent.com/Kvutza/baux/assets/readme-light.svg" width="100%" alt="BAUX: Epistemic Nearest-Neighbor Optimization and Multi-Region Trust-Region Envelopes">
    </picture>
  </a>
</p>

<details>
<summary><b>Copyable Python Quickstart &amp; CLI Commands</b></summary>

```python
import numpy as np
import baux

# 1. Composable transform pipeline with ask/tell interface
bounds = np.column_stack([-5.0 * np.ones(64), 5.0 * np.ones(64)])
opt = baux.create_optimizer(
    bounds,
    baux.chain(
        baux.lhd(16),
        baux.raasp(20),
        baux.enn(k=10),
        baux.thompson(),
        baux.tr_bounds(length_init=0.8, length_min=0.5**7, length_max=1.6),
        baux.regions(4, sharing="ellipsoid-ball", scheduling="incumbent-ucb1"),
    ),
    rng=42,
)
for _ in range(50):
    batch = opt.ask(num_arms=4)
    scores = -np.sum(np.asarray(batch.x) ** 2, axis=-1)
    opt.tell(batch, scores)

# 2. Pure functional state machine (init / step / scan)
state = baux.init(bounds, baux.turbo_enn(), seed=42)
state, best_y = baux.scan(lambda x: -np.sum(x**2, axis=-1), state, steps=50, num_arms=4)
```

```bash
./baux build
./baux test
./baux check
```

</details>
