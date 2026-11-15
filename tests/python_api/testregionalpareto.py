from dataclasses import replace

import numpy as np
from baux import _rust, create_optimizer, turbo_enn
from baux.turbo.config import ParetoAcquisitionConfig, RegionalSearchPolicy
from baux.turbo.config.encode import encode


def testregionalpareto():
    acquisitions = (ParetoAcquisitionConfig(), ParetoAcquisitionConfig())
    policy = RegionalSearchPolicy(count=2, acquisitions=acquisitions)
    assert policy.acquisitions == acquisitions
    config = replace(turbo_enn(num_init=2), trust_region=policy)
    cfg_encoded = encode(config)
    assert cfg_encoded["region_acquisitions"] == ["pareto", "pareto"]

    # Test via bo.create_optimizer wrapper
    optimizer = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=config,
        rng=np.random.default_rng(85),
    )
    points = optimizer.ask(2)
    values = np.column_stack((points[:, 0], 1.0 - points[:, 0]))
    optimizer.tell(points, values)
    assert optimizer.ask(2).shape == (2, 1)

    # Test via pure functional _rust.create_optimizer
    opt = _rust.create_optimizer(
        np.array([[0.0, 1.0]]),
        "enn",
        10,
        2,
        2,
        85,
        cfg=cfg_encoded,
    )
    points_rust, opt = opt.ask(2)
    values_rust = np.column_stack((points_rust[:, 0], 1.0 - points_rust[:, 0]))
    opt = opt.tell(points_rust, values_rust)
    next_points, opt = opt.ask(2)
    assert next_points.shape == (2, 1)
