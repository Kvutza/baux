from dataclasses import replace

import numpy as np
from baux import create_optimizer, turbo_enn
from baux.turbo.config import ParetoAcquisitionConfig, RegionalSearchPolicy
from baux.turbo.config.encode import encode


def testregionalpareto():
    acquisitions = (ParetoAcquisitionConfig(), ParetoAcquisitionConfig())
    policy = RegionalSearchPolicy(count=2, acquisitions=acquisitions)
    assert policy.acquisitions == acquisitions
    config = replace(turbo_enn(num_init=2), trust_region=policy)
    assert encode(config)["region_acquisitions"] == ["pareto", "pareto"]

    optimizer = create_optimizer(
        bounds=np.array([[0.0, 1.0]]),
        config=config,
        rng=np.random.default_rng(85),
    )
    points = optimizer.ask(2)
    values = np.column_stack((points[:, 0], 1.0 - points[:, 0]))
    optimizer.tell(points, values)
    assert optimizer.ask(2).shape == (2, 1)
