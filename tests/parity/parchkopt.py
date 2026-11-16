from __future__ import annotations

from .parity_types import ParityCase, ParityReport


def run_parity(
    report: ParityReport,
    name: str,
    config,
) -> None:
    import numpy as np

    from .parchksob import append_case

    try:
        from baux._rust import BayesOptimizer  # noqa: F401
    except ImportError:
        append_case(
            report,
            name=name,
            endpoint="baux.init",
            backend="rust",
            reason="Rust not available",
        )
        return

    def _run() -> None:
        import baux

        from .optimizer_checks import check_opt, make_optimizer

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
        opt = make_optimizer(bounds, config, seed=47)
        check_opt(opt, bounds)

        rng = np.random.default_rng(47)
        state = baux.init(bounds, config, seed=int(rng.integers(2**63 - 1)))
        state, batch = baux.ask(state, num_arms=2)
        assert batch.points.shape == (2, 2)
        y = np.array([[0.1], [0.2]])
        state, _ = baux.tell(state, batch.points, y)
        assert state.num_observations == 2

    try:
        _run()
    except Exception as e:
        report.cases.append(
            ParityCase(
                name=name,
                endpoint="baux.init",
                passed=False,
                error=str(e),
                backend="rust",
            )
        )
        report.failed += 1
    else:
        report.cases.append(
            ParityCase(name=name, endpoint="baux.init", passed=True, backend="rust")
        )
        report.passed += 1
    report.total += 1
