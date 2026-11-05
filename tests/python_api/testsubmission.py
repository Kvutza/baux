"""Submission semantics at the Python boundary, independent of native builds."""

import numpy as np
import pytest

from baux import Optimizer, TellPredictionError


class _Submission:
    def __init__(self):
        self.accepted = 0
        self.fail_prediction = True
        self.fail_ingestion = False
        self.model = True

    def tell(self, x, y, y_var=None):
        if self.fail_ingestion:
            raise ValueError("ingestion failed")
        self.accepted += len(x)

    def has_surrogate(self):
        return self.model

    def posterior_mu(self, x):
        if self.fail_prediction:
            raise ValueError("prediction failed")
        return np.full((len(x), 1), 3.0)


def test_predictionfailure():
    inner = _Submission()
    opt = Optimizer(2, np.random.default_rng(0), inner)
    x, y = np.ones((2, 2)), np.ones(2)
    with pytest.raises(TellPredictionError, match="observations accepted") as error:
        opt.tell(x, y)
    assert error.value.observations_accepted
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "prediction failed"
    assert inner.accepted == 2
    inner.fail_prediction = False
    np.testing.assert_array_equal(opt.posterior_mu(x), np.full((2, 1), 3.0))
    assert inner.accepted == 2


def test_ingestionfailure():
    inner = _Submission()
    inner.fail_ingestion = True
    opt = Optimizer(2, np.random.default_rng(0), inner)
    with pytest.raises(ValueError, match="ingestion failed"):
        opt.tell(np.ones((2, 2)), np.ones(2))
    assert inner.accepted == 0


def test_returnmean():
    inner = _Submission()
    inner.fail_prediction = False
    opt = Optimizer(2, np.random.default_rng(0), inner)
    result = opt.tell(np.ones((2, 2)), np.ones(2))
    np.testing.assert_array_equal(result, np.full(2, 3.0))


def test_nomodel():
    inner = _Submission()
    inner.model = False
    opt = Optimizer(2, np.random.default_rng(0), inner)
    result = opt.tell(np.ones((2, 2)), np.ones(2))
    np.testing.assert_array_equal(result, np.ones(2))
    assert inner.accepted == 2


def test_shapevalidation():
    inner = _Submission()
    opt = Optimizer(2, np.random.default_rng(0), inner)
    x, y = np.ones((2, 2)), np.ones(1)
    with pytest.raises(ValueError, match="shape"):
        opt.tell(x, y)
    assert x.flags.writeable and y.flags.writeable
    assert inner.accepted == 0
