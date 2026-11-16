"""Submission semantics at the Python boundary, independent of native builds."""

from __future__ import annotations

from dataclasses import replace

import baux
import numpy as np
import pytest
from baux import AskId, OptimizerConfig, OptimizerState, Telemetry
from baux.turbo.config import RegionalSearchPolicy, turbo_zero


class _Submission:
    def __init__(self):
        self.accepted = 0
        self.fail_prediction = False
        self.fail_ingestion = False
        self.model = True
        self.pending = []
        self._x = []
        self._y = []

    def tell(self, x, y, seed=0, y_var=None):
        if self.fail_ingestion:
            raise ValueError("ingestion failed")
        self.accepted += len(x)
        self._x.append(np.asarray(x))
        self._y.append(np.asarray(y))
        return self

    def import_observations(self, x, y, seed=0, y_var=None):
        if self.fail_ingestion:
            raise ValueError("ingestion failed")
        self.accepted += len(x)
        self._x.append(np.asarray(x))
        self._y.append(np.asarray(y))
        return self

    def has_surrogate(self):
        return self.model

    def posterior_mu(self, x):
        if self.fail_prediction:
            raise ValueError("prediction failed")
        return np.full((len(x), 1), 3.0)

    def telemetry(self):
        return Telemetry(0.0, 0.0, 0.0, 0.0, 0)

    def pending_ids(self):
        return tuple(self.pending)

    def num_observations(self):
        return self.accepted

    def num_regions(self):
        return 1

    def tr_length(self):
        return 0.8

    def init_progress(self):
        return None

    def x_obs(self):
        return np.vstack(self._x) if self._x else np.empty((0, 2))

    def y_obs(self):
        return np.vstack(self._y) if self._y else np.empty((0, 1))

    def incumbent_x(self):
        return self._x[0][0] if self._x else None

    def lengthscales(self):
        return None

    def complete_ids(self, ids, y, seed=0, y_var=None):
        self.pending = [i for i in self.pending if i not in ids]
        self.accepted += len(ids)
        return self

    def cancel_ids(self, ids, seed=0):
        self.pending = [i for i in self.pending if i not in ids]
        return self

    def restart_region(self, slot, center, seed=0):
        abandoned = tuple(self.pending)
        self.pending = []
        return self, abandoned


def _makestate(inner: _Submission, identified: bool = False) -> OptimizerState:
    raw_pending = inner.pending_ids()
    return OptimizerState(
        dim=2,
        bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        config=OptimizerConfig(),
        seed=0,
        epoch=0,
        spec=None,
        identified=identified,
        num_observations=inner.num_observations(),
        num_regions=inner.num_regions(),
        tr_length=inner.tr_length(),
        init_progress=inner.init_progress(),
        pending_ids=None if not raw_pending else tuple(int(i) for i in raw_pending),
        ingestion_failure=None,
        telemetry=Telemetry(0.0, 0.0, 0.0, 0.0, 0),
        _opt=inner,
    )


def test_predictionfailure():
    inner = _Submission()
    inner.fail_prediction = True
    s0 = _makestate(inner)
    x, y = np.ones((2, 2)), np.ones(2)
    with pytest.raises(Exception, match="observations accepted"):
        baux.tell(s0, x, y)
    assert inner.accepted == 2


def test_ingestionfailure():
    inner = _Submission()
    inner.fail_ingestion = True
    s0 = _makestate(inner)
    with pytest.raises(ValueError, match="ingestion failed"):
        baux.tell(s0, np.ones((2, 2)), np.ones(2))
    assert inner.accepted == 0
    assert s0.num_observations == 0


def test_returnmean():
    inner = _Submission()
    inner.fail_prediction = False
    s0 = _makestate(inner)
    s1, result = baux.tell(s0, np.ones((2, 2)), np.ones(2))
    np.testing.assert_array_equal(result, np.full(2, 3.0))
    assert s0.num_observations == 0
    assert s1.num_observations == 2


def test_nomodel():
    inner = _Submission()
    inner.model = False
    s0 = _makestate(inner)
    s1, result = baux.tell(s0, np.ones((2, 2)), np.ones(2))
    np.testing.assert_array_equal(result, np.ones(2))
    assert s0.num_observations == 0
    assert s1.num_observations == 2
    assert inner.accepted == 2


def test_shapevalidation():
    inner = _Submission()
    s0 = _makestate(inner)
    x, y = np.ones((2, 2)), np.ones(1)
    with pytest.raises(ValueError, match="shape"):
        baux.tell(s0, x, y)
    assert x.flags.writeable and y.flags.writeable
    assert inner.accepted == 0
    assert s0.num_observations == 0


def test_importobservations():
    inner = _Submission()
    inner.model = False
    s0 = _makestate(inner)
    s1, result = baux.import_observations(s0, np.ones((2, 2)), np.ones(2))
    np.testing.assert_array_equal(result, np.ones(2))
    assert s0.num_observations == 0
    assert s1.num_observations == 2
    assert inner.accepted == 2


def test_completecancel():
    inner = _Submission()
    inner.pending = [1, 2, 3]
    s0 = _makestate(inner)
    assert s0.pending_ids == (1, 2, 3)

    s1 = baux.complete(s0, [AskId(1)], np.array([10.0]))
    assert s0.pending_ids == (1, 2, 3)
    assert s1.pending_ids == (2, 3)

    s2 = baux.cancel(s1, [AskId(2)])
    assert s2.pending_ids == (3,)


def test_restartregion():
    inner = _Submission()
    inner.pending = [7]
    s0 = _makestate(inner, identified=True)
    center = np.zeros(2)
    s1, abandoned = baux.restart_region(s0, 0, center)
    assert abandoned == (AskId(7),)
    assert s1.pending_ids is None


def test_realcycle():
    pytest.importorskip("baux._rust", exc_type=ImportError)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    s0 = baux.init(bounds, pipe=baux.lhd(2) | baux.enn() | baux.regions(2), seed=42)
    assert s0.num_observations == 0

    s1, batch = baux.ask(s0, 2)
    assert len(batch.points) == 2
    assert not batch.points.flags.writeable

    s2, mu = baux.tell(s1, batch.points, np.array([0.5, -0.2]))
    assert mu is not None
    assert s2.num_observations == 2
    assert s1.num_observations == 0

    pred = baux.posterior_mu(s2, np.zeros((1, 2)))
    assert np.asarray(pred).shape[0] == 1

    s3, _ = baux.import_observations(s2, np.array([[0.1, 0.2]]), np.array([0.3]))
    assert s3.num_observations == 3
    assert s2.num_observations == 2


def test_realcomplete():
    pytest.importorskip("baux._rust", exc_type=ImportError)
    config = replace(
        turbo_zero(num_init=1),
        trust_region=RegionalSearchPolicy(count=2),
    )
    s0 = baux.init(
        domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
        config=config,
        seed=77,
    )
    s1, batch = baux.ask(s0, 2)
    assert len(batch.ids) == 2
    id1, id2 = batch.ids

    s2 = baux.complete(s1, [id1], np.array([[1.5]]))
    assert s2.pending_ids == (id2.value,)
    assert s1.pending_ids == (id1.value, id2.value)

    s3 = baux.cancel(s2, [id2])
    assert s3.pending_ids is None
