from dataclasses import replace

import pytest
from baux import turbo_zero
from baux.turbo.config import RegionalSearchPolicy, RestartCenterPolicy
from baux.turbo.config.encode import encode


def testcenterdefault():
    policy = RegionalSearchPolicy()
    assert policy.restart_center == RestartCenterPolicy()
    encoded = encode(replace(turbo_zero(), trust_region=policy))
    assert encoded["region_restart_candidates"] == 1
    assert encoded["region_restart_center_selection"] == "first"


def testcenternondefault():
    policy = RegionalSearchPolicy(
        restart_center=RestartCenterPolicy(candidates=32, selection="maximin")
    )
    encoded = encode(replace(turbo_zero(), trust_region=policy))
    assert encoded["region_restart_candidates"] == 32
    assert encoded["region_restart_center_selection"] == "maximin"


@pytest.mark.parametrize("candidates", [0, -1])
def testcenterpositive(candidates):
    with pytest.raises(ValueError, match="candidates must be positive"):
        RestartCenterPolicy(candidates=candidates)


@pytest.mark.parametrize("candidates", [True, 1.5])
def testcenterinteger(candidates):
    with pytest.raises(TypeError, match="candidates must be an integer"):
        RestartCenterPolicy(candidates=candidates)


def testcenterselection():
    with pytest.raises(ValueError, match="selection"):
        RestartCenterPolicy(selection="unknown")


def testregioncenterrequired():
    with pytest.raises(TypeError, match="RestartCenterPolicy"):
        RegionalSearchPolicy(restart_center=None)
