import pytest

from topoflow_prior.schemas import TopologySpec
from topoflow_prior.topology import MI300X, MI355X, get_topology


def test_mi300x_constants():
    assert isinstance(MI300X, TopologySpec)
    assert MI300X.name == "MI300X"
    assert MI300X.arch == "gfx942"
    assert MI300X.num_xcds == 8
    assert MI300X.l2_per_xcd_mb == 4.0
    assert MI300X.cu_per_xcd == 38


def test_mi355x_placeholder():
    assert isinstance(MI355X, TopologySpec)
    assert MI355X.arch == "gfx950"


def test_get_topology_lookup():
    assert get_topology("mi300x") is MI300X
    assert get_topology("MI300X") is MI300X
    assert get_topology("gfx942") is MI300X


def test_get_topology_unknown_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        get_topology("nvidia-h100")
