"""Tests for channel_fidelity."""
import numpy as np
import pytest

from toqito.channel_metrics import channel_fidelity
from toqito.channels import dephasing, depolarizing

dephasing_channel = dephasing(4)
depolarizing_channel = depolarizing(4)


def test_channel_fidelity_same_channel():
    """The fidelity of identical channels should yield 1."""
    assert np.isclose(channel_fidelity(dephasing_channel, dephasing_channel), 1, atol=1e-3)


def test_channel_fidelity_different_channel():
    """Calculate the channel fidelity of different channels."""
    assert np.isclose(channel_fidelity(dephasing_channel, depolarizing_channel), 1 / 2, atol=1e-3)


def test_channel_fidelity_inconsistent_dims():
    """Inconsistent dimensions between Choi matrices."""
    with pytest.raises(
        ValueError, match="The Choi matrices provided should be of equal dimension."
    ):
        small_dim_depolarizing_channel = depolarizing(2)
        channel_fidelity(depolarizing_channel, small_dim_depolarizing_channel)


def test_channel_fidelity_non_square():
    """Non-square inputs for channel fidelity."""
    with pytest.raises(ValueError, match="The Choi matrix provided must be square."):
        choi_1 = np.array([[1, 2, 3], [4, 5, 6]])
        choi_2 = np.array([[1, 2, 3], [4, 5, 6]])
        channel_fidelity(choi_1, choi_2)
