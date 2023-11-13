"""Tests for channel_fidelity."""
import numpy as np
import pytest

from toqito.channel_metrics import channel_fidelity
from toqito.channels import dephasing, depolarizing

# def test_channel_fidelity_same_channel():
#     """The fidelity of identical channels should yield 1."""
#     choi_1 = dephasing(4)
#     choi_2 = dephasing(4)
#     np.testing.assert_equal(np.isclose(channel_fidelity(choi_1, choi_2), 1, atol=1e-3), True)


# def test_channel_fidelity_different_channel():
#     """Calculate the channel fidelity of different channels."""
#     choi_1 = dephasing(4)
#     choi_2 = depolarizing(4)
#     np.testing.assert_equal(np.isclose(channel_fidelity(choi_1, choi_2), 1 / 2, atol=1e-3), True)


def test_channel_fidelity_inconsistent_dims():
    """Inconsistent dimensions between Choi matrices."""
    with pytest.raises(ValueError, match='The Choi matrices provided should be of equal dimension.'):
        choi_1 = depolarizing(4)
        choi_2 = dephasing(2)
        channel_fidelity(choi_1, choi_2)


def test_channel_fidelity_non_square():
    """Non-square inputs for channel fidelity."""
    with pytest.raises(ValueError, match='The Choi matrix provided must be square.'):
        choi_1 = np.array([[1, 2, 3], [4, 5, 6]])
        choi_2 = np.array([[1, 2, 3], [4, 5, 6]])
        channel_fidelity(choi_1, choi_2)
