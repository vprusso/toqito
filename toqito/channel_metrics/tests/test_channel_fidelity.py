"""Tests for channel_fidelity."""

import numpy as np
import pytest

from toqito.channel_metrics import channel_fidelity
from toqito.channels import dephasing, depolarizing

dephasing_channel = dephasing(4)
depolarizing_channel = depolarizing(4)


@pytest.mark.parametrize(
    "input1, input2, expected_value",
    [
        # Fidelity of identical channels.
        (dephasing_channel, dephasing_channel, 1),
        # Fidelity of different channels.
        (dephasing_channel, depolarizing_channel, 1 / 2),
    ],
)
def test_channel_fidelity(input1, input2, expected_value):
    """Test functions works as expected for valid inputs."""
    calculated_value = channel_fidelity(input1, input2, 1e-4)
    assert pytest.approx(expected_value, 1e-4) == calculated_value


@pytest.mark.parametrize(
    "input1, input2, expected_msg",
    [
        # Inconsistent dimensions between Choi matrices.
        (depolarizing_channel, depolarizing(2), "The Choi matrices provided should be of equal dimension."),
        # Non-square inputs for channel fidelity.
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            "The Choi matrix provided must be square.",
        ),
    ],
)
def test_channel_fidelity_raises_error(input1, input2, expected_msg):
    """Test functions works as expected for valid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        channel_fidelity(input1, input2)
