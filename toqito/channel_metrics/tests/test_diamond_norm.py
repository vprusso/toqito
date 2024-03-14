"""Tests for diamond_norm."""

import numpy as np
import pytest

from toqito.channel_metrics import diamond_norm
from toqito.channels import dephasing, depolarizing


@pytest.mark.parametrize(
    "test_input1, test_input_2, expected",
    [
        # The diamond norm of identical channels should yield 0
        (dephasing(2), dephasing(2), 0),
        # the diamond norm of different channels
        (dephasing(2), depolarizing(2), 1),
    ],
)
def test_diamond_norm_valid_inputs(test_input1, test_input_2, expected):
    """Test function works as expected for valid inputs."""
    calculated_value = diamond_norm(test_input1, test_input_2)
    assert pytest.approx(expected, 1e-3) == calculated_value


@pytest.mark.parametrize(
    "test_input1, test_input_2, expected_msg",
    [
        # Inconsistent dimensions between Choi matrices
        (depolarizing(4), dephasing(2), "The Choi matrices provided should be of equal dimension."),
        # Non-square inputs for diamond norm
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            "The Choi matrix provided must be square.",
        ),
    ],
)
def test_diamond_norm_invalid_inputs(test_input1, test_input_2, expected_msg):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError, match=expected_msg):
        diamond_norm(test_input1, test_input_2)
