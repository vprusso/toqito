"""Tests for dual_channel."""

import numpy as np
import pytest

from toqito.channel_ops import dual_channel
from toqito.channels import choi
from toqito.perms import swap_operator

kraus_1 = np.array([[1, 0, 1j, 0]])
kraus_2 = np.array([[0, 1, 0, 1j]])

expected_res_cp = [np.array([[1, 0, -1j, 0]]).T, np.array([[0, 1, 0, -1j]]).T]

expected_res_2d = [
    [np.array([[1, 0, -1j, 0]]).T, np.array([[1, 0, -1j, 0]]).T],
    [np.array([[0, 1, 0, -1j]]).T, np.array([[0, 1, 0, -1j]]).T],
]

input_diff_dims = np.array(
    [
        [1, -1j, 0, 0, 0, 1],
        [1j, -1, 0, 0, 0, -1j],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1j, 0, 0, 0, 1],
    ]
)

expected_res_diff_dims = np.array(
    [
        [1, 0, 0, 1j, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-1j, 0, 0, -1, 0, 1j],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, -1j, 0, 1],
    ]
)

expected_swap = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


@pytest.mark.parametrize(
    "test_input, expected, input_dim",
    [
        # Test dual_channel on a channel represented as Kraus operators (1d list, CP map)
        ([kraus_1, kraus_2], expected_res_cp, None),
        # Test dual_channel on a channel represented as Kraus operators (2d list).
        ([[kraus_1, kraus_1], [kraus_2, kraus_2]], expected_res_2d, None),
        # Test dual_channel on a 9x9 Choi matrix, inferring dims=[3,3]
        (choi(1, 1, 0), choi(1, 0, 1), None),
        # Test dual_channel on a Choi matrix with different input and output dimensions.
        (input_diff_dims, expected_res_diff_dims, [3, 2]),
        # Dual of a channel that transposes 3x2 matrices
        (swap_operator([2, 3]), expected_swap, [[3, 2], [2, 3]]),
    ],
)
def test_dual_channel(test_input, expected, input_dim):
    """Test function works as expected for valid inputs."""
    if input_dim is None:
        calculated = dual_channel(test_input)
        assert np.isclose(calculated, expected).all()

    calculated = dual_channel(test_input, dims=input_dim)
    assert np.isclose(calculated, expected).all()


@pytest.mark.parametrize(
    "test_input",
    [
        # If the channel is represented as an array, it must be two-dimensional (a matrix).
        (np.array([1, 2, 3, 4])),
        # Test output of function when the dimensions must be specified. If the size of the Choi matrix is not a perfect
        # square, the dimensions of the input and output spaces must be specified.
        (np.arange(36).reshape(6, 6)),
        # error for an invalid input
        ([0]),
    ],
)
def test_dual_channel_error(test_input):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError):
        dual_channel(test_input)
