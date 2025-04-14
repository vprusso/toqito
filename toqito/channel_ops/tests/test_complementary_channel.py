"""Tests for complementary_channel."""

import numpy as np
import pytest

from toqito.channel_ops import complementary_channel

# Define test cases for complementary map
kraus_1 = np.array([[1, 0], [0, 0]]) / np.sqrt(2)
kraus_2 = np.array([[0, 1], [0, 0]]) / np.sqrt(2)
kraus_3 = np.array([[0, 0], [1, 0]]) / np.sqrt(2)
kraus_4 = np.array([[0, 0], [0, 1]]) / np.sqrt(2)

# Expected results for the complementary map
expected_res_comp = [
    np.array([[1, 0], [0, 1], [0, 0], [0, 0]]) / np.sqrt(2),
    np.array([[0, 0], [0, 0], [1, 0], [0, 1]]) / np.sqrt(2),
]

# Higher-dimensional Kraus operators (3x3)
kraus_5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / np.sqrt(3)
kraus_6 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) / np.sqrt(3)
kraus_7 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) / np.sqrt(3)

expected_res_comp_high_dim = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / np.sqrt(3),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]) / np.sqrt(3),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) / np.sqrt(3),
]

# Single Kraus operator (edge case)
kraus_single = np.array([[1, 0], [0, 1]])

expected_res_single = [
    np.array([[1, 0]]),
    np.array([[0, 1]]),
]


@pytest.mark.parametrize(
    "kraus_ops, expected",
    [
        # Test complementary_channel on a set of 2x2 Kraus operators (the ones you gave).
        ([kraus_1, kraus_2, kraus_3, kraus_4], expected_res_comp),
        # Test complementary_channel with higher-dimensional (3x3) Kraus operators.
        ([kraus_5, kraus_6, kraus_7], expected_res_comp_high_dim),
        # Test complementary_channel with a single Kraus operator (edge case).
        ([kraus_single], expected_res_single),
    ],
)
def test_complementary_channel(kraus_ops, expected):
    """Test complementary_channel works as expected for valid inputs."""
    calculated = complementary_channel(kraus_ops)

    # Compare the shapes first to debug broadcasting issues
    assert len(calculated) == len(expected), "Mismatch in number of Kraus operators"
    for calc_op, exp_op in zip(calculated, expected):
        assert np.isclose(calc_op, exp_op, atol=1e-6).all()


@pytest.mark.parametrize(
    "kraus_ops",
    [
        # Invalid test case: non-square matrices
        ([np.array([[1, 0, 0], [0, 1, 0]])]),  # Not a square matrix
        # Invalid test case: empty list of Kraus operators
        ([]),
        # Invalid test case: single row matrix (not a square)
        ([np.array([[1, 0]])]),
        # Different dimenisions for kraus operators in a set
        ([np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]]), np.array([[1, 0], [0, 1]])]),
        # Invalid test case: Kraus operators that do not satisfy the completeness relation
        ([np.array([[1, 0], [0, 0.5]]), np.array([[0, 0.5], [0, 0.5]])]),  # Sum != I
    ],
)
def test_complementary_channel_error(kraus_ops):
    """Test function raises error as expected for invalid inputs."""
    with pytest.raises(ValueError):
        complementary_channel(kraus_ops)
