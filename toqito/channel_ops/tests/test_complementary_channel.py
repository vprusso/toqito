"""Tests for complementary_channel."""

import numpy as np
import pytest

from toqito.channel_ops import complementary_channel

# Define test cases for complementary map
kraus_1 = np.array([[1, 0], [0, 1]]) / np.sqrt(2)
kraus_2 = np.array([[0, 1], [1, 0]]) / np.sqrt(2)
kraus_3 = np.array([[0, -1j], [1j, 0]]) / np.sqrt(2)
kraus_4 = np.array([[1, 0], [0, -1]]) / np.sqrt(2)

# Expected results for the complementary map
expected_res_comp = [
    np.array([[1, 0], [0, 1], [0, 0], [0, -1j]]),
    np.array([[0, 1], [1, 0], [1j, 0], [0, -1]]),
]

# Higher-dimensional Kraus operators (3x3)
kraus_5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) / np.sqrt(3)
kraus_6 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(3)

expected_res_comp_high_dim = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]),
]

# Single Kraus operator (edge case)
kraus_single = np.array([[1, 0], [0, 1]])

expected_res_single = [np.array([[1, 0], [0, 1]])]  # Same as input

# Large size (4x4 Kraus operator)
kraus_large_1 = np.eye(4)
kraus_large_2 = np.fliplr(np.eye(4))

expected_res_large = [
    np.vstack([np.eye(4)[i, :] for i in range(4)]),
    np.vstack([np.fliplr(np.eye(4))[i, :] for i in range(4)]),
]

# @pytest.mark.parametrize(
#     "kraus_ops, expected",
#     [
#         # Test complementary_channel on a set of 2x2 Kraus operators (the ones you gave).
#         ([kraus_1, kraus_2, kraus_3, kraus_4], expected_res_comp),
#         # Test complementary_channel with higher-dimensional (3x3) Kraus operators.
#         ([kraus_5, kraus_6], expected_res_comp_high_dim),
#         # Test complementary_channel with a single Kraus operator (edge case).
#         ([kraus_single], expected_res_single),
#         # Test complementary_channel with large (4x4) Kraus operators.
#         ([kraus_large_1, kraus_large_2], expected_res_large),
#     ],
# )
# def test_complementary_channel(kraus_ops, expected):
#     """Test complementary_channel works as expected for valid inputs."""
#     calculated = complementary_channel(kraus_ops)

#     # Compare the shapes first to debug broadcasting issues
#     assert len(calculated) == len(expected), "Mismatch in number of Kraus operators"
#     for calc_op, exp_op in zip(calculated, expected):
#         assert np.isclose(calc_op, exp_op, atol=1e-6).all()

# @pytest.mark.parametrize(
#     "kraus_ops",
#     [
#         # Invalid test case: non-square matrices
#         ([np.array([[1, 0, 0], [0, 1, 0]])]),  # Not a square matrix
#         # Invalid test case: empty list of Kraus operators
#         ([]),
#         # Invalid test case: single row matrix (not a square)
#         ([np.array([[1, 0]])]),
#     ],
# )
# def test_complementary_channel_error(kraus_ops):
#     """Test function raises error as expected for invalid inputs."""
#     with pytest.raises(ValueError):
#         complementary_channel(kraus_ops)

# @pytest.mark.parametrize(
#     "kraus_ops",
#     [
#         # Test complementary_channel with identity operator (should return same operator rows stacked).
#         ([np.eye(2)]),
#         # Test complementary_channel with Kraus operators that are zero matrices.
#         ([np.zeros((2, 2))]),
#     ],
# )
# def test_complementary_channel_special_cases(kraus_ops):
#     """Test complementary_channel handles special cases like identity or zero operators."""
#     # Calculate the complementary map
#     calculated = complementary_channel(kraus_ops)

#     # For identity, the complementary map should stack identity rows
#     if np.array_equal(kraus_ops[0], np.eye(2)):
#         expected = [np.eye(2)]
#         assert np.isclose(calculated[0], expected[0]).all()

#     # For zero operators, complementary map should be zero as well
#     if np.array_equal(kraus_ops[0], np.zeros((2, 2))):
#         expected = [np.zeros((2, 2))]
#         assert np.isclose(calculated[0], expected[0]).all()
