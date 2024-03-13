"""Test is_mutually_unbiased_basis."""

import numpy as np
import pytest

from toqito.state_props import is_mutually_unbiased_basis
from toqito.states import basis

MUB_2 = [
    np.array([[1], [0]]),
    np.array([[0], [1]]),
    1 / np.sqrt(2) * (np.array([[1], [0]]) + np.array([[0], [1]])),
    1 / np.sqrt(2) * (np.array([[1], [0]]) - np.array([[0], [1]])),
    1 / np.sqrt(2) * (np.array([[1], [0]]) + 1j * np.array([[0], [1]])),
    1 / np.sqrt(2) * (np.array([[1], [0]]) - 1j * np.array([[0], [1]])),
]

MUB_4 = [
    np.array([[1], [0], [0], [0]]),
    np.array([[0], [1], [0], [0]]),
    np.array([[0], [0], [1], [0]]),
    np.array([[0], [0], [0], [1]]),
    1 / 2 * np.array([[1], [1], [1], [1]]),
    1 / 2 * np.array([[1], [1], [-1], [-1]]),
    1 / 2 * np.array([[1], [-1], [-1], [1]]),
    1 / 2 * np.array([[1], [-1], [1], [-1]]),
    1 / 2 * np.array([[1], [-1], [-1j], [-1j]]),
    1 / 2 * np.array([[1], [-1], [1j], [1j]]),
    1 / 2 * np.array([[1], [1], [1j], [-1j]]),
    1 / 2 * np.array([[1], [1], [-1j], [1j]]),
    1 / 2 * np.array([[1], [-1j], [-1j], [-1]]),
    1 / 2 * np.array([[1], [-1j], [1j], [1]]),
    1 / 2 * np.array([[1], [1j], [1j], [-1]]),
    1 / 2 * np.array([[1], [1j], [-1j], [1]]),
    1 / 2 * np.array([[1], [-1j], [-1], [-1j]]),
    1 / 2 * np.array([[1], [-1j], [1], [1j]]),
    1 / 2 * np.array([[1], [1j], [-1], [1j]]),
    1 / 2 * np.array([[1], [1j], [1], [-1j]]),
]

e_0, e_1 = basis(2, 0), basis(2, 1)


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Return True for MUB of dimension 2.
        (MUB_2, True),
        # Return True for MUB of dimension 4.
        (MUB_4, True),
        # Return False for non-MUB of dimension 2.
        (
            [
                e_0,
                e_1,
                1 / np.sqrt(2) * (e_0 + e_1),
                e_1,
                1 / np.sqrt(2) * (e_0 + 1j * e_1),
                e_0,
            ],
            False,
        ),
        # Return False for any vectors such that the number of vectors % dim != 0:
        (
            [
                np.array([1, 0]),
                np.array([1, 0]),
                np.array([1, 0]),
            ],
            False,
        ),
        # Return False for any vectors such that the number of vectors % dim != 0:
        (
            [
                np.array([1, 0]),
            ],
            False,
        ),
    ],
)
def test_is_mutually_unbiased(states, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_mutually_unbiased_basis(states), expected_result)
