"""Test is_mutually_unbiased_basis."""
import numpy as np
import pytest

from toqito.state_props import is_mutually_unbiased_basis


e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])


@pytest.mark.parametrize(
    "states, expected_result",
    [
        # Return True for MUB of dimension 2.
        (
            [
                [e_0, e_1],
                [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)],
                [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)],
            ],
            True,
        ),
        # Return False for non-MUB of dimension 2.
        (
            [
                [e_0, e_1],
                [1 / np.sqrt(2) * (e_0 + e_1), e_1],
                [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0],
            ],
            False,
        ),
    ],
)
def test_is_mutually_unbiased(states, expected_result):
    np.testing.assert_equal(is_mutually_unbiased_basis(states), expected_result)


@pytest.mark.parametrize(
    "states",
    [
        # Invalid input length.
        ([np.array([1, 0])]),
    ],
)
def test_is_mutually_unbiased_basis_invalid_input(states):
    with np.testing.assert_raises(ValueError):
        is_mutually_unbiased_basis(states)
