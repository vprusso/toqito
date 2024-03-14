"""Test negativity."""

import numpy as np
import pytest

from toqito.state_props import negativity


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # Test for log_negativity on rho.
        (
            np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]),
            None,
            1 / 2,
        ),
        # Test for negativity on rho for dimension as integer.
        (
            np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]),
            2,
            1 / 2,
        ),
    ],
)
def test_negativity(rho, dim, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(negativity(rho, dim), expected_result)


@pytest.mark.parametrize(
    "rho, dim",
    [
        # Invalid dim parameters.
        (np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]), 5),
        # Invalid dim parameters as list.
        (
            np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]),
            [2, 5],
        ),
    ],
)
def test_negativity_invalid_input(rho, dim):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        negativity(rho, dim)
