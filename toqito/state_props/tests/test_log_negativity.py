"""Test log_negativity."""

import numpy as np
import pytest

from toqito.state_props import log_negativity


@pytest.mark.parametrize(
    "rho, dim, expected_result",
    [
        # Test for log_negativity on rho.
        (
            np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]),
            None,
            1,
        ),
        # Test for log_negativity on rho (with dimension).
        (np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]), 2, 1),
    ],
)
def test_log_negativity(rho, dim, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(log_negativity(rho, dim), expected_result)


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
def test_log_negativity_invalid_input(rho, dim):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        log_negativity(rho, dim)
