"""Tests for purity."""

import numpy as np
import pytest

from toqito.state_props import purity
from toqito.states import werner


@pytest.mark.parametrize(
    "rho, expected_result",
    [
        # Test for identity matrix.
        (np.identity(4) / 4, 1 / 4),
        # Test purity of mixed Werner state.
        (werner(2, 1 / 4), 0.2653),
    ],
)
def test_purity(rho, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(purity(rho), expected_result, atol=4)


@pytest.mark.parametrize(
    "rho",
    [
        # Test purity on non-density matrix.
        (np.array([[1, 2], [3, 4]])),
    ],
)
def test_purity_invalid(rho):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        purity(rho)
