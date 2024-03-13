"""Test breuer."""

import numpy as np
import pytest

from toqito.states import breuer


@pytest.mark.parametrize(
    "dim, lam, expected_result",
    [
        # Generate Breuer state of dimension 4 with weight 0.1.
        (2, 0.1, np.array([[0.3, 0, 0, 0], [0, 0.2, 0.1, 0], [0, 0.1, 0.2, 0], [0, 0, 0, 0.3]])),
    ],
)
def test_breuer(dim, lam, expected_result):
    """Test function works as expected for a valid input."""
    np.testing.assert_allclose(breuer(dim, lam), expected_result)


@pytest.mark.parametrize(
    "dim, lam",
    [
        # Ensures that an odd dimension if not accepted.
        (3, 0.1)
    ],
)
def test_breuer_invalid(dim, lam):
    """Test function works as expected for an invalid input."""
    with np.testing.assert_raises(ValueError):
        breuer(dim, lam)
