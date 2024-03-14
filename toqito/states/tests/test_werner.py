"""Test werner."""

import numpy as np
import pytest

from toqito.matrix_props import is_density
from toqito.states import werner


def test_werner_qutrit():
    """Test for qutrit Werner state."""
    res = werner(3, 1 / 2)
    np.testing.assert_equal(np.isclose(res[0][0], 0.0666666), True)
    np.testing.assert_equal(np.isclose(res[1][3], -0.066666), True)


def test_werner_multipartite():
    """Test for multipartite Werner state."""
    res = werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
    np.testing.assert_equal(np.isclose(res[0][0], 0.1127, atol=1e-02), True)


def test_werner_multipartite_valid():
    """Test multipartite Werner states with valid alpha lengths."""
    # Valid alpha length for p=3 (2!-1 = 1)
    alpha = [0.5]
    dim = 2
    state = werner(dim, alpha)
    np.testing.assert_equal(is_density(state), True)


@pytest.mark.parametrize(
    "dim, alpha",
    [
        # Invalid alpha length (not matching p!-1 for any integer p > 1)
        (2, [0.5, 0.6, 0.7]),
        # Test with an integer (which is not a valid type for alpha)
        (2, 5),
        # Test with a string (which is not a valid type for alpha)
        (2, "invalid"),
        # Test with a dictionary (which is not a valid type for alpha)
        (2, {"key": "value"}),
    ],
)
def test_werner_state_invalid(dim, alpha):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError):
        werner(dim, alpha)
