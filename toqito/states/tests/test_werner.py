"""Test werner."""

import numpy as np
import pytest

from toqito.matrix_props import is_density
from toqito.perms import permutation_operator
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


def test_werner_integer_alpha():
    """Integer scalar alpha is accepted; werner(d, 1) is the antisymmetric projector state."""
    state = werner(3, 1)
    np.testing.assert_equal(is_density(state), True)


def test_werner_bipartite_alpha_out_of_range():
    """A bipartite alpha outside [-1, 1] is rejected."""
    with pytest.raises(ValueError, match="must be in the interval"):
        werner(2, 5)


def test_werner_multipartite_permutation_weights():
    """Test that alpha weights are applied to the correct permutations."""
    dim = 2
    alpha = [0.01, 0.02, 0.03, 0.04, 0.05]

    perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]

    expected = np.identity(dim**3)

    for weight, perm in zip(alpha, perms[1:]):
        expected -= weight * permutation_operator(dim, perm, False, True)

    expected /= np.trace(expected)

    np.testing.assert_allclose(werner(dim, alpha), expected)


def test_werner_multipartite_two_parties_matches_bipartite():
    """A one-element alpha list gives the bipartite Werner state with that alpha."""
    np.testing.assert_allclose(werner(3, [0.5]), werner(3, 0.5))
