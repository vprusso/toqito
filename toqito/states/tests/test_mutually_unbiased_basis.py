"""Test mutually_unbiased_basis."""

import re

import numpy as np
import pytest

from toqito.state_props import is_mutually_unbiased_basis
from toqito.states import mutually_unbiased_basis
from toqito.states.mutually_unbiased_basis import _is_prime_power


@pytest.mark.parametrize("dim", [2, 3, 5, 7])
def test_mutually_unbiased_basis(dim):
    """Test function works as expected for a valid input."""
    np.testing.assert_equal(is_mutually_unbiased_basis(mutually_unbiased_basis(dim)), True)


@pytest.mark.parametrize("dim", [4, 8, 9])
def test_mutually_unbiased_basis_prime_power_not_prime(dim):
    """Dimension is a prime power but not prime (this is not presently supported)."""
    with pytest.raises(
        ValueError,
        match=re.escape(f"Dimension {dim} is a prime power but not prime (more complicated no support at the moment)."),
    ):
        mutually_unbiased_basis(dim)


@pytest.mark.parametrize("dim", [6, 10, 12])
def test_mutually_unbiased_basis_unknown_for_dim(dim):
    """Dimension requested does not (at present) have a known way to generate."""
    with pytest.raises(ValueError, match=re.escape(f"No general construction of MUBs is known for dimension: {dim}.")):
        mutually_unbiased_basis(dim)


@pytest.mark.parametrize(
    "n, expected_result",
    [
        # Hard-coded non-prime power case.
        (1, False),
        # Non-prime powers.
        (6, False),
        (10, False),
        (12, False),
        (15, False),
        # Prime powers.
        (2, True),
        (3, True),
        (4, True),
        (5, True),
    ],
)
def test_is_prime_power(n, expected_result):
    """Test function works as expected for an invalid input."""
    np.testing.assert_array_equal(_is_prime_power(n), expected_result)
