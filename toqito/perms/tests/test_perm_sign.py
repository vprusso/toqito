"""Test perm_sign."""

import numpy as np
import pytest

from toqito.perms import perm_sign


def test_perm_sign_small_example_even():
    """Small example when permutation is even."""
    res = perm_sign([1, 2, 3, 4])
    np.testing.assert_equal(res, 1)


def test_perm_sign_small_example_odd():
    """Small example when permutation is odd."""
    res = perm_sign([1, 2, 4, 3, 5])
    np.testing.assert_equal(res, -1)


@pytest.mark.parametrize(
    "perm, expected",
    [
        # 3-cycle (even number of inversions).
        ([2, 3, 1], 1),
        ([3, 1, 2], 1),
        # Single transposition (odd).
        ([2, 1, 3], -1),
        ([1, 3, 2], -1),
        # Reversal of four elements has six inversions (even).
        ([4, 3, 2, 1], 1),
        # A scrambled permutation with three inversions (odd): (4,1), (4,2), (4,3).
        ([4, 1, 2, 3], -1),
    ],
)
def test_perm_sign_scrambled(perm, expected):
    """Sign of genuinely scrambled permutations matches the parity of their inversions."""
    np.testing.assert_allclose(perm_sign(perm), expected)
