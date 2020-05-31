"""Test perm_sign."""
import numpy as np

from toqito.perms import perm_sign


def test_perm_sign_small_example_even():
    """Small example when permutation is even."""
    res = perm_sign([1, 2, 3, 4])
    np.testing.assert_equal(res, 1)


def test_perm_sign_small_example_odd():
    """Small example when permutation is odd."""
    res = perm_sign([1, 2, 4, 3, 5])
    np.testing.assert_equal(res, -1)


if __name__ == "__main__":
    np.testing.run_module_suite()
