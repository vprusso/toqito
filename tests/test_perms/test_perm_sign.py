"""Tests for perm_sign function."""
import unittest

from toqito.perms import perm_sign


class TestPermSign(unittest.TestCase):
    """Unit test for perm_sign."""

    def test_small_example_even(self):
        """Small example when permutation is even."""
        res = perm_sign([1, 2, 3, 4])
        self.assertEqual(res, 1)

    def test_small_example_odd(self):
        """Small example when permutation is odd."""
        res = perm_sign([1, 2, 4, 3, 5])
        self.assertEqual(res, -1)


if __name__ == "__main__":
    unittest.main()
