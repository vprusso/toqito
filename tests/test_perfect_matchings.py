"""Tests for perfect_matchings function."""
import unittest
import numpy as np

from toqito.perms.perfect_matchings import perfect_matchings


class TestPerfectMatchings(unittest.TestCase):
    """Unit test for perfect_matchings."""

    def test_perfect_matchings_int_base_case(self):
        """Perfect matchings when input is integer and hit base case."""
        res = perfect_matchings(2)
        self.assertEqual(np.allclose(res, [0, 1]), True)

    def test_perfect_matchings_int_odd_num(self):
        """Perfect matchings when input is integer and hit odd number."""
        res = perfect_matchings(3)
        self.assertEqual(np.allclose(res, [0, 0, 0]), True)


if __name__ == '__main__':
    unittest.main()
