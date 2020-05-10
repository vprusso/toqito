"""Tests for reduction_map function."""
import unittest

from toqito.channels import reduction


class TestReduction(unittest.TestCase):
    """Unit test for reduction."""

    def test_reduction_map(self):
        """Test for the standard reduction map."""
        res = reduction(3)
        self.assertEqual(res[4, 0], -1)
        self.assertEqual(res[8, 0], -1)
        self.assertEqual(res[1, 1], 1)
        self.assertEqual(res[2, 2], 1)
        self.assertEqual(res[3, 3], 1)
        self.assertEqual(res[0, 4], -1)
        self.assertEqual(res[8, 4], -1)
        self.assertEqual(res[5, 5], 1)
        self.assertEqual(res[6, 6], 1)
        self.assertEqual(res[7, 7], 1)
        self.assertEqual(res[0, 8], -1)
        self.assertEqual(res[4, 8], -1)


if __name__ == "__main__":
    unittest.main()
