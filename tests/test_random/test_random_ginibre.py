"""Tests for random_ginibre function."""
import unittest

from toqito.random import random_ginibre


class TestRandomGinibre(unittest.TestCase):

    """Unit test for random_ginibre."""

    def test_random_ginibre_dims(self):
        """Generate random Ginibre matrix and check proper dimensions."""
        gin_mat = random_ginibre(2, 2)
        self.assertEqual(gin_mat.shape[0], 2)
        self.assertEqual(gin_mat.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
