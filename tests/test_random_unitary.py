"""Tests for random_unitary function."""
import unittest

from toqito.random.random_unitary import random_unitary
from toqito.linear_algebra.properties.is_unitary import is_unitary


class TestRandomUnitary(unittest.TestCase):
    """Unit test for random_unitary."""

    def test_random_unitary_not_real(self):
        """Generate random non-real unitary matrix."""
        mat = random_unitary(2)
        self.assertEqual(is_unitary(mat), True)

    def test_random_unitary_real(self):
        """Generate random real unitary matrix."""
        mat = random_unitary(2, True)
        self.assertEqual(is_unitary(mat), True)

    def test_random_unitary_vec_dim(self):
        """Generate random non-real unitary matrix."""
        mat = random_unitary([4, 4], True)
        self.assertEqual(is_unitary(mat), True)


if __name__ == "__main__":
    unittest.main()
