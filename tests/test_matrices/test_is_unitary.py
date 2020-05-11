"""Tests for is_unitary function."""
import unittest
import numpy as np

from toqito.matrix_props import is_unitary
from toqito.random import random_unitary


class TestIsUnitary(unittest.TestCase):

    """Unit test for is_unitary."""

    def test_is_unitary_random(self):
        """Test that random unitary matrix returns True."""
        mat = random_unitary(2)
        self.assertEqual(is_unitary(mat), True)

    def test_is_unitary_hardcoded(self):
        """Test that hardcoded unitary matrix returns True."""
        mat = np.array([[0, 1], [1, 0]])
        self.assertEqual(is_unitary(mat), True)

    def test_is_not_unitary(self):
        """Test that non-unitary matrix returns False."""
        mat = np.array([[1, 0], [1, 1]])
        self.assertEqual(is_unitary(mat), False)

    def test_is_not_unitary_matrix(self):
        """Test that non-unitary matrix returns False."""
        mat = np.array([[1, 0], [1, 1]])
        mat = np.matrix(mat)
        self.assertEqual(is_unitary(mat), False)


if __name__ == "__main__":
    unittest.main()
