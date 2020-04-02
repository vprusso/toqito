"""Tests for random_state_vector function."""
import unittest

from toqito.state.properties.is_pure import is_pure
from toqito.random.random_state_vector import random_state_vector


class TestRandomStateVector(unittest.TestCase):
    """Unit test for random_state_vector."""

    def test_random_complex_state_purity(self):
        """Check that complex state vector from random state vector is pure."""
        vec = random_state_vector(2)
        mat = vec.conj().T * vec
        self.assertEqual(is_pure(mat), True)

    def test_random_real_state_purity(self):
        """Check that real state vector from random state vector is pure."""
        vec = random_state_vector(2, True)
        mat = vec.conj().T * vec
        self.assertEqual(is_pure(mat), True)


if __name__ == "__main__":
    unittest.main()
