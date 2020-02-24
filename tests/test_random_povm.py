"""Tests for random_povm function."""
import unittest
import numpy as np

from toqito.random.random_povm import random_povm


class TestRandomPOVM(unittest.TestCase):
    """Unit test for random_povm."""

    def test_random_unitary_not_real(self):
        """Generate random POVMs and check that they sum to the identity."""
        dim, num_inputs, num_outputs = 2, 2, 2
        povms = random_povm(dim, num_inputs, num_outputs)
        self.assertEqual(np.allclose(
            povms[:, :, 0, 0] + povms[:, :, 0, 1],
            np.identity(dim)), True)


if __name__ == '__main__':
    unittest.main()
