"""Test random_povm."""
import unittest
import numpy as np

from toqito.random import random_povm


class TestRandomPOVM(unittest.TestCase):
    """Unit test for NonlocalGame."""

    def test_random_povm_unitary_not_real(self):
        """Generate random POVMs and check that they sum to the identity."""
        dim, num_inputs, num_outputs = 2, 2, 2
        povms = random_povm(dim, num_inputs, num_outputs)

        self.assertEqual(povms.shape, (dim, dim, num_inputs, num_outputs))

        np.testing.assert_allclose(
            povms[:, :, 0, 0] + povms[:, :, 0, 1], np.identity(dim), atol=1e-7
        )

    def test_random_povm_uneven_dimensions(self):
        """Generate random POVMs of uneven dimensions"""
        dim, num_inputs, num_outputs = 2, 3, 4
        povms = random_povm(dim, num_inputs, num_outputs)

        self.assertEqual(povms.shape, (dim, dim, num_inputs, num_outputs))

        for i in range(num_inputs):
            povm_sum = np.sum(povms[:, :, i, :], axis=-1)
            np.testing.assert_allclose(povm_sum, np.identity(dim), atol=1e-7)


if __name__ == "__main__":
    np.testing.run_module_suite()
