"""Tests for super_fidelity function."""
import unittest
import numpy as np

from toqito.states.states.bell import bell
from toqito.states.distance.super_fidelity import super_fidelity


class TestSuperFidelity(unittest.TestCase):
    """Unit test for super_fidelity."""

    def test_super_fidelity_default(self):
        """Test super_fidelity default arguments."""
        rho = bell(0) * bell(0).conj().T
        sigma = rho

        res = super_fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_non_square_super_fidelity(self):
        """Tests for invalid dim for super_fidelity."""
        rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1, 0, 0, 1]])
        sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [2, 0, 0, 2]])
        with self.assertRaises(ValueError):
            super_fidelity(rho, sigma)

    def test_invalid_dim_super_fidelity(self):
        """Tests for invalid dim for super_fidelity."""
        rho = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        sigma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError):
            super_fidelity(rho, sigma)


if __name__ == "__main__":
    unittest.main()
