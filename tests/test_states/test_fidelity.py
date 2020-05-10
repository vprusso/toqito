"""Tests for fidelity function."""
import unittest
import cvxpy
import numpy as np

from toqito.states import basis
from toqito.state_metrics import fidelity


class TestFidelity(unittest.TestCase):
    """Unit test for fidelity."""

    def test_fidelity_default(self):
        """Test fidelity default arguments."""
        rho = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        sigma = rho

        res = fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_fidelity_cvx(self):
        """Test fidelity for cvx objects."""
        rho = cvxpy.bmat(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        sigma = rho

        res = fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_fidelity_non_identical_states_1(self):
        """Test the fidelity between two non-identical states."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T
        self.assertEqual(np.isclose(fidelity(rho, sigma), 0.996, rtol=1e-03), True)

    def test_fidelity_non_identical_states_2(self):
        """Test the fidelity between two non-identical states."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T
        self.assertEqual(np.isclose(fidelity(rho, sigma), 0.774, rtol=1e-03), True)

    def test_non_square(self):
        """Tests for invalid dim."""
        rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
        sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
        with self.assertRaises(ValueError):
            fidelity(rho, sigma)

    def test_invalid_dim(self):
        """Tests for invalid dim."""
        rho = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            fidelity(rho, sigma)


if __name__ == "__main__":
    unittest.main()
