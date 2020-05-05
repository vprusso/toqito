"""Tests for sub_fidelity function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.states.bell import bell
from toqito.states.distance.fidelity import fidelity
from toqito.states.distance.sub_fidelity import sub_fidelity


class TestSubFidelity(unittest.TestCase):
    """Unit test for sub_fidelity."""

    def test_sub_fidelity_default(self):
        """Test sub_fidelity default arguments."""
        rho = bell(0) * bell(0).conj().T
        sigma = rho

        res = sub_fidelity(rho, sigma)
        self.assertEqual(np.isclose(res, 1), True)

    def test_sub_fidelity_lower_bound_1(self):
        """Test sub_fidelity is lower bound on fidelity for rho and sigma."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        sigma = 2 / 3 * e_0 * e_0.conj().T + 1 / 3 * e_1 * e_1.conj().T

        res = sub_fidelity(rho, sigma)
        self.assertLessEqual(res, fidelity(rho, sigma))

    def test_sub_fidelity_lower_bound_2(self):
        """Test sub_fidelity is lower bound on fidelity for rho and pi."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
        sigma = 1 / 8 * e_0 * e_0.conj().T + 7 / 8 * e_1 * e_1.conj().T

        res = sub_fidelity(rho, sigma)
        self.assertLessEqual(res, fidelity(rho, sigma))

    def test_non_square_sub_fidelity(self):
        """Tests for invalid dim for sub_fidelity."""
        rho = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [1, 0, 0, 1]])
        sigma = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [2, 0, 0, 2]])
        with self.assertRaises(ValueError):
            sub_fidelity(rho, sigma)

    def test_invalid_dim_sub_fidelity(self):
        """Tests for invalid dim for sub_fidelity."""
        rho = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )
        sigma = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError):
            sub_fidelity(rho, sigma)


if __name__ == "__main__":
    unittest.main()
