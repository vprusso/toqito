"""Tests for helstrom_holevo function."""
import unittest
import numpy as np
from toqito.states import basis
from toqito.state_metrics import helstrom_holevo


class TestHelstromHolevo(unittest.TestCase):
    """Unit test for helstrom_holevo."""

    def test_helstrom_holevo_same_state(self):
        r"""Test Helstrom-Holevo distance on same state."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_11 = np.kron(e_1, e_1)

        u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
        rho = u_vec * u_vec.conj().T
        sigma = rho

        res = helstrom_holevo(rho, sigma)

        self.assertEqual(np.isclose(res, 1 / 2), True)

    def test_helstrom_holevo_non_density_matrix(self):
        r"""Test Helstrom-Holevo distance on non-density matrix."""
        rho = np.array([[1, 2], [3, 4]])
        sigma = np.array([[5, 6], [7, 8]])

        with self.assertRaises(ValueError):
            helstrom_holevo(rho, sigma)


if __name__ == "__main__":
    unittest.main()
