"""Tests for concurrence function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.entanglement.concurrence import concurrence


class TestConcurrence(unittest.TestCase):
    """Unit test for concurrence."""

    def test_concurrence_entangled(self):
        """The concurrence on maximally entangled Bell state."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)

        u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
        rho = u_vec * u_vec.conj().T

        res = concurrence(rho)
        self.assertEqual(np.isclose(res, 1), True)

    def test_concurrence_separable(self):
        """The concurrence of a product state is zero."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        v_vec = np.kron(e_0, e_1)
        sigma = v_vec * v_vec.conj().T

        res = concurrence(sigma)
        self.assertEqual(np.isclose(res, 0), True)

    def test_invalid_dim(self):
        """Tests for invalid dimension inputs."""
        with self.assertRaises(ValueError):
            rho = np.identity(5)
            concurrence(rho)


if __name__ == "__main__":
    unittest.main()
