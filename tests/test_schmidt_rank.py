"""Tests for schmidt_rank function."""
import unittest
import numpy as np

from toqito.base.ket import ket
from toqito.state.states.bell import bell
from toqito.state.operations.schmidt_rank import schmidt_rank


class TestSchmidtRank(unittest.TestCase):
    """Unit tests for schmidt_rank."""

    def test_schmidt_rank_bell_state(self):
        """
        Computing the Schmidt rank of the entangled Bell state should yield a
        value greater than 1.
        """
        rho = bell(0).conj().T * bell(0)
        self.assertEqual(schmidt_rank(rho) > 1, True)

    def test_schmidt_rank_singlet_state(self):
        """
        Computing the Schmidt rank of the entangled singlet state should yield
        a value greater than 1.
        """
        e_0, e_1 = ket(2, 0), ket(2, 1)
        rho = 1/np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
        rho = rho.conj().T * rho
        self.assertEqual(schmidt_rank(rho) > 1, True)

    def test_schmidt_rank_separable_state(self):
        """
        Computing the Schmidt rank of a separable state should yield a value
        equal to 1.
        """
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_00 = np.kron(e_0, e_0)
        e_01 = np.kron(e_0, e_1)
        e_10 = np.kron(e_1, e_0)
        e_11 = np.kron(e_1, e_1)
        rho = 1/2 * (e_00 - e_01 - e_10 + e_11)
        rho = rho.conj().T * rho
        self.assertEqual(schmidt_rank(rho) == 1, True)


if __name__ == '__main__':
    unittest.main()
