import unittest
import numpy as np

from toqito.matrix_ops.k_incoherence import is_k_incoherence

class TestIsKIncoherent(unittest.TestCase):
    def test_diagonal_state_k1(self):
        """Diagonal state should be 1-incoherent."""
        rho = np.diag([0.2, 0.5, 0.3])  # a 3x3 diagonal density matrix
        self.assertTrue(is_k_incoherence(rho, 1))
        # It should also be k-incoherent for any k>=1 (e.g., k=2 or 3) since diagonal states are incoherent.
        self.assertTrue(is_k_incoherence(rho, 2))
        self.assertTrue(is_k_incoherence(rho, 3))

    def test_off_diagonal_not_1_incoherent(self):
        """Off-diagonal state is not 1-incoherent (not diagonal in the basis)."""
        rho = np.array([[0.5, 0.4],
                        [0.4, 0.5]])
        # This 2x2 state has off-diagonal coherence, so it should not be 1-incoherent.
        self.assertFalse(is_k_incoherence(rho, 1))
        # For k=2 (which is >= dimension), it should be trivially incoherent.
        self.assertTrue(is_k_incoherence(rho, 2))

    def test_pure_state_incoherence(self):
        """Pure state: incoherent if support size <= k."""
        # Prepare a pure state |psi> with 2 nonzero components in 4-dim space.
        psi = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        rho = np.outer(psi, psi.conj())  # rank-1 projector
        # This pure state has support of size 2, so it should be 2-incoherent (and also 3- or 4-incoherent), but not 1-incoherent.
        self.assertFalse(is_k_incoherence(rho, 1))
        self.assertTrue(is_k_incoherence(rho, 2))
        self.assertTrue(is_k_incoherence(rho, 3))
        self.assertTrue(is_k_incoherence(rho, 4))

    def test_invalid_input_not_square(self):
        """Non-square input matrix should raise ValueError."""
        rho = np.array([[0.5, 0.1, 0.1],
                        [0.1, 0.4, 0.1]])  # 2x3 matrix, not square
        with self.assertRaises(ValueError) as cm:
            is_k_incoherence(rho, 1)
        self.assertIn("must be square", str(cm.exception))

    def test_invalid_input_bad_k(self):
        """Invalid k (<=0 or non-integer) should raise ValueError."""
        rho = np.eye(2)
        with self.assertRaises(ValueError):
            is_k_incoherence(rho, 0)
        with self.assertRaises(ValueError):
            is_k_incoherence(rho, -1)
        with self.assertRaises(ValueError):
            is_k_incoherence(rho, 1.5)
