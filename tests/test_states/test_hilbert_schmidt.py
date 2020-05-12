"""Tests for hilbert_schmidt function."""
import unittest
import numpy as np
from toqito.states import bell
from toqito.state_metrics import hilbert_schmidt


class TestHilbertSchmidt(unittest.TestCase):
    """Unit test for hilbert_schmidt."""

    def test_hilbert_schmidt_bell(self):
        r"""Test Hilbert-Schmidt distance on two Bell states."""

        rho = bell(0) * bell(0).conj().T
        sigma = bell(3) * bell(3).conj().T

        res = hilbert_schmidt(rho, sigma)

        self.assertEqual(np.isclose(res, 1), True)

    def test_hilbert_schmidt_non_density_matrix(self):
        r"""Test Hilbert-Schmidt distance on non-density matrix."""
        rho = np.array([[1, 2], [3, 4]])
        sigma = np.array([[5, 6], [7, 8]])

        with self.assertRaises(ValueError):
            hilbert_schmidt(rho, sigma)


if __name__ == "__main__":
    unittest.main()
