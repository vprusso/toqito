"""Tests for gen_pauli function."""
import unittest
import numpy as np

from toqito.linear_algebra.matrices.gen_pauli import gen_pauli


class TestGenPauli(unittest.TestCase):
    """Unit test for gen_pauli."""

    def test_gen_pauli_1_0_2(self):
        """Generalized Pauli operator for k_1 = 1, k_2 = 0, and dim = 2."""
        dim = 2
        k_1 = 1
        k_2 = 0

        # Pauli-X operator.
        expected_res = np.array([[0, 1], [1, 0]])
        res = gen_pauli(k_1, k_2, dim)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gen_pauli_0_1_2(self):
        """Generalized Pauli operator for k_1 = 0, k_2 = 1, and dim = 2."""
        dim = 2
        k_1 = 0
        k_2 = 1

        # Pauli-Z operator.
        expected_res = np.array([[1, 0], [0, -1]])
        res = gen_pauli(k_1, k_2, dim)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gen_pauli_1_1_2(self):
        """Generalized Pauli operator for k_1 = 1, k_2 = 1, and dim = 2."""
        dim = 2
        k_1 = 1
        k_2 = 1

        # Pauli-Y operator.
        expected_res = np.array([[0, -1], [1, 0]])
        res = gen_pauli(k_1, k_2, dim)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
