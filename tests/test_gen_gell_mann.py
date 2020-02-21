"""Tests for gen_gell_man function."""
import unittest
import numpy as np

from toqito.matrix.matrices.gen_gell_mann import gen_gell_mann


class TestGenGellMann(unittest.TestCase):
    """Unit test for gen_gell_mann."""

    def test_gell_mann_identity(self):
        """Generalized Gell-Mann operator identity."""
        expected_res = np.array([[1, 0],
                                 [0, 1]])
        res = gen_gell_mann(0, 0, 2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_pauli_x(self):
        """Generalized Gell-Mann operator Pauli-X."""
        expected_res = np.array([[0, 1],
                                 [1, 0]])
        res = gen_gell_mann(0, 1, 2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_pauli_y(self):
        """Generalized Gell-Mann operator Pauli-Y."""
        expected_res = np.array([[0, -1j],
                                 [1j, 0]])
        res = gen_gell_mann(1, 0, 2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_pauli_z(self):
        """Generalized Gell-Mann operator Pauli-Z."""
        expected_res = np.array([[1, 0],
                                 [0, -1]])
        res = gen_gell_mann(1, 1, 2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_dim_3_1(self):
        """Generalized Gell-Mann operator 3-dimensional."""
        expected_res = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, 0]])
        res = gen_gell_mann(0, 1, 3)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_dim_3_2(self):
        """Generalized Gell-Mann operator 3-dimensional."""
        expected_res = np.array([[0, 0, 1],
                                 [0, 0, 0],
                                 [1, 0, 0]])
        res = gen_gell_mann(0, 2, 3)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_dim_3_3(self):
        """Generalized Gell-Mann operator 3-dimensional."""
        expected_res = np.array([[1/np.sqrt(3), 0, 0],
                                 [0, 1/np.sqrt(3), 0],
                                 [0, 0, -2*1/np.sqrt(3)]])
        res = gen_gell_mann(2, 2, 3)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_dim_4_1(self):
        """Generalized Gell-Mann operator 4-dimensional."""
        expected_res = np.array([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]])
        res = gen_gell_mann(2, 3, 4)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_sparse(self):
        """Generalized Gell-Mann operator sparse."""
        res = gen_gell_mann(205, 34, 500, True)

        self.assertEqual(res[34, 205], -1j)
        self.assertEqual(res[205, 34], 1j)


if __name__ == '__main__':
    unittest.main()
