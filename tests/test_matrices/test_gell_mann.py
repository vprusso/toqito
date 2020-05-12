"""Tests for gell_man function."""
import unittest
import numpy as np

from scipy.sparse import csr_matrix
from toqito.matrices import gell_mann


class TestGellMann(unittest.TestCase):
    """Unit test for gell_mann."""

    def test_gell_mann_idx_0(self):
        """Gell-Mann operator for index = 0."""
        expected_res = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        res = gell_mann(0)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_1(self):
        """Gell-Mann operator for index = 1."""
        expected_res = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        res = gell_mann(1)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_2(self):
        """Gell-Mann operator for index = 2."""
        expected_res = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        res = gell_mann(2)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_3(self):
        """Gell-Mann operator for index = 3."""
        expected_res = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        res = gell_mann(3)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_4(self):
        """Gell-Mann operator for index = 4."""
        expected_res = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

        res = gell_mann(4)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_5(self):
        """Gell-Mann operator for index = 5."""
        expected_res = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])

        res = gell_mann(5)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_6(self):
        """Gell-Mann operator for index = 6."""
        expected_res = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

        res = gell_mann(6)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_7(self):
        """Gell-Mann operator for index = 7."""
        expected_res = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])

        res = gell_mann(7)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_gell_mann_idx_8(self):
        """Gell-Mann operator for index = 8."""
        expected_res = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
        res = gell_mann(8)

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_idx(self):
        """Invalid Gell-Mann parameters."""
        with self.assertRaises(ValueError):
            gell_mann(9)

    def test_gell_mann_sparse(self):
        """Test sparse Gell-Mann matrix."""
        res = gell_mann(3, is_sparse=True)
        self.assertEqual(isinstance(res, csr_matrix), True)


if __name__ == "__main__":
    unittest.main()
