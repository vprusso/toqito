"""Tests for pauli function."""
import unittest
import numpy as np
import scipy

from toqito.linear_algebra.matrices.pauli import pauli


class TestPauli(unittest.TestCase):
    """Unit test for pauli."""

    def test_pauli_str_sparse(self):
        """Pauli-I operator with argument "I"."""
        res = pauli("I", True)

        self.assertEqual(scipy.sparse.issparse(res), True)

    def test_pauli_int_sparse(self):
        """Pauli-I operator with argument "I"."""
        res = pauli(0, True)

        self.assertEqual(scipy.sparse.issparse(res), True)

    def test_pauli_i(self):
        """Pauli-I operator with argument "I"."""
        expected_res = np.array([[1, 0], [0, 1]])
        res = pauli("I")

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_x(self):
        """Pauli-X operator with argument "X"."""
        expected_res = np.array([[0, 1], [1, 0]])
        res = pauli("X")

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_1(self):
        """Pauli-X operator with argument 1."""
        expected_res = np.array([[0, 1], [1, 0]])
        res = pauli(1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_y(self):
        """Pauli-Y operator with argument "Y"."""
        expected_res = np.array([[0, -1j], [1j, 0]])
        res = pauli("Y")

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_2(self):
        """Pauli-Y operator with argument 2."""
        expected_res = np.array([[0, -1j], [1j, 0]])
        res = pauli(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_z(self):
        """Pauli-Z operator with argument "Z"."""
        expected_res = np.array([[1, 0], [0, -1]])
        res = pauli("Z")

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_3(self):
        """Pauli-Z operator with argument 3."""
        expected_res = np.array([[1, 0], [0, -1]])
        res = pauli(3)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_int_list(self):
        """Test with list of Paulis of ints."""
        expected_res = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        )
        res = pauli([1, 1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pauli_str_list(self):
        """Test with list of Paulis of str."""
        expected_res = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        )
        res = pauli(["x", "x"])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
