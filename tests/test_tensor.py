"""Tests for tensor function."""
import unittest
import numpy as np

from toqito.base.ket import ket
from toqito.matrix.operations.tensor import tensor, tensor_n, tensor_list


class TestTensor(unittest.TestCase):
    """Unit test for tensor."""

    def test_tensor(self):
        """Test standard tensor on vectors."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = np.kron(e0, e0)

        res = tensor(e0, e0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_0(self):
        """Test tensor n=0 times."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = None

        res = tensor_n(e0, 0)
        self.assertEqual(res, expected_res)

    def test_tensor_n_1(self):
        """Test tensor n=1 times."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = e0

        res = tensor_n(e0, 1)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_2(self):
        """Test tensor n=2 times."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = np.kron(e0, e0)

        res = tensor_n(e0, 2)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_3(self):
        """Test tensor n=3 times."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = np.kron(np.kron(e0, e0), e0)

        res = tensor_n(e0, 3)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_0(self):
        """Test tensor empty list."""
        expected_res = None

        res = tensor_list([])
        self.assertEqual(res, expected_res)

    def test_tensor_list_1(self):
        """Test tensor list with one item."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = e0

        res = tensor_list([e0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_2(self):
        """Test tensor list with two items."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = np.kron(e0, e1)

        res = tensor_list([e0, e1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_3(self):
        """Test tensor list with three items."""
        e0, e1 = ket(2, 0), ket(2, 1)
        expected_res = np.kron(np.kron(e0, e1), e0)

        res = tensor_list([e0, e1, e0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
