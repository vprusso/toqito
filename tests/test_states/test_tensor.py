"""Tests for tensor function."""
import unittest
import numpy as np

from toqito.states import basis
from toqito.state_ops import tensor


class TestTensor(unittest.TestCase):
    """Unit tests for tensor."""

    def test_tensor(self):
        """Test standard tensor on vectors."""
        e_0 = basis(2, 0)
        expected_res = np.kron(e_0, e_0)

        res = tensor(e_0, e_0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_single_arg(self):
        """Performing tensor product on one item should return item back."""
        input_arr = np.array([[1, 2], [3, 4]])
        res = tensor(input_arr)

        bool_mat = np.isclose(res, input_arr)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_multiple_args(self):
        """Performing tensor product on multiple matrices."""
        input_arr_1 = np.identity(2)
        input_arr_2 = np.identity(2)
        input_arr_3 = np.identity(2)
        input_arr_4 = np.identity(2)
        res = tensor(input_arr_1, input_arr_2, input_arr_3, input_arr_4)

        bool_mat = np.isclose(res, np.identity(16))
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_0(self):
        """Test tensor n=0 times."""
        e_0 = basis(2, 0)
        expected_res = None

        res = tensor(e_0, 0)
        self.assertEqual(res, expected_res)

    def test_tensor_n_1(self):
        """Test tensor n=1 times."""
        e_0 = basis(2, 0)
        expected_res = e_0

        res = tensor(e_0, 1)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_2(self):
        """Test tensor n=2 times."""
        e_0 = basis(2, 0)
        expected_res = np.kron(e_0, e_0)

        res = tensor(e_0, 2)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_n_3(self):
        """Test tensor n=3 times."""
        e_0 = basis(2, 0)
        expected_res = np.kron(np.kron(e_0, e_0), e_0)

        res = tensor(e_0, 3)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_0(self):
        """Test tensor empty list."""
        expected_res = None

        res = tensor([])
        self.assertEqual(res, expected_res)

    def test_tensor_list_1(self):
        """Test tensor list with one item."""
        e_0 = basis(2, 0)
        expected_res = e_0

        res = tensor([e_0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_2(self):
        """Test tensor list with two items."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        expected_res = np.kron(e_0, e_1)

        res = tensor([e_0, e_1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_list_3(self):
        """Test tensor list with three items."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        expected_res = np.kron(np.kron(e_0, e_1), e_0)

        res = tensor([e_0, e_1, e_0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_tensor_empty_args(self):
        r"""Test tensor with no arguments."""
        with self.assertRaises(ValueError):
            tensor()


if __name__ == "__main__":
    unittest.main()
