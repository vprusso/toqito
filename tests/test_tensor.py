from toqito.helper.constants import e0, e1
from toqito.matrix.operations.tensor import tensor, tensor_n, tensor_list

import itertools
import numpy as np
import unittest


class TestTensor(unittest.TestCase):
    """Unit test for tensor."""

    def test_tensor(self):
        expected_res = np.kron(e0, e0)

        res = tensor(e0, e0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_n_0(self):
        expected_res = None

        res = tensor_n(e0, 0)
        self.assertEqual(res, expected_res)

    def test_tensor_n_1(self):
        expected_res = e0

        res = tensor_n(e0, 1)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_n_2(self):
        expected_res = np.kron(e0, e0)

        res = tensor_n(e0, 2)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_n_3(self):
        expected_res = np.kron(np.kron(e0, e0), e0)

        res = tensor_n(e0, 3)
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_list_0(self):
        expected_res = None

        res = tensor_list([])
        self.assertEqual(res, expected_res)

    def test_tensor_list_1(self):
        expected_res = e0

        res = tensor_list([e0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_list_2(self):
        expected_res = np.kron(e0, e1)

        res = tensor_list([e0, e1])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_tensor_list_3(self):
        expected_res = np.kron(np.kron(e0, e1), e0)

        res = tensor_list([e0, e1, e0])

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()
