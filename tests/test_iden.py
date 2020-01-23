from toqito.helper.iden import iden

import itertools
import unittest
import numpy as np


class TestIden(unittest.TestCase):
    """Unit test for iden."""

    def test_iden_full(self):
        expected_res = np.array([[1, 0],
                                 [0, 1]])
        res = iden(2, 0)
       
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_iden_sparse(self):
        expected_res = np.array([[1, 0], [0, 1]])
        res = iden(2, 1).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()

