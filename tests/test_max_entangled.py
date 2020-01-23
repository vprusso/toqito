from toqito.helper.constants import e0, e1
from toqito.states.max_entangled import max_entangled

import itertools
import unittest
import numpy as np


class TestMaxEntangled(unittest.TestCase):
    """Unit test for max_entangled."""

    def test_max_ent_2(self):
        expected_res = 1/np.sqrt(2) * (np.kron(e0, e0) + np.kron(e1, e1))
        res = max_entangled(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_max_ent_2_0_0(self):
        expected_res = 1 * (np.kron(e0, e0) + np.kron(e1, e1))
        res = max_entangled(2, 0, 0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)


if __name__ == '__main__':
    unittest.main()

