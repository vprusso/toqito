from toqito.helper.constants import e0, e1
from toqito.states.ghz_state import ghz_state
from toqito.matrix.operations.tensor import tensor_list

import itertools
import unittest
import numpy as np


class TestGHZState(unittest.TestCase):
    """Unit test for ghz_state."""

    def test_ghz_2_3(self):
        """
        The following generates the 3-qubit GHZ state:
            1/sqrt(2) * (|000> + |111>)
        """
        expected_res = 1/np.sqrt(2) * (
                tensor_list([e0, e0, e0]) +
                tensor_list([e1, e1, e1]))

        res = ghz_state(2, 3).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_ghz_4_7(self):
        """
        The following generates the following GHZ state in (C^4)^{\otimes 7}:
        1/sqrt(30) * (|0000000> + 2|1111111> + 3|2222222> + 4|3333333>)
        """
        e0_4 = np.array([[1], [0], [0], [0]])
        e1_4 = np.array([[0], [1], [0], [0]])
        e2_4 = np.array([[0], [0], [1], [0]])
        e3_4 = np.array([[0], [0], [0], [1]])

        expected_res = 1/np.sqrt(30) * (
                tensor_list([e0_4, e0_4, e0_4, e0_4, e0_4, e0_4, e0_4]) +
                2 * tensor_list([e1_4, e1_4, e1_4, e1_4, e1_4, e1_4, e1_4]) +
                3 * tensor_list([e2_4, e2_4, e2_4, e2_4, e2_4, e2_4, e2_4]) +
                4 * tensor_list([e3_4, e3_4, e3_4, e3_4, e3_4, e3_4, e3_4]))

        res = ghz_state(4, 7, [1, 2, 3, 4]/np.sqrt(30)).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)





if __name__ == '__main__':
    unittest.main()

