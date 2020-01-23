from toqito.helper.constants import e0, e1
from toqito.matrix.operations.tensor import tensor_list
from toqito.states.w_state import w_state

import itertools
import unittest
import numpy as np


class TestWState(unittest.TestCase):
    """Unit test for w_state."""

    def test_w_state_3(self):
        """The 3-qubit W-state."""
        expected_res = 1/np.sqrt(3) * (tensor_list([e1, e0, e0]) +
                                       tensor_list([e0, e1, e0]) +
                                       tensor_list([e0, e0, e1]))
        
        res = w_state(3)

        bool_mat = np.isclose(res, expected_res, atol=0.2)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_generalized_w_state(self):
        """Generalized 4-qubit W-state."""
        expected_res = 1/np.sqrt(30) * (tensor_list([e1, e0, e0, e0]) +
                                        2 * tensor_list([e0, e1, e0, e0]) +
                                        3 * tensor_list([e0, e0, e1, e0]) +
                                        4 * tensor_list([e0, e0, e0, e1]))
        
        res = w_state(4, [1, 2, 3, 4]/np.sqrt(30))

        bool_mat = np.isclose(res, expected_res, atol=0.2)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_invalid_num_qubits(self):
        """Number of qubits needs to be greater than 2."""
        with self.assertRaises(ValueError):
            w_state(1)

    def test_invalid_coeff_list(self):
        """Length of coeff list needs to be equal to number of qubits."""
        with self.assertRaises(ValueError):
            w_state(4, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
