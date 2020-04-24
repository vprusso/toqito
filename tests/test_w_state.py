"""Tests for w_state function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.linear_algebra.operations.tensor import tensor
from toqito.states.states.w_state import w_state


class TestWState(unittest.TestCase):
    """Unit test for w_state."""

    def test_w_state_3(self):
        """The 3-qubit W-state."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = (
            1
            / np.sqrt(3)
            * (tensor(e_1, e_0, e_0) + tensor(e_0, e_1, e_0) + tensor(e_0, e_0, e_1))
        )

        res = w_state(3)

        bool_mat = np.isclose(res, expected_res, atol=0.2)
        self.assertEqual(np.all(bool_mat), True)

    def test_generalized_w_state(self):
        """Generalized 4-qubit W-state."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = (
            1
            / np.sqrt(30)
            * (
                tensor(e_1, e_0, e_0, e_0)
                + 2 * tensor(e_0, e_1, e_0, e_0)
                + 3 * tensor(e_0, e_0, e_1, e_0)
                + 4 * tensor(e_0, e_0, e_0, e_1)
            )
        )

        coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
        res = w_state(4, coeffs)

        bool_mat = np.isclose(res, expected_res, atol=0.2)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_num_qubits(self):
        """Number of qubits needs to be greater than 2."""
        with self.assertRaises(ValueError):
            w_state(1)

    def test_invalid_coeff_list(self):
        """Length of coeff list needs to be equal to number of qubits."""
        with self.assertRaises(ValueError):
            w_state(4, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
