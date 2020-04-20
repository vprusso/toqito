"""Tests for ghz function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.states.ghz import ghz
from toqito.linear_algebra.operations.tensor import tensor_list


class TestGHZ(unittest.TestCase):
    """Unit test for ghz."""

    def test_ghz_2_3(self):
        """Produces the 3-qubit GHZ state: `1/sqrt(2) * (|000> + |111>)`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = (
            1
            / np.sqrt(2)
            * (tensor_list([e_0, e_0, e_0]) + tensor_list([e_1, e_1, e_1]))
        )

        res = ghz(2, 3).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_ghz_4_7(self):
        r"""
        The following generates the following GHZ state in `(C^4)^{\otimes 7}`.

        `1/sqrt(30) * (|0000000> + 2|1111111> + 3|2222222> + 4|3333333>)`.
        """
        e0_4 = np.array([[1], [0], [0], [0]])
        e1_4 = np.array([[0], [1], [0], [0]])
        e2_4 = np.array([[0], [0], [1], [0]])
        e3_4 = np.array([[0], [0], [0], [1]])

        expected_res = (
            1
            / np.sqrt(30)
            * (
                tensor_list([e0_4, e0_4, e0_4, e0_4, e0_4, e0_4, e0_4])
                + 2 * tensor_list([e1_4, e1_4, e1_4, e1_4, e1_4, e1_4, e1_4])
                + 3 * tensor_list([e2_4, e2_4, e2_4, e2_4, e2_4, e2_4, e2_4])
                + 4 * tensor_list([e3_4, e3_4, e3_4, e3_4, e3_4, e3_4, e3_4])
            )
        )

        res = ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30)).toarray()

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_dim(self):
        """Tests for invalid dimensions."""
        with self.assertRaises(ValueError):
            ghz(1, 2)

    def test_invalid_qubits(self):
        """Tests for invalid number of qubits."""
        with self.assertRaises(ValueError):
            ghz(2, 1)

    def test_invalid_coeff(self):
        """Tests for invalid coefficients."""
        with self.assertRaises(ValueError):
            ghz(2, 3, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
