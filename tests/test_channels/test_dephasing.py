"""Tests for dephasing function."""
import unittest
import numpy as np

from toqito.channels import apply_map
from toqito.channels import dephasing


class TestDephasingChannel(unittest.TestCase):
    """Unit test for dephasing."""

    def test_completely_dephasing(self):
        """The completely dephasing channel kills everything off diagonal."""
        test_input_mat = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        expected_res = np.array(
            [[1, 0, 0, 0], [0, 6, 0, 0], [0, 0, 11, 0], [0, 0, 0, 16]]
        )

        res = apply_map(test_input_mat, dephasing(4))

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_partially_dephasing(self):
        """The partially dephasing channel for `p = 0.5`."""
        test_input_mat = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        expected_res = np.array(
            [[17.5, 0, 0, 0], [0, 20, 0, 0], [0, 0, 22.5, 0], [0, 0, 0, 25]]
        )

        res = apply_map(test_input_mat, dephasing(4, 0.5))

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
