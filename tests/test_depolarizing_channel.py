"""Tests for depolarizing_channel function."""
import unittest
import numpy as np

from toqito.super_operators.apply_map import apply_map
from toqito.super_operators.depolarizing_channel import depolarizing_channel


class TestDepolarizingChannel(unittest.TestCase):
    """Unit test for depolarizing_channel."""

    def test_standard_depolarizing_channel(self):
        """Maps every density matrix to the maximally-mixed state."""
        test_input_mat = np.array([[1/2, 0, 0, 1/2],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [1/2, 0, 0, 1/2]])

        expected_res = 1/4 * np.array([[1/2, 0, 0, 1/2],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [1/2, 0, 0, 1/2]])

        res = apply_map(test_input_mat, depolarizing_channel(4))

        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == '__main__':
    unittest.main()
