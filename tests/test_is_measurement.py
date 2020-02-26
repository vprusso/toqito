"""Tests for is_measurement function."""
import unittest
import numpy as np

from toqito.measure.is_measurement import is_measurement
from toqito.random.random_povm import random_povm


class TestIsMeasurement(unittest.TestCase):
    """Unit test for is_measurement."""

    def test_is_measurement_true(self):
        """Test if valid measurement returns True."""
        dim, num_inputs, num_outputs = 2, 2, 2
        measurements = random_povm(dim, num_inputs, num_outputs)
        
        self.assertEqual(is_measurement([measurements[:, :, 0, 0],
                                         measurements[:, :, 0, 1]]), True)

    def test_is_measurement_false(self):
        """Test if invalid measurement returns False."""
        non_meas_1 = np.array([[1, 2], [3, 4]])
        non_meas_2 = np.array([[5, 6], [7, 8]])
        non_meas = [non_meas_1, non_meas_2]

        self.assertEqual(is_measurement(non_meas), False)

    def test_is_measurement_false_not_sum_identity(self):
        """
        Test if invalid measurement (does not sum to identity)
        returns False.
        """
        non_meas_1 = np.array([[1, 0], [0, 1]])
        non_meas_2 = np.array([[1, 0], [0, 1]])
        non_meas = [non_meas_1, non_meas_2]

        self.assertEqual(is_measurement(non_meas), False)


if __name__ == '__main__':
    unittest.main()
