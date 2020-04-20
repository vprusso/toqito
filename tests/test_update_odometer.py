"""Tests for update_odometer function."""
import unittest
import numpy as np

from toqito.helper.update_odometer import update_odometer


class TestUpdateOdometer(unittest.TestCase):
    """Unit test for update_odometer."""

    def test_update_odometer_0_0(self):
        """Update odometer from [2, 2] to [0, 0]."""
        vec = np.array([2, 2])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([0, 0], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_0_1(self):
        """Update odometer from [0, 0] to [0, 1]."""
        vec = np.array([0, 0])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([0, 1], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_1_0(self):
        """Update odometer from [0, 1] to [1, 0]."""
        vec = np.array([0, 1])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([1, 0], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_2_0(self):
        """Update odometer from [1, 1] to [2, 0]."""
        vec = np.array([1, 1])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([2, 0], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_2_1(self):
        """Update odometer from [2, 0] to [2, 1]."""
        vec = np.array([2, 0])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([2, 1], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_2_2(self):
        """Update odometer from [2, 1] to [0, 0]."""
        vec = np.array([2, 1])
        upper_lim = np.array([3, 2])
        res = update_odometer(vec, upper_lim)

        bool_mat = np.isclose([0, 0], res)
        self.assertEqual(np.all(bool_mat), True)

    def test_update_odometer_empty(self):
        """Return `None` if empty lists are provided."""
        vec = np.array([])
        upper_lim = np.array([])
        res = update_odometer(vec, upper_lim)

        self.assertEqual(res, None)


if __name__ == "__main__":
    unittest.main()
