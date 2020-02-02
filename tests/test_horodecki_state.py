"""Tests for horodeck_state function."""
import unittest
import numpy as np

from toqito.states.horodecki_state import horodecki_state


class TestHorodeckiState(unittest.TestCase):
    """Unit test for horodecki_state."""

    def test_horodecki_state_3_3_default(self):
        """The 3-by-3 Horodecki state (no dimensions specified on input)."""
        expected_res = np.array(
            [[0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
             [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
             [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
             [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
             [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
             [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500]])

        res = horodecki_state(0.5)
        bool_mat = np.isclose(expected_res, res, atol=0.0001)
        self.assertEqual(np.all(bool_mat), True)

    def test_horodecki_state_3_3(self):
        """The 3-by-3 Horodecki state."""
        expected_res = np.array(
            [[0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
             [0, 0.1000, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0.1000, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0.1000, 0, 0, 0, 0, 0],
             [0.1000, 0, 0, 0, 0.1000, 0, 0, 0, 0.1000],
             [0, 0, 0, 0, 0, 0.1000, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0.1500, 0, 0.0866],
             [0, 0, 0, 0, 0, 0, 0, 0.1000, 0],
             [0.1000, 0, 0, 0, 0.1000, 0, 0.0866, 0, 0.1500]])

        res = horodecki_state(0.5, [3, 3])
        bool_mat = np.isclose(expected_res, res, atol=0.0001)
        self.assertEqual(np.all(bool_mat), True)

    def test_horodecki_state_2_4(self):
        """The 2-by-4 Horodecki state."""
        expected_res = np.array(
            [[0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
             [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
             [0, 0, 0.1111, 0, 0, 0, 0, 0.1111],
             [0, 0, 0, 0.1111, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0.1667, 0, 0.0962],
             [0.1111, 0, 0, 0, 0, 0.1111, 0, 0],
             [0, 0.1111, 0, 0, 0, 0, 0.1111, 0],
             [0, 0, 0.1111, 0, 0, 0.0962, 0, 0.1667]])

        res = horodecki_state(0.5, [2, 4])
        bool_mat = np.isclose(expected_res, res, atol=0.2)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_a_param(self):
        """Tests for invalid a_param inputs."""
        with self.assertRaises(ValueError):
            horodecki_state(-5)
        with self.assertRaises(ValueError):
            horodecki_state(5)

    def test_invalid_dim(self):
        """Tests for invalid dimension inputs."""
        with self.assertRaises(ValueError):
            horodecki_state(0.5, [3, 4])


if __name__ == '__main__':
    unittest.main()
