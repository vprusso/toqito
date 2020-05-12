"""Tests for vec function."""
import unittest
import numpy as np

from toqito.matrix_ops import vec


class TestVec(unittest.TestCase):
    """Unit test for vec."""

    def test_vec(self):
        """Test standard vec operation on a matrix."""
        expected_res = np.array([[1], [3], [2], [4]])

        test_input_mat = np.array([[1, 2], [3, 4]])

        res = vec(test_input_mat)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
