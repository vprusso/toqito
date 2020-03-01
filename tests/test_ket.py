"""Tests for ket function."""
import unittest
import numpy as np

from toqito.base.ket import ket


class TestKet(unittest.TestCase):
    """Unit test for ket."""

    def test_ket_0(self):
        """Test for |0>."""
        expected_res = np.array([[1], [0]])
        res = ket(2, 0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_ket_1(self):
        """Test for |1>."""
        expected_res = np.array([[0], [1]])
        res = ket(2, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_ket_0000(self):
        """Test for |0000>."""
        expected_res = np.array([[1], [0], [0], [0]])
        res = ket(4, 0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_dim(self):
        """Tests for invalid dimension inputs."""
        with self.assertRaises(ValueError):
            ket(4, 4)


if __name__ == '__main__':
    unittest.main()
