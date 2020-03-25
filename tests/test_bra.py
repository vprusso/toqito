"""Tests for bra function."""
import unittest
import numpy as np

from toqito.base.bra import bra


class TestBra(unittest.TestCase):
    """Unit test for bra."""

    def test_bra_0(self):
        """Test for <0|."""
        expected_res = np.array([1, 0])
        res = bra(2, 0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bra_1(self):
        """Test for <1|."""
        expected_res = np.array([0, 1])
        res = bra(2, 1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bra_0000(self):
        """Test for <0000|."""
        expected_res = np.array([1, 0, 0, 0])
        res = bra(4, 0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_dim(self):
        """Tests for invalid dimension inputs."""
        with self.assertRaises(ValueError):
            bra(4, 4)


if __name__ == "__main__":
    unittest.main()
