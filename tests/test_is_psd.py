"""Tests for is_psd function."""
import unittest
import numpy as np

from toqito.matrix.properties.is_psd import is_psd


class TestIsPSD(unittest.TestCase):
    """Unit test for is_psd."""

    def test_is_psd(self):
        """Test that positive semidefinite matrix returns True."""
        mat = np.array([[1, -1],
                        [-1, 1]])
        self.assertEqual(is_psd(mat), True)

    def test_is_not_psd(self):
        """Test that non-positive semidefinite matrix returns False."""
        mat = np.array([[-1, -1],
                        [-1, -1]])
        self.assertEqual(is_psd(mat), False)


if __name__ == '__main__':
    unittest.main()
