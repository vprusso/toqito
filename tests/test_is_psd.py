from toqito.matrix.properties.is_psd import is_psd

import unittest
import numpy as np


class TestIsPSD(unittest.TestCase):
    """Unit test for is_psd."""

    def test_is_psd(self):
        mat = np.array([[1, -1],
                        [-1, 1]])
        self.assertEqual(is_psd(mat), True)

    def test_is_not_psd(self):
        mat = np.array([[-1, -1],
                        [-1, -1]])
        self.assertEqual(is_psd(mat), False)


if __name__ == '__main__':
    unittest.main()
