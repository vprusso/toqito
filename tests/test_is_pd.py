from toqito.matrix.properties.is_pd import is_pd

import unittest
import numpy as np


class TestIsPD(unittest.TestCase):
    """Unit test for is_pd."""

    def test_is_pd(self):
        mat = np.array([[2, -1, 0],
                        [-1, 2, -1],
                        [0, -1, 2]])
        self.assertEqual(is_pd(mat), True)

    def test_is_not_pd(self):
        mat = np.array([[-1, -1],
                        [-1, -1]])
        self.assertEqual(is_pd(mat), False)

    def test_is_not_pd2(self):
        mat = np.array([[1, 2, 3],
                        [2, 1, 4]])
        self.assertEqual(is_pd(mat), False)


if __name__ == '__main__':
    unittest.main()
