import unittest
import numpy as np
from toqito.states.purity import purity


class TestPurity(unittest.TestCase):
    """Unit test for purity."""

    def test_purity(self):
        expected_res = 1/4
        res = purity(np.identity(4)/4)
        self.assertEqual(res, expected_res)


if __name__ == '__main__':
    unittest.main()
