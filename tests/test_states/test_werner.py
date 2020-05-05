"""Tests for werner function."""
import unittest
import numpy as np

from toqito.states.states.werner import werner


class TestWerner(unittest.TestCase):
    """Unit test for werner."""

    def test_qutrit_werner(self):
        """Test for qutrit Werner state."""
        res = werner(3, 1 / 2)
        self.assertEqual(np.isclose(res[0][0], 0.0666666), True)
        self.assertEqual(np.isclose(res[1][3], -0.066666), True)

    def test_multipartite_werner(self):
        """Test for multipartite Werner state."""
        res = werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertEqual(np.isclose(res[0][0], 0.1127, atol=1e-02), True)

    def test_invalid_alpha(self):
        """Test for invalid `alpha` parameter."""
        with self.assertRaises(ValueError):
            werner(3, [1, 2])


if __name__ == "__main__":
    unittest.main()
