"""Tests for entropy function."""
import unittest
import numpy as np

from toqito.state.distance.entropy import entropy


class TestEntropy(unittest.TestCase):
    """Unit test for entropy."""

    def test_entropy_default(self):
        """Test entropy default arguments."""
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )

        res = entropy(test_input_mat)
        self.assertEqual(np.isclose(res, 0), True)

    def test_entropy_log_base_10(self):
        """Test entropy log base 10."""
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )

        res = entropy(test_input_mat, 10)
        self.assertEqual(np.isclose(res, 0), True)

    def test_entropy_inf_alpha(self):
        """Test entropy with infinite alpha."""
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )

        res = entropy(test_input_mat, log_base=2, alpha=float("inf"))
        self.assertEqual(np.isclose(res, 0), True)

    def test_entropy_identity(self):
        """Test entropy identity."""
        res = entropy(np.identity(4) / 4)
        self.assertEqual(np.isclose(res, 2), True)

    def test_invalid_alpha(self):
        """Tests for invalid alpha."""
        with self.assertRaises(ValueError):
            entropy(np.identity(4) / 4, 2, -1)


if __name__ == "__main__":
    unittest.main()
