"""Tests for negativity function."""
import unittest
import numpy as np

from toqito.entanglement.negativity import negativity


class TestNegativity(unittest.TestCase):
    """Unit test for negativity."""

    def test_negativity_rho(self):
        """Test for negativty on rho."""
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )

        self.assertEqual(np.isclose(negativity(test_input_mat), 1 / 2), True)

    def test_negativity_rho_dim_int(self):
        """Test for negativty on rho."""
        test_input_mat = np.array(
            [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
        )

        self.assertEqual(np.isclose(negativity(test_input_mat, 2), 1 / 2), True)

    def test_invalid_rho_dim_int(self):
        """Invalid dim parameters."""
        with self.assertRaises(ValueError):
            test_input_mat = np.array(
                [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
            )
            negativity(test_input_mat, 5)

    def test_invalid_rho_dim_vec(self):
        """Invalid dim parameters."""
        with self.assertRaises(ValueError):
            test_input_mat = np.array(
                [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
            )
            negativity(test_input_mat, [2, 5])


if __name__ == "__main__":
    unittest.main()
