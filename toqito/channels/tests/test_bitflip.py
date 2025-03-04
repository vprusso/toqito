"""Test cases for the bit-flip channel in Toqito."""

import unittest

import numpy as np

from toqito.channels import bitflip


class TestBitflipChannel(unittest.TestCase):
    """Tests the bitflip implementation."""

    def test_kraus_operators(self):
        """Test if the function returns correct Kraus operators for given probability."""
        prob = 0.3
        kraus_ops = bitflip(prob=prob)
        expected_k0 = np.sqrt(1 - prob) * np.eye(2)
        expected_k1 = np.sqrt(prob) * np.array([[0, 1], [1, 0]])

        np.testing.assert_almost_equal(kraus_ops[0], expected_k0)
        np.testing.assert_almost_equal(kraus_ops[1], expected_k1)

    def test_apply_to_state_0(self):
        """Test bitflip application to |0><0| state."""
        prob = 0.3
        rho = np.array([[1, 0], [0, 0]])  # |0><0|
        expected_output = (1 - prob) * rho + prob * np.array([[0, 0], [0, 1]])
        result = bitflip(rho, prob=prob)

        np.testing.assert_almost_equal(result, expected_output)

    def test_apply_to_state_1(self):
        """Test bitflip application to |1><1| state."""
        prob = 0.3
        rho = np.array([[0, 0], [0, 1]])  # |1><1|
        expected_output = (1 - prob) * rho + prob * np.array([[1, 0], [0, 0]])
        result = bitflip(rho, prob=prob)

        np.testing.assert_almost_equal(result, expected_output)

    def test_probability_0(self):
        """Test bitflip with probability 0 (no change)."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = bitflip(rho, prob=0)
        np.testing.assert_almost_equal(result, rho)

    def test_probability_1(self):
        """Test bitflip with probability 1 (always flips)."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        expected_output = np.array([[0.5, 0.5], [0.5, 0.5]])  # Same for maximally mixed state
        result = bitflip(rho, prob=1)
        np.testing.assert_almost_equal(result, expected_output)

    def test_invalid_probability_negative(self):
        """Test that a negative probability raises an error."""
        with self.assertRaises(ValueError):
            bitflip(prob=-0.1)

    def test_invalid_probability_greater_than_1(self):
        """Test that a probability greater than 1 raises an error."""
        with self.assertRaises(ValueError):
            bitflip(prob=1.1)

    def test_invalid_dimension(self):
        """Test that invalid dimensions raise an error."""
        with self.assertRaises(ValueError):
            bitflip(prob=0.3, dim=3)

    def test_apply_to_mixed_state(self):
        """Test bitflip channel on a mixed state."""
        prob = 0.4
        rho = np.array([[0.7, 0.2], [0.2, 0.3]])
        expected_output = (1 - prob) * rho + prob * np.array([[0.3, 0.2], [0.2, 0.7]])
        result = bitflip(rho, prob=prob)

        np.testing.assert_almost_equal(result, expected_output)


if __name__ == "__main__":
    unittest.main()
