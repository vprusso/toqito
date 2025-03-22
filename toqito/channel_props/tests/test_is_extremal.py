"""Tests for the is_extremal function in the toqito library."""

import unittest

import numpy as np

from toqito.channel_ops.kraus_to_choi import kraus_to_choi
from toqito.channel_props.is_extremal import is_extremal


class TestIsExtremal(unittest.TestCase):
    """Test suite for checking extremality of quantum channels."""

    def test_extremal_unitary_channel(self):
        """Test that a unitary channel is correctly identified as extremal."""
        U = np.array([[0, 1], [1, 0]])
        kraus_ops = [U]
        self.assertTrue(is_extremal(kraus_ops))
        self.assertTrue(is_extremal({"kraus": kraus_ops}))

    def test_non_extremal_channel(self):
        """Test that a non-extremal channel is correctly identified."""
        A1 = np.sqrt(0.5) * np.array([[1, 0], [0, 1]])
        A2 = np.sqrt(0.5) * np.array([[1, 0], [0, 1]])
        kraus_ops = [A1, A2]
        self.assertFalse(is_extremal(kraus_ops))
        self.assertFalse(is_extremal({"kraus": kraus_ops}))

    def test_choi_input(self):
        """Test that a channel provided as a Choi matrix is correctly processed."""
        U = np.array([[0, 1], [1, 0]])
        kraus_ops = [U]
        choi = kraus_to_choi(kraus_ops)
        self.assertTrue(is_extremal(choi))
        self.assertTrue(is_extremal({"choi": choi}))

    def test_example_from_watrous(self):
        """Test the example 2.33 from Watrous's book."""
        A0 = (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]])
        A1 = (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]])
        kraus_ops = [A0, A1]
        self.assertTrue(is_extremal(kraus_ops))

    def test_depolarizing_channel(self):
        """Test the depolarizing channel, which is not extremal for d>2."""
        d = 2
        p = 0.75
        identity_matrix = np.eye(d)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        K0 = np.sqrt(1 - 3 * p / 4) * identity_matrix
        K1 = np.sqrt(p / 4) * X
        K2 = np.sqrt(p / 4) * Y
        K3 = np.sqrt(p / 4) * Z
        kraus_ops = [K0, K1, K2, K3]
        self.assertFalse(is_extremal(kraus_ops))


if __name__ == "__main__":
    unittest.main()
