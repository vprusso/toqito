"""Tests for is_completely_positive function."""
import unittest
import numpy as np

from toqito.channels.properties.is_completely_positive import is_completely_positive
from toqito.channels.channels.depolarizing import depolarizing


class TestIsCompletelyPositive(unittest.TestCase):
    """Unit test for is_completely_positive."""

    def test_is_completely_positive_kraus_false(self):
        """Verify non-completely positive channel as Kraus ops as False."""
        unitary_mat = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
        kraus_ops = [[np.identity(2), np.identity(2)], [unitary_mat, -unitary_mat]]

        self.assertEqual(is_completely_positive(kraus_ops), False)

    def test_is_completely_positive_choi_true(self):
        """
        Verify that the Choi matrix of the depolarizing map is completely
        positive.
        """
        self.assertEqual(is_completely_positive(depolarizing(2)), True)


if __name__ == "__main__":
    unittest.main()
