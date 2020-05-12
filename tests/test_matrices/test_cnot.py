"""Tests for cnot function."""
import unittest
import numpy as np

from toqito.matrices import cnot


class TestCNOT(unittest.TestCase):
    """Unit test for cnot."""

    def test_cnot(self):
        """Test standard CNOT gate."""
        res = cnot()
        expected_res = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        )
        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
