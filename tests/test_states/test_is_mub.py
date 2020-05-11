"""Tests for is_mub function."""
import unittest
import numpy as np

from toqito.states import basis
from toqito.state_props import is_mub


class TestIsMUB(unittest.TestCase):

    """Unit test for is_mub."""

    def test_is_mub_dim_2(self):
        """Return True for MUB of dimension 2."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        mub_1 = [e_0, e_1]
        mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)]
        mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)]
        mubs = [mub_1, mub_2, mub_3]
        self.assertEqual(is_mub(mubs), True)

    def test_is_not_mub_dim_2(self):
        """Return False for non-MUB of dimension 2."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        mub_1 = [e_0, e_1]
        mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), e_1]
        mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0]
        mubs = [mub_1, mub_2, mub_3]
        self.assertEqual(is_mub(mubs), False)

    def test_invalid_input_len(self):
        """Tests for invalid input len."""
        with self.assertRaises(ValueError):
            vec_list = [np.array([1, 0])]
            is_mub(vec_list)


if __name__ == "__main__":
    unittest.main()
