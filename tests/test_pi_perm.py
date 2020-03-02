"""Tests for pi_perm function."""
import unittest
import numpy as np

from toqito.perms.permutation_operator import permutation_operator
from toqito.perms.pi_perm import pi_perm


class TestPiPerm(unittest.TestCase):
    """Unit test for pi_perm."""

    def test_pi_perm_1(self):
        """Test for pi_perm with dimension equal to 1."""
        dim = 1
        expected_res = np.identity(dim)

        res = pi_perm(dim)
        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pi_perm_2(self):
        """Test for pi_perm with dimension equal to 2."""
        dim = 2
        expected_res = permutation_operator(2, [1, 3, 2, 4])

        res = pi_perm(dim)
        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pi_perm_3(self):
        """Test for pi_perm with dimension equal to 3."""
        dim = 3
        expected_res = permutation_operator(2, [1, 4, 2, 5, 3, 6])

        res = pi_perm(dim)
        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pi_perm_4(self):
        """Test for pi_perm with dimension equal to 4."""
        dim = 4
        expected_res = permutation_operator(2, [1, 5, 2, 6, 3, 7, 4, 8])

        res = pi_perm(dim)
        bool_mat = np.isclose(expected_res, res)
        self.assertEqual(np.all(bool_mat), True)
