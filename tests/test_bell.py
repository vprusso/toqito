"""Tests for bell function."""
import itertools
import unittest
import numpy as np

from toqito.helper.constants import e0, e1
from toqito.states.bell import bell


class TestBell(unittest.TestCase):
    """Unit test for bell."""

    def test_bell_0(self):
        """
        Generates the Bell state:
            1/sqrt(2) * (|00> + |11>)
        """
        expected_res = 1/np.sqrt(2) * (np.kron(e0, e0) + np.kron(e1, e1))

        res = bell(0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_bell_1(self):
        """
        Generates the Bell state:
            1/sqrt(2) * (|00> - |11>)
        """
        expected_res = 1/np.sqrt(2) * (np.kron(e0, e0) - np.kron(e1, e1))

        res = bell(1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_bell_2(self):
        """
        Generates the Bell state:
            1/sqrt(2) * (|01> + |10>)
        """
        expected_res = 1/np.sqrt(2) * (np.kron(e0, e1) + np.kron(e1, e0))

        res = bell(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_bell_3(self):
        """
        Generates the Bell state:
            1/sqrt(2) * (|01> - |10>)
        """
        expected_res = 1/np.sqrt(2) * (np.kron(e0, e1) - np.kron(e1, e0))

        res = bell(3)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(all(x == 1 for x in itertools.chain(*bool_mat)), True)

    def test_bell_invalid(self):
        """
        Ensures that an integer above 3 is error-checked.
        """
        with self.assertRaises(ValueError):
            bell(4)


if __name__ == '__main__':
    unittest.main()
