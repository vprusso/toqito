"""Tests for bell function."""
import unittest
import numpy as np

from toqito.core.ket import ket
from toqito.states.states.bell import bell


class TestBell(unittest.TestCase):
    """Unit test for bell."""

    def test_bell_0(self):
        """Generate the Bell state: `1/sqrt(2) * (|00> + |11>)`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))

        res = bell(0)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bell_1(self):
        """Generates the Bell state: `1/sqrt(2) * (|00> - |11>)`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))

        res = bell(1)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bell_2(self):
        """Generates the Bell state: `1/sqrt(2) * (|01> + |10>)`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))

        res = bell(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bell_3(self):
        """Generates the Bell state: `1/sqrt(2) * (|01> - |10>)`."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))

        res = bell(3)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_bell_invalid(self):
        """Ensures that an integer above 3 is error-checked."""
        with self.assertRaises(ValueError):
            bell(4)


if __name__ == "__main__":
    unittest.main()
