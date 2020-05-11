"""Tests for max_entangled function."""
import unittest
import numpy as np

from toqito.states import basis, max_entangled


class TestMaxEntangled(unittest.TestCase):

    """Unit test for max_entangled."""

    def test_max_ent_2(self):
        """Generate maximally entangled state: `1/sqrt(2) * (|00> + |11>)`."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        expected_res = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
        res = max_entangled(2)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_max_ent_2_0_0(self):
        """Generate maximally entangled state: `|00> + |11>`."""
        e_0, e_1 = basis(2, 0), basis(2, 1)
        expected_res = 1 * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
        res = max_entangled(2, False, False)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)


if __name__ == "__main__":
    unittest.main()
