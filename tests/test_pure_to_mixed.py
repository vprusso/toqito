"""Tests for pure_to_mixed function."""
import unittest
import numpy as np

from toqito.states.states.bell import bell
from toqito.states.operations.pure_to_mixed import pure_to_mixed


class TestPureToMixed(unittest.TestCase):
    """Unit test for pure_to_mixed."""

    def test_pure_to_mixed_state_vector(self):
        """Convert pure state to mixed state vector."""
        expected_res = np.array([[1/2, 0, 0, 1/2],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [1/2, 0, 0, 1/2]])

        phi = bell(0)
        res = pure_to_mixed(phi)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_pure_to_mixed_density_matrix(self):
        """Convert pure state to mixed state density matrix."""
        expected_res = np.array([[1/2, 0, 0, 1/2],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [1/2, 0, 0, 1/2]])

        phi = bell(0)*bell(0).conj().T
        res = pure_to_mixed(phi)

        bool_mat = np.isclose(res, expected_res)
        self.assertEqual(np.all(bool_mat), True)

    def test_invalid_pure_to_mixed_input(self):
        """Invalid arguments for pure_to_mixed."""
        with self.assertRaises(ValueError):
            non_valid_input = np.array([[1/2, 0, 0, 1/2],
                                        [1/2, 0, 0, 1/2]])
            pure_to_mixed(non_valid_input)


if __name__ == '__main__':
    unittest.main()
