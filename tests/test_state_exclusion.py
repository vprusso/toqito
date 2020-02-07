"""Tests for state_exclusion function."""
import unittest
import numpy as np

from toqito.states.bell import bell
from toqito.states.state_exclusion import state_exclusion


class TestStateExclusion(unittest.TestCase):
    """Unit test for state_exclusion."""

    def test_state_exclusion_one_state(self):
        """State exclusion for single state."""
        rho = bell(0) * bell(0).conj().T
        states = [rho]

        res = state_exclusion(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_exclusion_three_state(self):
        """State exclusion for single state."""
        rho1 = bell(0) * bell(0).conj().T
        rho2 = bell(1) * bell(1).conj().T
        rho3 = bell(2) * bell(2).conj().T
        states = [rho1, rho2, rho3]
        probs = [1/3, 1/3, 1/3]

        res = state_exclusion(states, probs)
        self.assertEqual(np.isclose(res, 0), True)

    def test_invalid_probs(self):
        """Invalid probability vector."""
        with self.assertRaises(ValueError):
            rho1 = bell(0) * bell(0).conj().T
            rho2 = bell(1) * bell(1).conj().T
            states = [rho1, rho2]
            state_exclusion(states, [1, 2, 3])

    def test_invalid_states(self):
        """Invalid number of states."""
        with self.assertRaises(ValueError):
            states = []
            state_exclusion(states)


if __name__ == '__main__':
    unittest.main()
