"""Tests for conclusive_state_exclusion function."""
import unittest
import numpy as np

from toqito.state.states.bell import bell
from toqito.state.optimizations.conclusive_state_exclusion import \
    conclusive_state_exclusion


class TestConclusiveStateExclusion(unittest.TestCase):
    """Unit test for conclusive_state_exclusion."""

    def test_state_exclusion_one_state(self):
        """State exclusion for single state."""
        rho = bell(0) * bell(0).conj().T
        states = [rho]

        res = conclusive_state_exclusion(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_exclusion_one_state_vec(self):
        """State exclusion for single vector state."""
        rho = bell(0)
        states = [rho]

        res = conclusive_state_exclusion(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_exclusion_three_state(self):
        """State exclusion for three Bell state density matrices."""
        rho1 = bell(0) * bell(0).conj().T
        rho2 = bell(1) * bell(1).conj().T
        rho3 = bell(2) * bell(2).conj().T
        states = [rho1, rho2, rho3]
        probs = [1/3, 1/3, 1/3]

        res = conclusive_state_exclusion(states, probs)
        self.assertEqual(np.isclose(res, 0), True)

    def test_state_exclusion_three_state_vec(self):
        """State exclusion for three Bell state vectors."""
        rho1 = bell(0)
        rho2 = bell(1)
        rho3 = bell(2)
        states = [rho1, rho2, rho3]
        probs = [1/3, 1/3, 1/3]

        res = conclusive_state_exclusion(states, probs)
        self.assertEqual(np.isclose(res, 0), True)

    def test_invalid_state_exclusion_probs(self):
        """Invalid probability vector."""
        with self.assertRaises(ValueError):
            rho1 = bell(0) * bell(0).conj().T
            rho2 = bell(1) * bell(1).conj().T
            states = [rho1, rho2]
            conclusive_state_exclusion(states, [1, 2, 3])

    def test_invalid_state_exclusion_states(self):
        """Invalid number of states."""
        with self.assertRaises(ValueError):
            states = []
            conclusive_state_exclusion(states)


if __name__ == '__main__':
    unittest.main()
