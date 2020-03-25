"""Tests for state_discrimination function."""
import unittest
import numpy as np

from toqito.base.ket import ket
from toqito.state.states.bell import bell
from toqito.state.optimizations.state_discrimination import state_discrimination


class TestStateDiscrimination(unittest.TestCase):
    """Unit test for state_discrimination."""

    def test_state_discrimination_one_state(self):
        """State discrimination for single state."""
        rho = bell(0) * bell(0).conj().T
        states = [rho]

        res = state_discrimination(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_discrimination_one_state_vec(self):
        """State discrimination for single vector state."""
        rho = bell(0)
        states = [rho]

        res = state_discrimination(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_discrimination_two_states(self):
        """State discrimination for two state density matrices."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_00 = e_0 * e_0.conj().T
        e_11 = e_1 * e_1.conj().T
        states = [e_00, e_11]
        probs = [1 / 2, 1 / 2]

        res = state_discrimination(states, probs)
        self.assertEqual(np.isclose(res, 1 / 2), True)

    def test_state_discrimination_three_state_vec(self):
        """State discrimination for two state vectors."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        states = [e_0, e_1]
        probs = [1 / 2, 1 / 2]

        res = state_discrimination(states, probs)
        self.assertEqual(np.isclose(res, 1 / 2), True)

    def test_invalid_state_discrim_probs(self):
        """Invalid probability vector."""
        with self.assertRaises(ValueError):
            rho1 = bell(0) * bell(0).conj().T
            rho2 = bell(1) * bell(1).conj().T
            states = [rho1, rho2]
            state_discrimination(states, [1, 2, 3])

    def test_invalid_state_discrim_states(self):
        """Invalid number of states."""
        with self.assertRaises(ValueError):
            states = []
            state_discrimination(states)


if __name__ == "__main__":
    unittest.main()
