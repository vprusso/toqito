"""Tests for state_distinguishability function."""
import unittest
import numpy as np

from toqito.base.ket import ket
from toqito.state.states.bell import bell
from toqito.state.optimizations.state_distinguishability import state_distinguishability


class TestStateDistinguishability(unittest.TestCase):
    """Unit test for state_distinguishability."""

    def test_state_distinguishability_one_state(self):
        """State distinguishability for single state."""
        rho = bell(0) * bell(0).conj().T
        states = [rho]

        res = state_distinguishability(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_distinguishability_one_state_vec(self):
        """State distinguishability for single vector state."""
        rho = bell(0)
        states = [rho]

        res = state_distinguishability(states)
        self.assertEqual(np.isclose(res, 1), True)

    def test_state_distinguishability_two_states(self):
        """State distinguishability for two state density matrices."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        e_00 = e_0 * e_0.conj().T
        e_11 = e_1 * e_1.conj().T
        states = [e_00, e_11]
        probs = [1 / 2, 1 / 2]

        res = state_distinguishability(states, probs)
        self.assertEqual(np.isclose(res, 1 / 2), True)

    def test_state_distinguishability_three_state_vec(self):
        """State distinguishability for two state vectors."""
        e_0, e_1 = ket(2, 0), ket(2, 1)
        states = [e_0, e_1]
        probs = [1 / 2, 1 / 2]

        res = state_distinguishability(states, probs)
        self.assertEqual(np.isclose(res, 1 / 2), True)

    def test_invalid_state_distinguishability_probs(self):
        """Invalid probability vector for state distinguishability."""
        with self.assertRaises(ValueError):
            rho1 = bell(0) * bell(0).conj().T
            rho2 = bell(1) * bell(1).conj().T
            states = [rho1, rho2]
            state_distinguishability(states, [1, 2, 3])

    def test_invalid_state_distinguishability_states(self):
        """Invalid number of states for state distinguishability."""
        with self.assertRaises(ValueError):
            states = []
            state_distinguishability(states)


if __name__ == "__main__":
    unittest.main()
