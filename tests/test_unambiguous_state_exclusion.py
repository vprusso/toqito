"""Tests for unambiguous_state_exclusion function."""
import unittest
import numpy as np

from toqito.state.states.bell import bell
from toqito.state.optimizations.unambiguous_state_exclusion import (
    unambiguous_state_exclusion,
)


class TestUnambiguousStateExclusion(unittest.TestCase):
    """Unit test for unambiguous_state_exclusion."""

    def test_unambiguous_state_exclusion_one_state(self):
        """Unambiguous state exclusion for single state."""
        mat = bell(0) * bell(0).conj().T
        states = [mat]

        res = unambiguous_state_exclusion(states)
        self.assertEqual(np.isclose(res, 0), True)

    def test_unambiguous_state_exclusion_one_state_vec(self):
        """Unambiguous state exclusion for single vector state."""
        vec = bell(0)
        states = [vec]

        res = unambiguous_state_exclusion(states)
        self.assertEqual(np.isclose(res, 0), True)

    def test_unambiguous_state_exclusion_three_state(self):
        """Unambiguous state exclusion for three Bell state density matrices."""
        mat1 = bell(0) * bell(0).conj().T
        mat2 = bell(1) * bell(1).conj().T
        mat3 = bell(2) * bell(2).conj().T
        states = [mat1, mat2, mat3]
        probs = [1 / 3, 1 / 3, 1 / 3]

        res = unambiguous_state_exclusion(states, probs)
        self.assertEqual(np.isclose(res, 0), True)

    def test_unambiguous_state_exclusion_three_state_vec(self):
        """Unambiguous state exclusion for three Bell state vectors."""
        mat1 = bell(0)
        mat2 = bell(1)
        mat3 = bell(2)
        states = [mat1, mat2, mat3]
        probs = [1 / 3, 1 / 3, 1 / 3]

        res = unambiguous_state_exclusion(states, probs)
        self.assertEqual(np.isclose(res, 0), True)

    def test_invalid_unambiguous_state_exclusion_probs(self):
        """Invalid probability vector for unambiguous state exclusion."""
        with self.assertRaises(ValueError):
            mat1 = bell(0) * bell(0).conj().T
            mat2 = bell(1) * bell(1).conj().T
            states = [mat1, mat2]
            unambiguous_state_exclusion(states, [1, 2, 3])

    def test_invalid_unambiguous_state_exclusion_states(self):
        """Invalid number of states for unambiguous state exclusion."""
        with self.assertRaises(ValueError):
            states = []
            unambiguous_state_exclusion(states)


if __name__ == "__main__":
    unittest.main()
