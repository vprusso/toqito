"""Tests for werner_state function."""
import unittest
import numpy as np

from toqito.state.states.werner_state import werner_state


class TestWernerState(unittest.TestCase):
    """Unit test for werner_state."""

    def test_qutrit_werner(self):
        """Test for qutrit Werner state."""
        res = werner_state(3, 1/2)
        print(res[0][0])
        print(2/3)
        self.assertEqual(np.isclose(res[0][0], 0.0666666), True)
        self.assertEqual(np.isclose(res[1][3], -0.066666), True)

    def test_multipartite_werner(self):
        """Test for multipartite Werner state."""
        res = werner_state(2, [0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertEqual(np.isclose(res[0][0], 0.1127, atol=1e-02), True)


if __name__ == '__main__':
    unittest.main()
