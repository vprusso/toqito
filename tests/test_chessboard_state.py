"""Tests for chessboard_state function."""
import unittest
import numpy as np

from toqito.states.states.chessboard_state import chessboard_state


class TestChessboardState(unittest.TestCase):
    """Unit test for chessboard_state."""

    def test_chessboard_state(self):
        """The chessboard_state."""
        res = chessboard_state([1, 2, 3, 4, 5, 6], 7, 8)
        self.assertEqual(np.isclose(res[0][0], 0.22592592592592592), True)

    def test_chessboard_state_default_s(self):
        """The chessboard_state."""
        res = chessboard_state([1, 2, 3, 4, 5, 6], 7)
        self.assertEqual(np.isclose(res[0][0], 0.29519938056523426), True)

    def test_chessboard_state_default_s_t(self):
        """The chessboard_state."""
        res = chessboard_state([1, 2, 3, 4, 5, 6])
        self.assertEqual(np.isclose(res[0][0], 0.3863449236810438), True)


if __name__ == '__main__':
    unittest.main()
