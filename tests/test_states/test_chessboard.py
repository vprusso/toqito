"""Test chessboard."""
import numpy as np

from toqito.states import chessboard


def test_chessboard():
    """The chessboard_state."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7, 8)
    np.testing.assert_equal(np.isclose(res[0][0], 0.22592592592592592), True)


def test_chessboard_default_s():
    """The chessboard_state with default `s_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6], 7)
    np.testing.assert_equal(np.isclose(res[0][0], 0.29519938056523426), True)


def test_chessboard_default_s_t():
    """The chessboard_state with default `s_param` and `t_param`."""
    res = chessboard([1, 2, 3, 4, 5, 6])
    np.testing.assert_equal(np.isclose(res[0][0], 0.3863449236810438), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
