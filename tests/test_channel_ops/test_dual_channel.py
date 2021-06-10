"""Tests for dual_channel"""
import numpy as np

from toqito.channel_ops import dual_channel
from toqito.channels import choi


def test_dual_channel_kraus():
    """Test dual_channel on a channel represented as Kraus operators."""
    kraus_1 = np.array([[1, 0, 1j, 0]])
    kraus_2 = np.array([[0, 1, 0, 1j]])

    expected_res = [
    [np.array([[1, 0, -1j, 0]]).T, np.array([[1, 0, -1j, 0]]).T],
    [np.array([[0, 1, 0, -1j]]).T, np.array([[0, 1, 0, -1j]]).T]
    ]

    res = dual_channel([[kraus_1, kraus_1], [kraus_2, kraus_2]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_dual_channel_choi_square():
    """Test dual_channel on a 9x9 Choi matrix, inferring dims=[3,3]."""
    res = dual_channel(choi(1, 1, 0))

    bool_mat = np.isclose(res, choi(1, 0, 1))
    np.testing.assert_equal(np.all(bool_mat), True)

def test_dual_channel_choi_dims():
    """Test dual_channel on a Choi matrix with different input and output dimensions."""
    j = np.array(
        [
            [1, -1j, 0, 0, 0, 1],
            [1j, -1, 0, 0, 0, -1j],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1j, 0, 0, 0, 1]
        ]
    )

    expected_res = np.array(
        [
            [1, 0, 0, 1j, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-1j, 0, 0, -1, 0, 1j],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, -1j, 0, 1]
        ]
    )

    res = dual_channel(j, [3,2])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)

def test_dual_channel_nonsquare_matrix():
    """If the channel is represented as a Choi matrix, it must be square."""
    with np.testing.assert_raises(ValueError):
        j = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        dual_channel(j)

def test_dual_channel_not_matrix():
    """If the channel is represented as an array, it must be two-dimensional (a matrix)."""
    with np.testing.assert_raises(ValueError):
        j = np.array([1, 2, 3, 4])
        dual_channel(j)

def test_dual_channel_unspecified_dims():
    """If the size of the Choi matrix is not a perfect square,
    the dimensions of the input and output spaces must be specified."""
    with np.testing.assert_raises(ValueError):
        j = np.arange(36).reshape(6,6)
        dual_channel(j)

def test_dual_channel_invalid_input():
    """Invalid input"""
    with np.testing.assert_raises(ValueError):
        dual_channel(0)


if __name__ == "__main__":
    np.testing.run_module_suite()
