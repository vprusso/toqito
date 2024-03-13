"""Test channel dimensions."""

import numpy as np
import pytest

from toqito.helper import channel_dim
from toqito.matrices import pauli
from toqito.perms import swap_operator


def test_channel_dim_kraus():
    """Test channel dim of the superoperator PHI defined below.

    Phi(X) = [[1,5],[1,0],[0,2]] X [[0,1][2,3][4,5]].conj().T -
    [[1,0],[0,0],[0,1]] X [[0,0][1,1],[0,0]].conj().T
    """
    kraus_1 = np.array([[1, 5], [1, 0], [0, 2]])
    kraus_2 = np.array([[0, 1], [2, 3], [4, 5]])
    kraus_3 = np.array([[-1, 0], [0, 0], [0, -1]])
    kraus_4 = np.array([[0, 0], [1, 1], [0, 0]])

    din, dout, de = channel_dim([[kraus_1, kraus_2], [kraus_3, kraus_4]])
    np.testing.assert_equal((din, dout, de), (2, 3, 2))


@pytest.mark.parametrize("nested", [1, 2, 3])
def test_channel_dim_cpt_kraus(nested):
    """Test channel dim of a single qubit depolarizing channel."""
    kraus = [0.5 * pauli(ind) for ind in range(4)]
    if nested == 2:
        kraus = [kraus]
    elif nested == 3:
        kraus = [[mat] for mat in kraus]

    din, dout, de = channel_dim(kraus, allow_rect=False)
    np.testing.assert_equal((din, dout, de), (2, 2, 4))


def test_channel_dim_non_square():
    """Test channel dim of a channel with non square inputs."""
    kraus_1 = np.arange(6).reshape((2, 3))
    kraus_2 = np.arange(8).reshape((4, 2))
    din, dout, de = channel_dim([[kraus_1, kraus_2]])
    np.testing.assert_equal(din, np.array([3, 2]))
    np.testing.assert_equal(dout, np.array([2, 4]))
    np.testing.assert_equal(de, 1)


def test_channel_dim_non_square_and_allow_rect_disabled():
    """Test channel dim of a channel with non square inputs with allow_rect set to Falses."""
    kraus_1 = np.arange(6).reshape((2, 3))
    kraus_2 = np.arange(8).reshape((4, 2))
    with np.testing.assert_raises(ValueError):
        channel_dim([[kraus_1, kraus_2]], allow_rect=False)


def test_channel_dim_with_wrong_dim_input():
    """Test channel dim if user provided dim that doesn't match with the channel."""
    kraus = [0.5 * pauli(ind) for ind in range(4)]
    with np.testing.assert_raises(ValueError):
        channel_dim(kraus, dim=3)


def test_channel_dim_with_kraus_op_of_different_shapes():
    """Test channel dim when Kraus operators have different shapes."""
    kraus_1 = np.arange(6).reshape((2, 3))
    kraus_2 = np.arange(8).reshape((4, 2))
    with np.testing.assert_raises(ValueError):
        channel_dim([[kraus_1, kraus_2], [kraus_2, kraus_1]])


def test_channel_dim_choi():
    """Test channel dim of the transpose map."""
    din, dout, de = channel_dim(swap_operator(2))
    np.testing.assert_equal((din, dout, de), (2, 2, 4))


def test_channel_dim_choi_non_square():
    """Test channel dim of the transpose map."""
    din, dout, de = channel_dim(swap_operator([2, 3]), dim=np.array([[3, 2], [2, 3]]))
    np.testing.assert_equal(din, np.array([3, 2]))
    np.testing.assert_equal(dout, np.array([2, 3]))
    np.testing.assert_equal(de, 6)


def test_channel_dim_choi_non_square_and_allow_rect_disabled():
    """Test channel dim of a channel with non square inputs with allow_rect set to Falses."""
    with np.testing.assert_raises(ValueError):
        channel_dim(swap_operator([2, 3]), dim=np.array([[3, 2], [2, 3]]), allow_rect=False)


def test_channel_dim_choi_with_wrong_dim_input():
    """Test channel dim of the transpose map but withn input dim mismatch."""
    with np.testing.assert_raises(ValueError):
        channel_dim(swap_operator(3), dim=[2, 3])


def test_channel_dim_with_invalid_dim_input():
    """Test channel dim with invalid dim."""
    with np.testing.assert_raises(ValueError):
        channel_dim(swap_operator(3), dim=np.eye(3))


def test_channel_dim_with_vector_dim_input():
    """Test channel dim with vector dim."""
    v_mat = np.array([[1, 0, 0], [0, 1, 0]])
    din, dout, de = channel_dim([v_mat], dim=[3, 2])
    np.testing.assert_equal((din, dout, de), (3, 2, 1))


def test_channel_dim_with_compute_env_dim_disabled():
    """Test channel dim without computing the enviroment dimension."""
    din, dout, de = channel_dim(swap_operator(2), compute_env_dim=False)
    np.testing.assert_equal((din, dout, de), (2, 2, None))
