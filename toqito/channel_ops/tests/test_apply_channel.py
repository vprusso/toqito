"""Tests for apply_channel."""

import numpy as np
import pytest

from toqito.channel_ops import apply_channel
from toqito.matrices import pauli
from toqito.perms import swap_operator

kraus_1 = np.array([[1, 5], [1, 0], [0, 2]])
kraus_2 = np.array([[0, 1], [2, 3], [4, 5]])
kraus_3 = np.array([[-1, 0], [0, 0], [0, -1]])
kraus_4 = np.array([[0, 0], [1, 1], [0, 0]])


@pytest.mark.parametrize(
    "input_mat, expected_result, apply_channel_arg2",
    [
        # The swap operator is the Choi matrix of the transpose map.
        # The following test is a (non-ideal, but illustrative) way of computing the transpose of a matrix.
        (np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), swap_operator(3)),
        # The swap operator is the Choi matrix of the transpose map.
        # The following test is a (non-ideal, but illustrative) way of computing the transpose of a non-square matrix.
        (np.array([[0, 1], [2, 3], [4, 5]]), np.array([[0, 2, 4], [1, 3, 5]]), swap_operator([2, 3])),
        # Apply Kraus map.
        # The following test computes PHI(X) where X = [[1, 2], [3, 4]] and where PHI is the superoperator defined by:
        # Phi(X) = [[1,5],[1,0],[0,2]] X [[0,1][2,3][4,5]].conj().T -[[1,0],[0,0],[0,1]] X [[0,0][1,1],[0,0]].conj().T
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[22, 95, 174], [2, 8, 14], [8, 29, 64]]),
            [[kraus_1, kraus_2], [kraus_3, kraus_4]],
        ),
    ],
)
def test_apply_channel(input_mat, expected_result, apply_channel_arg2):
    """Test function works as expected for valid inputs."""
    calculated_result = apply_channel(input_mat, apply_channel_arg2)
    assert (calculated_result == expected_result).all()


@pytest.mark.parametrize("nested", [1, 2, 3])
def test_apply_channel_cpt_kraus(nested):
    """Apply Kraus map of single qubit depolarizing channel."""
    test_input_mat = np.array([[1, 0], [0, 0]])

    expected_res = np.array([[0.5, 0], [0, 0.5]])

    kraus = [0.5 * pauli(ind) for ind in range(4)]
    if nested == 2:
        kraus = [kraus]
    elif nested == 3:
        kraus = [[mat] for mat in kraus]

    res = apply_channel(test_input_mat, kraus)
    assert (res == expected_res).all()


def test_apply_channel_invalid_input():
    """Invalid input for apply map."""
    with pytest.raises(ValueError):
        apply_channel(np.array([[1, 2], [3, 4]]), 2)
