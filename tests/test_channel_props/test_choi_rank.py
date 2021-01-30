"""Tests for choi_rank."""
import numpy as np
import pytest

from toqito.channel_props import choi_rank


def test_choi_rank_list_kraus():
    """Verify that a list of Kraus operators gives correct Choi rank"""
    kraus_1 = np.array([[1, 0], [0, 0]])
    kraus_2 = np.array([[1, 0], [0, 0]]).conj().T
    kraus_3 = np.array([[0, 1], [0, 0]])
    kraus_4 = np.array([[0, 1], [0, 0]]).conj().T
    kraus_5 = np.array([[0, 0], [1, 0]])
    kraus_6 = np.array([[0, 0], [1, 0]]).conj().T
    kraus_7 = np.array([[0, 0], [0, 1]])
    kraus_8 = np.array([[0, 0], [0, 1]]).conj().T

    kraus_ops = [
        [kraus_1, kraus_2],
        [kraus_3, kraus_4],
        [kraus_5, kraus_6],
        [kraus_7, kraus_8],
    ]
    np.testing.assert_equal(choi_rank(kraus_ops), 4)


def test_choi_rank_choi_matrix():
    """Verify Choi matrix of the swap operator map gives correct Choi rank."""
    choi_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_equal(choi_rank(choi_matrix), 4)


def test_choi_bad_input():
    """Verify that a bad input (such as a string which still passes
    with `numpy.linalg.matrix_rank`) raises an error"""
    with pytest.raises(ValueError, match="Not a valid"):
        bad_input = "string"
        choi_rank(bad_input)


if __name__ == "__main__":
    np.testing.run_module_suite()
