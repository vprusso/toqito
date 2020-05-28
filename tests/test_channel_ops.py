"""Tests for channel_ops."""
import numpy as np

from toqito.channel_ops import apply_map
from toqito.channel_ops import kraus_to_choi

from toqito.perms import swap_operator


def test_apply_map_choi():
    """
    The swap operator is the Choi matrix of the transpose map.

    The following test is a (non-ideal, but illustrative) way of computing
    the transpose of a matrix.
    """
    test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    expected_res = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    res = apply_map(test_input_mat, swap_operator(3))

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_apply_map_kraus():
    """
    Apply Kraus map.

    The following test computes PHI(X) where X = [[1, 2], [3, 4]] and
    where PHI is the superoperator defined by:
    Phi(X) = [[1,5],[1,0],[0,2]] X [[0,1][2,3][4,5]].conj().T -
    [[1,0],[0,0],[0,1]] X [[0,0][1,1],[0,0]].conj().T
    """
    test_input_mat = np.array([[1, 2], [3, 4]])

    kraus_1 = np.array([[1, 5], [1, 0], [0, 2]])
    kraus_2 = np.array([[0, 1], [2, 3], [4, 5]])
    kraus_3 = np.array([[-1, 0], [0, 0], [0, -1]])
    kraus_4 = np.array([[0, 0], [1, 1], [0, 0]])

    expected_res = np.array([[22, 95, 174], [2, 8, 14], [8, 29, 64]])

    res = apply_map(test_input_mat, [[kraus_1, kraus_2], [kraus_3, kraus_4]])

    bool_mat = np.isclose(res, expected_res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_kraus_to_choi_max_ent_2():
    """Choi matrix of the transpose map is the swap operator."""
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

    choi_res = kraus_to_choi(kraus_ops)
    expected_choi_res = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )

    bool_mat = np.isclose(choi_res, expected_choi_res)
    np.testing.assert_equal(np.all(bool_mat), True)


if __name__ == "__main__":
    np.testing.run_module_suite()
