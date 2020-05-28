"""Tests for channel_ops."""
import numpy as np

from toqito.channel_ops import apply_map
from toqito.channel_ops import kraus_to_choi
from toqito.channel_ops import partial_map

from toqito.channels import depolarizing
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


def test_partial_map_depolarizing_first_system():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on first system.
    """
    rho = np.array(
        [
            [
                0.3101,
                -0.0220 - 0.0219 * 1j,
                -0.0671 - 0.0030 * 1j,
                -0.0170 - 0.0694 * 1j,
            ],
            [
                -0.0220 + 0.0219 * 1j,
                0.1008,
                -0.0775 + 0.0492 * 1j,
                -0.0613 + 0.0529 * 1j,
            ],
            [
                -0.0671 + 0.0030 * 1j,
                -0.0775 - 0.0492 * 1j,
                0.1361,
                0.0602 + 0.0062 * 1j,
            ],
            [
                -0.0170 + 0.0694 * 1j,
                -0.0613 - 0.0529 * 1j,
                0.0602 - 0.0062 * 1j,
                0.4530,
            ],
        ]
    )
    res = partial_map(rho, depolarizing(2))

    expected_res = np.array(
        [
            [0.20545 + 0.0j, 0.0 + 0.0j, -0.0642 + 0.02495j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.20545 + 0.0j, 0.0 + 0.0j, -0.0642 + 0.02495j],
            [-0.0642 - 0.02495j, 0.0 + 0.0j, 0.29455 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.0642 - 0.02495j, 0.0 + 0.0j, 0.29455 + 0.0j],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_map_depolarizing_second_system():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on second system.
    """
    rho = np.array(
        [
            [
                0.3101,
                -0.0220 - 0.0219 * 1j,
                -0.0671 - 0.0030 * 1j,
                -0.0170 - 0.0694 * 1j,
            ],
            [
                -0.0220 + 0.0219 * 1j,
                0.1008,
                -0.0775 + 0.0492 * 1j,
                -0.0613 + 0.0529 * 1j,
            ],
            [
                -0.0671 + 0.0030 * 1j,
                -0.0775 - 0.0492 * 1j,
                0.1361,
                0.0602 + 0.0062 * 1j,
            ],
            [
                -0.0170 + 0.0694 * 1j,
                -0.0613 - 0.0529 * 1j,
                0.0602 - 0.0062 * 1j,
                0.4530,
            ],
        ]
    )
    res = partial_map(rho, depolarizing(2), 1)

    expected_res = np.array(
        [
            [0.2231 + 0.0j, 0.0191 - 0.00785j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0191 + 0.00785j, 0.2769 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.2231 + 0.0j, 0.0191 - 0.00785j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0191 + 0.00785j, 0.2769 + 0.0j],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_map_dim_list():
    """
    Perform the partial map using the depolarizing channel as the Choi
    matrix on first system when the dimension is specified as list.
    """
    rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    res = partial_map(rho, depolarizing(2), 2, [2, 2])

    expected_res = np.array(
        [
            [3.5, 0.0, 5.5, 0.0],
            [0.0, 3.5, 0.0, 5.5],
            [11.5, 0.0, 13.5, 0.0],
            [0.0, 11.5, 0.0, 13.5],
        ]
    )

    bool_mat = np.isclose(expected_res, res)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_partial_map_non_square_matrix():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 1, 1, 1], [5, 6, 7, 8], [3, 3, 3, 3]])
        partial_map(rho, depolarizing(3))


def test_partial_map_non_square_matrix_2():
    """Matrix must be square."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
        partial_map(rho, depolarizing(3), 2)


def test_partial_map_invalid_dim():
    """Invalid dimension for partial map."""
    with np.testing.assert_raises(ValueError):
        rho = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [12, 11, 10, 9]])
        partial_map(rho, depolarizing(3), 1, [2, 2])


if __name__ == "__main__":
    np.testing.run_module_suite()
