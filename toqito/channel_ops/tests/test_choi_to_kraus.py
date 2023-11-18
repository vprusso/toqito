"""Tests for choi_to_kraus."""
import numpy as np

from toqito.channel_ops import choi_to_kraus
from toqito.perms import swap_operator


def test_choi_to_kraus():
    """Choi matrix of the swap operator."""

    choi_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    kraus_ops = [
        [
            np.array([[0.0, 0.70710678], [-0.70710678, 0.0]]),
            np.array([[0.0, -0.70710678], [0.70710678, 0.0]]),
        ],
        [
            np.array([[0.0, 0.70710678], [0.70710678, 0.0]]),
            np.array([[0.0, 0.70710678], [0.70710678, 0.0]]),
        ],
        [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])],
        [np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [0.0, 1.0]])],
    ]
    res_kraus_ops = choi_to_kraus(choi_mat)

    bool_mat = np.isclose(kraus_ops[0], res_kraus_ops[0])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[1], res_kraus_ops[1])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[2], res_kraus_ops[2])
    np.testing.assert_equal(np.all(bool_mat), True)

    bool_mat = np.isclose(kraus_ops[3], res_kraus_ops[3])
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_to_kraus_cpt():
    """Choi matrix of an isometry map."""
    choi_mat = np.array(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    kraus_ops = [np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])]

    res_kraus_ops = choi_to_kraus(choi_mat, dim=[3, 2])

    bool_mat = np.isclose(kraus_ops, res_kraus_ops)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_to_kraus_non_square():
    """Choi matrix of the swap operator for non square input/output."""

    choi_mat = swap_operator([2, 3])
    kraus_ops = [
        [
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ],
        [
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[-0.0, -0.0], [-1.0, -0.0], [-0.0, -0.0]]),
        ],
        [
            np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]),
            np.array([[-0.0, -0.0], [-0.0, -0.0], [-1.0, -0.0]]),
        ],
        [
            np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.array([[-0.0, -1.0], [-0.0, -0.0], [-0.0, -0.0]]),
        ],
        [
            np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
            np.array([[-0.0, -0.0], [-0.0, -1.0], [-0.0, -0.0]]),
        ],
        [
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        ],
    ]
    res_kraus_ops = choi_to_kraus(choi_mat, dim=[[3, 2], [2, 3]])
    np.testing.assert_equal(
        all(
            np.allclose(k_op[0], res_k_op[0]) and np.allclose(k_op[1], res_k_op[1])
            for k_op, res_k_op in zip(kraus_ops, res_kraus_ops)
        ),
        True,
    )


def test_choi_to_kraus_general_map():
    """Choi matrix of a map that erases everything and keeps the M(1, 2) entry of the input matrix."""
    choi_mat = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    kraus_ops = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

    res_kraus_ops = choi_to_kraus(choi_mat)

    bool_mat = np.isclose(kraus_ops, res_kraus_ops)
    np.testing.assert_equal(np.all(bool_mat), True)


def test_choi_to_kraus_reduced_rank():
    """Choi matrix of a hermicity preserving map with reduced rank."""
    choi_mat = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, -1],
        ]
    )
    c0 = np.sqrt(1 / (2 + 2 * np.sqrt(2)))
    c1 = np.sqrt(1 / (2 + 2 * np.sqrt(2)) + 1)
    kraus_ops = [
        [
            np.array([[0, 0], [c0, -c1]]),
            np.array([[0, 0], [-c0, c1]]),
        ],
        [
            np.array([[0, 0], [c1, c0]]),
            np.array([[0, 0], [c1, c0]]),
        ],
        [np.array([[-1, -1], [0, 0]]), np.array([[-1, -1], [0, 0]])],
    ]

    res_kraus_ops = choi_to_kraus(choi_mat)
    np.testing.assert_equal(
        all(
            np.allclose(k_op[0], res_k_op[0]) and np.allclose(k_op[1], res_k_op[1])
            for k_op, res_k_op in zip(kraus_ops, res_kraus_ops)
        ),
        True,
    )
