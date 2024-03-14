"""Tests for choi_to_kraus."""

import numpy as np
import pytest

from toqito.channel_ops import choi_to_kraus
from toqito.perms import swap_operator

choi_mat_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
kraus_ops_swap = [
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

choi_mat_iso = np.array(
    [
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
kraus_op_iso = [np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])]


@pytest.mark.parametrize(
    "test_input, expected, input_dim",
    [
        # choi of swap
        (choi_mat_swap, kraus_ops_swap, None),
        (choi_mat_iso, kraus_op_iso, [3, 2]),
    ],
)
def test_choi_to_kraus(test_input, expected, input_dim):
    """Test function works as expected for valid inputs."""
    if input_dim is None:
        calculated = choi_to_kraus(test_input)

        for i, cal_value in enumerate(calculated):
            assert np.isclose(expected[i], cal_value).all()

    calculated = choi_to_kraus(test_input, dim=input_dim)
    for i, cal_value in enumerate(calculated):
        assert np.isclose(expected[i], cal_value).all()


choi_mat_non_square = swap_operator([2, 3])
kraus_ops_non_square = [
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

choi_mat_reduced_rank = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
)
c0 = np.sqrt(1 / (2 + 2 * np.sqrt(2)))
c1 = np.sqrt(1 / (2 + 2 * np.sqrt(2)) + 1)
kraus_ops_reduced_rank = [
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


@pytest.mark.parametrize(
    "test_input, expected, input_dim",
    [
        # Choi matrix of the swap operator for non square input/output.
        (choi_mat_non_square, kraus_ops_non_square, [[3, 2], [2, 3]]),
        # Choi matrix of a hermicity preserving map with reduced rank
        (choi_mat_reduced_rank, kraus_ops_reduced_rank, None),
    ],
)
def test_choi_to_kraus_non_square_reduced_rank(test_input, expected, input_dim):
    """Choi to kraus output for non-square input/output and reduced rank scenarios."""
    calculated = choi_to_kraus(test_input, dim=input_dim)
    assert all(
        np.allclose(k_op[0], res_k_op[0]) and np.allclose(k_op[1], res_k_op[1])
        for k_op, res_k_op in zip(expected, calculated)
    )


def test_choi_to_kraus_general_map():
    """Choi matrix of a map that erases everything and keeps the M(1, 2) entry of the input matrix."""
    choi_mat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    kraus_ops = [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]

    res_kraus_ops = choi_to_kraus(choi_mat)

    assert np.isclose(kraus_ops, res_kraus_ops).all()
