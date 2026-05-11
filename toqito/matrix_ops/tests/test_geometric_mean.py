"""Tests for geometric_mean."""

import re

import numpy as np
import pytest

from toqito.matrix_ops import geometric_mean

A_diag = np.diag([2.0, 4.0])
B_diag = np.diag([8.0, 1.0])

A_sym = np.array([[2.0, 0.0], [0.0, 3.0]])
B_sym = np.array([[2.0, 1.0], [1.0, 2.0]])
t_sym = 0.4
# Reference value for G_{0.4}(A_sym, B_sym) (scipy.linalg same formula as `geometric_mean`).
expected_sym = np.array(
    [
        [1.9461120605297577, 0.4694128833387108],
        [0.4694128833387108, 2.4497552074559246],
    ]
)

I_2 = np.eye(2)
B_scaled = np.diag([4.0, 9.0])


@pytest.mark.parametrize(
    "input_a, input_b, t_weight, expected",
    [
        # endpoints: t=0 returns A, t=1 returns B
        (A_diag, B_diag, 0, A_diag),
        (A_diag, B_diag, 1, B_diag),
        # commuting diagonal: G_t(A, B) = A^{1-t} B^t elementwise
        (
            A_diag,
            B_diag,
            0.25,
            np.diag([2.0**0.75 * 8.0**0.25, 4.0**0.75 * 1.0**0.25]),
        ),
        # A = I gives G_t(I, B) = B^t
        (I_2, B_scaled, 0.5, np.diag([2.0, 3.0])),
        (A_sym, B_sym, t_sym, expected_sym),
        (B_sym, A_sym, 1 - t_sym, expected_sym),
        (
            A_diag,
            B_diag,
            -1.0,
            np.diag([2.0**2.0 / 8.0, 4.0**2.0 / 1.0]),
        ),
        (
            A_diag,
            B_diag,
            2.0,
            np.diag([(1.0 / 2.0) * 8.0**2.0, (1.0 / 4.0) * 1.0**2.0]),
        ),
        (
            A_diag,
            B_diag,
            -0.5,
            np.diag([2.0**1.5 * 8.0**-0.5, 4.0**1.5 * 1.0**-0.5]),
        ),
        (
            A_diag,
            B_diag,
            1.5,
            np.diag([2.0**-0.5 * 8.0**1.5, 4.0**-0.5 * 1.0**1.5]),
        ),
        # A = I: G_t(I, B) = B^t for diagonal B
        (I_2, B_scaled, -0.5, np.diag([4.0**-0.5, 9.0**-0.5])),
        (I_2, B_scaled, 1.5, np.diag([4.0**1.5, 9.0**1.5])),
    ],
)
def test_geometric_mean(input_a, input_b, t_weight, expected):
    """Test function works as expected for valid inputs."""
    calculated = geometric_mean(input_a, input_b, t_weight)
    np.testing.assert_allclose(calculated, expected, rtol=1e-5, atol=1e-8)


def test_geometric_mean_symmetry_extended_t():
    """G_t(A,B) = G_{1-t}(B,A) for t outside [0, 1] (same identity as on [0, 1])."""
    t = 1.2
    left = geometric_mean(A_sym, B_sym, t)
    right = geometric_mean(B_sym, A_sym, 1.0 - t)
    np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-8)


A_2 = np.array([[2.0, 0.1], [0.1, 1.0]])
B_3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

A_rect = np.ones((2, 3))
B_rect = np.ones((2, 3))

A_not2d = np.ones((2, 2, 1))
B_not2d = np.ones((2, 2, 1))

rho_psd = np.array([[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]])
sigma_pd = np.eye(4)

non_hermitian = np.array([[1.0, 2.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "input_a, input_b, t_weight, expected_msg",
    [
        # mismatched dimensions
        (A_2, B_3, 0.5, "The matrices must be the same size."),
        # non-square (same outer shape)
        (A_rect, B_rect, 0.5, "The matrices must be square."),
        # same shape but not 2D
        (A_not2d, B_not2d, 0.5, "The matrices must be 2D."),
        # weight out of range
        (A_diag, B_diag, -1.01, "The weight must be in the range [-1, 2]."),
        (A_diag, B_diag, 2.01, "The weight must be in the range [-1, 2]."),
        # positive semidefinite but not positive definite
        (rho_psd, sigma_pd, 0.5, "The matrices must be positive definite."),
        # non-Hermitian: fails Hermitian check in ``is_positive_definite``).
        (non_hermitian, I_2, 0.5, "The matrices must be positive definite."),
    ],
)
def test_geometric_mean_invalid_input(input_a, input_b, t_weight, expected_msg):
    """Test function raises an error for invalid inputs."""
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        geometric_mean(input_a, input_b, t_weight)
