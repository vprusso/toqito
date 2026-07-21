"""Test is_absolutely_k_incoherent."""

import numpy as np
import pytest

from toqito.matrix_props import is_absolutely_k_incoherent


@pytest.mark.parametrize(
    "mat, k, expected",
    [
        # 2x2 maximally mixed state.
        (np.eye(2) / 2, 1, True),
        # 2x2 maximally mixed state with k equal to n (trivial).
        (np.eye(2) / 2, 2, True),
        # 2x2 pure state (non-maximally mixed density matrix).
        (np.array([[1, 0], [0, 0]]), 1, False),
        # 2x2 pure state with k equal to n (trivial).
        (np.array([[1, 0], [0, 0]]), 2, True),
        # 2x2 non-density matrix (not PSD).
        (np.array([[1, 2], [2, 1]]), 1, False),
        # 2x2 non-density matrix with k equal to n (trivial).
        (np.array([[1, 2], [2, 1]]), 2, True),
        # 3x3 example matrix.
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 1, False),
        # 3x3 example matrix with k equal to n (trivial).
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 3, True),
        # 3x3 example matrix with k greater than n (trivial).
        (np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]]), 4, True),
        # 3x3 maximally mixed state.
        (np.eye(3) / 3, 1, True),
        # 3x3 maximally mixed state with an eigenvalue condition.
        (np.eye(3) / 3, 2, True),
        # 3x3 maximally mixed state with k equal to n (trivial).
        (np.eye(3) / 3, 3, True),
        # 4x4 custom density matrix: diag(0.5, 0.25, 0.125, 0.125).
        (np.diag([0.5, 0.25, 0.125, 0.125]), 1, False),
        # 4x4 custom density matrix with k = 2.
        (np.diag([0.5, 0.25, 0.125, 0.125]), 2, False),
        # 4x4 custom density matrix with k = 3.
        (np.diag([0.5, 0.25, 0.125, 0.125]), 3, True),
        # 4x4 custom density matrix with k equal to n (trivial).
        (np.diag([0.5, 0.25, 0.125, 0.125]), 4, True),
        # [1] Theorem 4 branch (equal nonzero eigenvalues).
        # For n = 4 and k = 3, a matrix with eigenvalues [0.5, 0.5, 0, 0] has rank 2 and the nonzero eigenvalues are
        # equal.
        (np.diag([0.5, 0.5, 0, 0]), 3, True),
        # [1] Theorem 4 branch (non-equal nonzero eigenvalues).
        # For n = 4 and k = 3, a matrix with eigenvalues [0.6, 0.4, 0, 0] has rank 2 but the nonzero
        # eigenvalues differ; the spectral condition sqrt(0.6) > sqrt(0.4) shows it is not absolutely
        # 3-incoherent.
        (np.diag([0.6, 0.4, 0, 0]), 3, False),
        # k == 2 branch (Frobenius norm fails).
        # For n = 3, a pure state with Frobenius norm squared = 1 (which is greater than 1/(3-1)=0.5) returns False.
        (np.diag([1, 0, 0]), 2, False),
        # k == 2 branch (Frobenius norm passes).
        # For n = 4, the 4x4 maximally mixed state: Frobenius norm squared = 0.25 <= 1/(4-1)≈0.3333.
        (np.eye(4) / 4, 2, True),
        # k == 2 branch for n=4 pure state.
        # For n = 4, a pure state (e.g. diag(1,0,0,0)) has Frobenius norm squared = 1 which exceeds 1/(4-1); and n>3 so
        # the branch falls out.
        (np.diag([1, 0, 0, 0]), 2, False),
        # [1] Theorem 8 branch.
        # For n = 4 and k = 3 (n - 1), if lmax > 1 - 1/4 (i.e. lmax > 0.75), then it returns False.
        (np.diag([0.8, 0.1, 0.1, 0]), 3, False),
        # [1] Theorem 8 branch.
        # For n = 4 and k = 3: sqrt(0.7) > 2 sqrt(0.15), so the Theorem 8 condition fails.
        (np.diag([0.7, 0.15, 0.15, 0]), 3, False),
        # k <= 0 raise ValueError.
        (np.diag([0.8, 0.1, 0.1, 0]), 0, False),
        (np.diag([0.34, 0.22, 0.22, 0.22]), 2, True),
        (np.diag([0.9, 0.05, 0.05]), 2, False),
        (np.diag([0.7, 0.15, 0.15, 0]), 3, False),
        # sqrt(0.6) < 2 sqrt(0.2), so the Theorem 8 condition holds.
        (np.diag([0.6, 0.2, 0.2, 0]), 3, True),
        (np.diag([0.8, 0.1, 0.05, 0.05]), 3, False),
        # Regression: a first-order SDP solver falsely reported this spectrum feasible for the
        # Theorem 8 SDP; the closed-form condition correctly rejects it.
        (np.diag([0.75948417, 0.18316435, 0.03951963, 0.01305637, 0.00477548]), 4, False),
    ],
)
def test_is_absolutely_k_incoherent(mat, k, expected):
    """Test that is_absolutely_k_incoherent returns the correct boolean value on valid inputs."""
    if k <= 0:
        with pytest.raises(ValueError, match="k must be a positive integer."):
            is_absolutely_k_incoherent(mat, k)
    else:
        assert is_absolutely_k_incoherent(mat, k) == expected


def test_is_absolutely_k_incoherent_non_square():
    """Test that passing a non-square matrix raises a ValueError."""
    # Construct a non-square matrix.
    mat = np.array([[1, 0, 0], [0, 1, 0]])

    with pytest.raises(ValueError, match="Input matrix must be square."):
        is_absolutely_k_incoherent(mat, 1)


def test_non_equal_eigenvalues_branch():
    """Rank n-k+1 with unequal nonzero eigenvalues is not absolutely (n-1)-incoherent."""
    mat = np.diag([0.6, 0.4, 0, 0])
    assert is_absolutely_k_incoherent(mat, 3) is False


def test_k_equals_n_minus_one_true():
    """When k = n - 1 and the Theorem 8 spectral condition holds, the function returns True."""
    mat = np.diag([0.6, 0.2, 0.2, 0.0])
    assert is_absolutely_k_incoherent(mat, 3) is True


def test_k_equals_n_minus_one_large_eigenvalue():
    """When k = n - 1 and the largest eigenvalue exceeds the bound, return False immediately."""
    mat = np.diag([0.9, 0.05, 0.05, 0.0])
    assert is_absolutely_k_incoherent(mat, 3) is False


def test_is_absolutely_k_incoherent_k_outside_two_and_n_minus_one():
    """A peaked spectrum with k outside {2, n-1} falls through to the final `return False`."""
    rho = np.diag([0.9, 0.05, 0.03, 0.01, 0.01])
    assert is_absolutely_k_incoherent(rho, 3) is False
