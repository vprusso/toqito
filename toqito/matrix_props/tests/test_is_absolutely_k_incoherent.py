"""Test is_absolutely_k_incoherent."""

import numpy as np
import pytest
import cvxpy as cp

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
        (np.diag([0.5, 0.25, 0.125, 0.125]), 4, True)
    ],
)
def test_is_absolutely_k_incoherent(mat, k, expected):
    """Test that is_absolutely_k_incoherent returns the correct boolean value on valid inputs."""
    np.testing.assert_equal(is_absolutely_k_incoherent(mat, k), expected)


def test_is_absolutely_k_incoherent_non_square():
    """Test that passing a non-square matrix raises a ValueError."""
    # Construct a non-square matrix.
    mat = np.array([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="Input matrix must be square."):
        is_absolutely_k_incoherent(mat, 1)


@pytest.mark.parametrize(
    "mat, k, expected, description",
    [
        # [1] Theorem 4 branch.
        # For n = 4 and k = 3, a pure state with rank = 1 satisfies rankX <= 4 - 3 (i.e. 1 <= 1),
        # so it should return False.
        (np.diag([1, 0, 0, 0]), 3, False, "Pure state triggers rankX <= n-k."),
        # [1] Theorem 4 branch.
        # For n = 4 and k = 3, we have n - k + 1 = 2. A density matrix with eigenvalues [0.5, 0.5, 0, 0]
        # has rank 2 and equal nonzero eigenvalues, so it returns True.
        (np.diag([0.5, 0.5, 0, 0]), 3, True, "Equal nonzero eigenvalues with rank equal to n-k+1."),
        # k == 2 branch.
        # For n = 3, a pure state (Frobenius norm squared = 1 > 1/(3-1)=0.5) should return False.
        (np.diag([1, 0, 0]), 2, False, "3x3 pure state for k==2 fails Frobenius norm condition."),
        # [1] Theorem 8 branch.
        # For n = 4 and k = 3 (n - 1), if lmax > 1 - 1/4 (i.e. lmax > 0.75) then it should return False.
        (np.diag([0.8, 0.1, 0.1, 0]), 3, False, "Eigenvalue too high: lmax > 1 - 1/n triggers False."),
        # [1] Theorem 8 branch.
        # For n = 4 and k = 3, if lmax is below the cutoff (1 - 1/4 = 0.75), then the SDP is executed.
        # Here we choose eigenvalues [0.7, 0.15, 0.15, 0]. Assuming the SDP is feasible, it should return True.
        (np.diag([0.7, 0.15, 0.15, 0]), 3, True, "SDP branch: feasible SDP for k equal to n - 1.")
    ],
)
def test_is_absolutely_k_incoherent_additional(mat, k, expected, description):
    """Additional tests to cover specific branches in is_absolutely_k_incoherent."""
    np.testing.assert_equal(is_absolutely_k_incoherent(mat, k), expected)


def test_sdp_solver_error(monkeypatch):
    """Test that a SolverError in the SDP branch causes is_absolutely_k_incoherent to return False."""
    # Create a 4x4 matrix with eigenvalues that trigger the SDP branch.
    # For n = 4 and k = 3, we need lmax <= 1 - 1/4 = 0.75.
    mat = np.diag([0.7, 0.15, 0.15, 0])
    # Define a fake solve method that raises cp.SolverError.
    def fake_solve(self, *args, **kwargs):
        raise cp.SolverError("Forced solver error for testing.")
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    np.testing.assert_equal(is_absolutely_k_incoherent(mat, 3), False)


def test_sdp_not_optimal(monkeypatch):
    """Test that if the SDP returns a status other than 'optimal' or 'optimal_inaccurate', the function returns False."""
    # Create a 4x4 matrix with eigenvalues that trigger the SDP branch.
    mat = np.diag([0.7, 0.15, 0.15, 0])
    # Define a fake solve method that sets the status to an unfavorable value.
    def fake_solve(self, *args, **kwargs):
        self.status = "infeasible"
        return
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    np.testing.assert_equal(is_absolutely_k_incoherent(mat, 3), False)
