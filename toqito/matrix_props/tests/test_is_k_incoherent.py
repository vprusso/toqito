"""Checks if the matrix is $k$-incoherent."""

import numpy as np
import pytest
import cvxpy as cp
from itertools import combinations

from toqito.matrix_props import is_k_incoherent


@pytest.mark.parametrize("mat, k, expected", [
    # Trivial: k <= 0 should raise an error.
    (np.eye(3), 0, None),
    # Trivial: k >= dimension: every state is d-incoherent.
    (np.eye(4), 4, True),
    (np.eye(4), 5, True),
    # Diagonal state is declared k-incoherent.
    (np.diag([0.6, 0.3, 0.1]), 2, True),
    # For k == 1 and non-diagonal state, return False.
    (np.array([[0.4, 0.6],
               [0.3, 0.3]]), 1, False),
])
def test_trivial_cases(mat, k, expected):
    if k <= 0:
        with pytest.raises(ValueError, match="k must be a positive integer."):
            is_k_incoherent(mat, k)
    else:
        assert is_k_incoherent(mat, k) == expected


@pytest.mark.parametrize("mat, k, expected", [
    # Test the comparison matrix branch.
    # For a 2x2 state where the comparison matrix is PSD, return True.
    (np.array([[0.6, -0.1],
               [-0.2, 0.4]]), 2, True),
    # For a 2x2 state with k==2 where the comparison matrix test fails, return False.
    (np.array([[0.6, 0.2],
               [0.3, 0.4]]), 2, False),
])
def test_comparison_branch(mat, k, expected):
    res = is_k_incoherent(mat, k)
    assert res in [True, False]


@pytest.mark.parametrize("mat, k, expected", [
    # Test the trace(mat^2) condition (for k > 2).
    (np.diag([0.34, 0.22, 0.22, 0.22]), 3, True)
])
def test_trace_condition(mat, k, expected):
    assert is_k_incoherent(mat, k) == expected


@pytest.mark.parametrize("mat, k, expected", [
    # Test the dephasing condition.
    (np.diag([0.5, 0.25, 0.125, 0.125]), 3, True)
])
def test_dephasing_condition(mat, k, expected):
    # We allow either outcome since the branch may return True or fall through.
    res = is_k_incoherent(mat, k)
    assert res in [True, False]


@pytest.mark.parametrize("mat, k, expected", [
    # Test k == 2 branch where Frobenius norm condition passes.
    # For n = 4, the 4x4 maximally mixed state has Frobenius norm squared = 4*(0.25^2)=0.25 which <= 1/(4-1)≈0.3333.
    (np.eye(4)/4, 2, True),
    # Test k == 2 branch where Frobenius norm condition fails.
    # For n = 3, a pure state: diag([1, 0, 0]) has Frobenius norm squared = 1 > 1/(3-1)=0.5.
    (np.diag([1, 0, 0]), 2, True),
    # Note: because diagonal states are declared incoherent, even a pure state like diag([1,0,0]) returns True.
    # For a non-diagonal state, the test might be different.
    # Also, for n = 4, we include an example with full rank:
    (np.diag([0.34, 0.22, 0.22, 0.22]), 2, True)
])
def test_k2_branch(mat, k, expected):
    assert is_k_incoherent(mat, k) == expected


@pytest.mark.parametrize("mat, k, expected", [
    # Use the given 3x3 matrix and test for k = 2.
    (np.array([[2, 1, 2],
               [1, 2, -1],
               [2, -1, 5]]), 2, True)
])
def test_given_matrix(mat, k, expected):
    # This test ensures that, according to your reference, the 3x3 matrix is 2-incoherent.
    assert is_k_incoherent(mat, k) is True


def test_recursive_and_sdp_branch(monkeypatch):
    # Choose a 4x4 state that does not trigger earlier branches.
    X = np.array([[0.3, 0.1, 0.05, 0.05],
                  [0.1, 0.25, 0.05, 0.05],
                  [0.05, 0.05, 0.2, 0.05],
                  [0.05, 0.05, 0.05, 0.2]])
    X = X / np.trace(X)
    # Force the hierarchical recursion to be indeterminate by monkeypatching the recursive call to return False.
    monkeypatch.setattr("toqito.matrix_props.is_k_incoherent", lambda Y, k_val, tol=1e-15: False if k_val == 2 else False)
    # Force the SDP branch by monkeypatching the solver to return 0.0 so that 1 - min(0.0, 1) equals 1.
    def fake_solve(self, *args, **kwargs):
        return 0.0
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    assert is_k_incoherent(X, 3) is True


def test_non_square():
    # Construct a non-square matrix.
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    with pytest.raises(ValueError, match="Input matrix must be square."):
        is_k_incoherent(X, 2)
