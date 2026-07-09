"""Tests for internal cone utilities."""

import cvxpy
import numpy as np

from toqito.cones._utils import _contains_effective_variables


def test_contains_effective_variables_numpy():
    """Numpy arrays are not CVXPY expressions."""
    assert not _contains_effective_variables(np.eye(2))


def test_contains_effective_variables_constant():
    """Constant CVXPY expressions are not effectively variable-dependent."""
    expr = cvxpy.Constant(np.eye(2))
    assert not _contains_effective_variables(expr)


def test_contains_effective_variables_variable():
    """A standalone variable should be detected as effective."""
    x = cvxpy.Variable((2, 2))
    x.value = np.eye(2)

    assert _contains_effective_variables(x)


def test_contains_effective_variables_cancelled():
    """Variables that cancel algebraically should not be treated as effective."""
    x = cvxpy.Variable((2, 2))
    x.value = np.eye(2)

    expr = cvxpy.Constant(np.eye(2)) + x - x

    assert not _contains_effective_variables(expr)
