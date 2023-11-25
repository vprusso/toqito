"""Test state_exclusion."""
import numpy as np
import pytest

from toqito.states import bell
from toqito.matrices import standard_basis
from toqito.state_opt import state_exclusion


e_0, e_1 = standard_basis(2)


@pytest.mark.parametrize("vectors, probs, solver, primal_dual, expected_result", [
    # Bell states (default uniform probs with primal).
    ([bell(0), bell(1), bell(2), bell(3)], None, "cvxopt", "primal", 0),
    # Bell states (default uniform probs with dual).
    ([bell(0), bell(1), bell(2), bell(3)], None, "cvxopt", "dual", 0),
    # Bell states uniform probs with primal.
    ([bell(0), bell(1), bell(2), bell(3)], [1/4, 1/4, 1/4, 1/4], "cvxopt", "primal", 0),
    # Bell states uniform probs with dual.
    ([bell(0), bell(1), bell(2), bell(3)], [1/4, 1/4, 1/4, 1/4], "cvxopt", "dual", 0),
])
def test_conclusive_state_exclusion(vectors, probs, solver, primal_dual, expected_result):
    val, _ = state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual)
    assert abs(val- expected_result) <=1e-8


@pytest.mark.parametrize("vectors, probs, solver, primal_dual", [
    # Bell states (default uniform probs with dual).
    ([bell(0), bell(1), bell(2), e_0], None, "cvxopt", "dual"),
])
def test_state_exclusion_invalid_vectors(vectors, probs, solver, primal_dual):
    with pytest.raises(ValueError):
        state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual)
