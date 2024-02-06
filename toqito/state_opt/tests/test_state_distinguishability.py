"""Test state_distinguishability."""
import pytest

from toqito.matrices import standard_basis
from toqito.state_ops import pure_to_mixed
from toqito.state_opt import state_distinguishability
from toqito.states import bell

e_0, e_1 = standard_basis(2)


@pytest.mark.parametrize("vectors, probs, solver, primal_dual, expected_result", [
    # Bell states (default uniform probs with primal).
    ([bell(0), bell(1), bell(2), bell(3)], None, "cvxopt", "primal", 1),
    # Bell states (default uniform probs with dual).
    ([bell(0), bell(1), bell(2), bell(3)], None, "cvxopt", "dual", 1),
    # Bell states uniform probs with primal.
    ([bell(0), bell(1), bell(2), bell(3)], [1/4, 1/4, 1/4, 1/4], "cvxopt", "primal", 1),
    # Bell states uniform probs with dual.
    ([bell(0), bell(1), bell(2), bell(3)], [1/4, 1/4, 1/4, 1/4], "cvxopt", "dual", 1),
    # Density matrix Bell states (default uniform probs with primal).
    (
        [pure_to_mixed(bell(0)), pure_to_mixed(bell(1))],
         None,
         "cvxopt",
         "primal",
         1
    ),
    # Density matrix Bell states (default uniform probs with dual).
    (
        [pure_to_mixed(bell(0)), pure_to_mixed(bell(1))],
        None,
        "cvxopt",
        "dual",
        1
    ),
])
def test_state_distinguishability(vectors, probs, solver, primal_dual, expected_result):
    """Test function works as expected for a valid input."""
    val, _ = state_distinguishability(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual)
    assert abs(val - expected_result) <=1e-8


@pytest.mark.parametrize("vectors, probs, solver, primal_dual", [
    # Bell states (default uniform probs with dual).
    ([bell(0), bell(1), bell(2), e_0], None, "cvxopt", "dual"),
])
def test_state_distinguishability_invalid_vectors(vectors, probs, solver, primal_dual):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match = "Vectors for state distinguishability must all have the same dimension."):
        state_distinguishability(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual)
