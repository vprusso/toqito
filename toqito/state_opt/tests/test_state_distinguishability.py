"""Test state_distinguishability."""

import numpy as np
import pytest

from toqito.matrices import standard_basis
from toqito.matrix_ops import to_density_matrix
from toqito.state_opt import state_distinguishability
from toqito.states import bb84, bell

e_0, e_1 = standard_basis(2)

states_min_error = [
    # Bell states (should be perfectly distinguishable)
    ([bell(0), bell(1), bell(2), bell(3)], 1),
    # Bell states as density matrices
    ([to_density_matrix(bell(i)) for i in range(4)], 1),
    # BB84 states
    (bb84()[0] + bb84()[1], 0.5),
]

states_unambiguous = [
    # Bell states (should be perfectly distinguishable)
    ([bell(0), bell(1)], 1),
    # |0> and |+> states (should be unambiguously distinguishable with probability 1 - 1 / sqrt(2))
    ([np.array([[1.0], [0.0]]), np.array([[1.0], [1.0]]) / np.sqrt(2)], 0.29289321881345254),
]

probs_min_error = [None, [1 / 4, 1 / 4, 1 / 4, 1 / 4]]

probs_unambiguous = [None, [1 / 2, 1 / 2]]

solvers = ["cvxopt"]
primal_duals = ["primal", "dual"]


@pytest.mark.parametrize("vectors, expected_result", states_min_error)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
@pytest.mark.parametrize("probs", probs_min_error)
def test_state_distinguishability_min_error(vectors, probs, solver, primal_dual, expected_result):
    """Test function works as expected for a valid input for the min_error strategy."""
    val, _ = state_distinguishability(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual)
    assert abs(val - expected_result) <= 1e-8


@pytest.mark.parametrize("vectors, expected_result", states_unambiguous)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
@pytest.mark.parametrize("probs", probs_unambiguous)
def test_state_distinguishability_unambiguous(vectors, probs, solver, primal_dual, expected_result):
    """Test function works as expected for a valid input for the unambiguous strategy."""
    val, _ = state_distinguishability(
        vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy="unambiguous"
    )
    assert abs(val - expected_result) <= 1e-8


@pytest.mark.parametrize(
    "vectors, probs, solver, primal_dual",
    [
        # Bell states (default uniform probs with dual).
        ([bell(0), bell(1), bell(2), e_0], None, "cvxopt", "dual"),
    ],
)
@pytest.mark.parametrize(
    "strategy",
    [
        "min_error",
        "unambiguous",
    ],
)
def test_state_distinguishability_invalid_vectors(vectors, probs, solver, primal_dual, strategy):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match="Vectors for state distinguishability must all have the same dimension."):
        state_distinguishability(
            vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy=strategy
        )
