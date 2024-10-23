"""Test state_exclusion."""

import numpy as np
import pytest

from toqito.matrices import standard_basis
from toqito.matrix_ops import to_density_matrix
from toqito.state_opt import state_exclusion
from toqito.states import bell

e_0, e_1 = standard_basis(2)

states_min_error = [
    # Bell states are perfectly distinguishable, so the probability of error should be nil
    ([bell(0), bell(1), bell(2), bell(3)], None, 0, {}),
    ([bell(0), bell(1), bell(2), bell(3)], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], None, 0, {"cvxopt_kktsolver": "ldl"}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], [1 / 2, 1 / 2], 0, {"cvxopt_kktsolver": "ldl"}),
    # For |0> and |+>, this probability is 1/2 - 1/(2*sqrt(2))
    (
        [np.array([[1.], [0.]]), np.array([[1.], [1.]]) / np.sqrt(2)],
        None,
        0.14644660940672627,
        {"cvxopt_kktsolver": "ldl"}
    ),
    (
        [np.array([[1.], [0.]]), np.array([[1.], [1.]]) / np.sqrt(2)],
        [1 / 2, 1 / 2],
        0.14644660940672627,
        {"cvxopt_kktsolver": "ldl"}
    ),
]

states_unambiguous = [
    # Bell states are perfectly distinguishable, so the probability of error should be nil
    ([bell(0), bell(1), bell(2), bell(3)], None, 0, {}),
    ([bell(0), bell(1), bell(2), bell(3)], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], None, 0, {}),
    ([to_density_matrix(bell(0)), to_density_matrix(bell(1))], [1 / 2, 1 / 2], 0, {}),
    # For |0> and |+>, this probability is 1/sqrt(2)
    (
        [np.array([[1.], [0.]]), np.array([[1.], [1.]]) / np.sqrt(2)],
        None,
        0.707106781186547,
        {
            "abs_ipm_opt_tol": 1e-5,
        }
    ),
    (
        [np.array([[1.], [0.]]), np.array([[1.], [1.]]) / np.sqrt(2)],
        [1 / 2, 1 / 2],
        0.707106781186547,
        {
            "abs_ipm_opt_tol": 1e-5,
        }
    ),
]

solvers = [
    "cvxopt"
]

primal_duals = [
    "primal",
    "dual"
]


@pytest.mark.parametrize("vectors, probs, expected_result, kwargs", states_min_error)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
def test_state_exclusion_min_error(vectors, probs, solver, primal_dual, expected_result, kwargs):
    """Test function works as expected for a valid input."""
    val, _ = state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, **kwargs)
    assert abs(val - expected_result) <= 1e-8


@pytest.mark.parametrize("vectors, probs, expected_result, kwargs", states_unambiguous)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
def test_state_exclusion_unambiguous(vectors, probs, solver, primal_dual, expected_result, kwargs):
    """Test function works as expected for a valid input."""
    val, _ = state_exclusion(
        vectors=vectors,
        probs=probs,
        solver=solver,
        primal_dual=primal_dual,
        strategy="unambiguous",
        **kwargs
    )
    # Accuracy is quite low bcause of primals=None
    assert abs(val - expected_result) <= 1e-3


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
    ]
)
def test_state_exclusion_invalid_vectors(vectors, probs, solver, primal_dual, strategy):
    """Test function works as expected for an invalid input."""
    with pytest.raises(ValueError, match="Vectors for state distinguishability must all have the same dimension."):
        state_exclusion(vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy=strategy)
