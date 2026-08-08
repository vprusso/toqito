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

# Mixed states for unambiguous discrimination
states_unambiguous_mixed = [
    # Two orthogonal mixed states (Werner-like states)
    (
        [
            0.7 * np.array([[1.0, 0.0], [0.0, 0.0]]) + 0.3 * np.eye(2) / 2,
            0.7 * np.array([[0.0, 0.0], [0.0, 1.0]]) + 0.3 * np.eye(2) / 2,
        ],
        0.49,
    ),
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


@pytest.mark.parametrize("vectors, expected_result", states_unambiguous_mixed)
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("primal_dual", primal_duals)
@pytest.mark.parametrize("probs", probs_unambiguous)
def test_state_distinguishability_unambiguous_mixed(vectors, probs, solver, primal_dual, expected_result):
    """Test function works as expected for mixed states with the unambiguous strategy."""
    val, measurements = state_distinguishability(
        vectors=vectors, probs=probs, solver=solver, primal_dual=primal_dual, strategy="unambiguous"
    )
    assert abs(val - expected_result) <= 1e-2

    # For primal, also verify POVM properties
    if primal_dual == "primal":
        # Check that measurements sum to at most identity
        povm_sum = sum(measurements)
        assert np.allclose(povm_sum, np.eye(len(vectors[0])), atol=1e-6)

        # Check that all measurements are positive semidefinite
        for m in measurements:
            eigvals = np.linalg.eigvalsh(m)
            assert np.all(eigvals >= -1e-8)


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


def test_state_distinguishability_ppt_requires_subsystems_and_dimensions():
    """Using `measurement='ppt'` without subsystems/dimensions should raise a clear ValueError."""
    with pytest.raises(ValueError, match="subsystems.*dimensions.*required"):
        state_distinguishability(vectors=[bell(0), bell(1)], measurement="ppt")


def test_state_distinguishability_ppt_wrong_dimensions():
    """PPT dimensions must multiply to the state dimension."""
    with pytest.raises(ValueError, match="product of `dimensions`"):
        state_distinguishability(
            vectors=[bell(0), bell(1)],
            measurement="ppt",
            subsystems=[0],
            dimensions=[2, 3],
        )


def test_state_distinguishability_ppt_subsystems_out_of_range():
    """PPT subsystem indices must be valid for the provided dimensions."""
    with pytest.raises(ValueError, match="index into `dimensions`"):
        state_distinguishability(
            vectors=[bell(0), bell(1)],
            measurement="ppt",
            subsystems=[5],
            dimensions=[2, 2],
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"probs": [0.5]}, "must equal the number of states"),
        ({"probs": [-0.5, 1.5]}, "nonnegative"),
        ({"strategy": "bogus"}, "strategy must be either"),
        ({"measurement": "bogus"}, "measurement.*positive.*ppt"),
        ({"primal_dual": "bogus"}, "primal_dual option"),
    ],
)
def test_state_distinguishability_invalid_inputs(kwargs, match):
    """Invalid probs/strategy are rejected before the SDP is set up."""
    vectors = [bell(0), bell(1)]
    with pytest.raises(ValueError, match=match):
        state_distinguishability(vectors, **kwargs)


def test_unambiguous_dual_matches_primal_for_complex_states():
    """The unambiguous dual must equal the primal for complex states (Hermitian dual var, #1657)."""
    v0 = np.array([[1.0], [0.0]], dtype=complex)
    v1 = np.array([[1.0], [1.0j]], dtype=complex) / np.sqrt(2)
    states = [v0, v1]
    primal, _ = state_distinguishability(states, strategy="unambiguous", primal_dual="primal")
    dual, _ = state_distinguishability(states, strategy="unambiguous", primal_dual="dual")
    np.testing.assert_allclose(primal, dual, atol=1e-5)


# |00> and |01> differ only on the second qubit, so measuring it distinguishes them
# perfectly. The optimal POVM is a pair of product projectors, which is its own partial
# transpose, so the PPT constraint is not binding and the value is exactly 1.
_PPT_STATES = [np.kron(e_0, e_0), np.kron(e_0, e_1)]

# cvxopt does not meet its convergence tolerance on these PPT problems, and picos leaves
# `max_iterations` unset by default, which the cvxopt backend turns into `maxiters = 1e6`.
# Left to run that long the solver's scaling degrades until it fails inside LAPACK with
# `ArithmeticError: 7`, so the iteration count has to be capped for the solve to return.
# Whether the breakdown is reached at all depends on the cvxopt/LAPACK build, which is why
# this is required on Linux but not on macOS.
_PPT_MAX_ITERATIONS = 500


@pytest.mark.parametrize("primal_dual", primal_duals)
def test_state_distinguishability_ppt_forwards_solver_kwargs(primal_dual):
    """`kwargs` must reach picos on the PPT path, as they already do on the non-PPT paths.

    Regression test: `_ppt_primal`/`_ppt_dual` previously accepted no `**kwargs` and called
    `problem.solve(solver=solver)`, silently discarding the solver options the docstring
    promises to forward. picos rejects an unknown option, so the rejection itself is proof
    the option arrived; if the options are dropped again the solve simply succeeds and this
    test fails.
    """
    with pytest.raises(LookupError, match="Unknown option"):
        state_distinguishability(
            vectors=_PPT_STATES,
            probs=[1 / 2, 1 / 2],
            measurement="ppt",
            subsystems=[0],
            dimensions=[2, 2],
            primal_dual=primal_dual,
            definitely_not_a_real_solver_option=123,
        )


def test_state_distinguishability_ppt_unambiguous():
    """The unambiguous branch of the PPT primal returns the exact value and an inconclusive POVM element.

    `max_iterations` only reaches the solver because of the forwarding this change adds, so
    the solve breaking down again would mean the forwarding had regressed.
    """
    val, measurements = state_distinguishability(
        vectors=_PPT_STATES,
        probs=[1 / 2, 1 / 2],
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="primal",
        strategy="unambiguous",
        max_iterations=_PPT_MAX_ITERATIONS,
    )
    assert abs(val - 1) <= 1e-6
    # Unambiguous discrimination adds an inconclusive outcome, so there is one POVM
    # element per state plus one.
    assert len(measurements) == len(_PPT_STATES) + 1


def test_state_distinguishability_ppt_dual_rejects_unambiguous():
    """The PPT dual only implements the min-error problem and must say so."""
    with pytest.raises(ValueError, match="Only min_error strategy is supported for PPT dual."):
        state_distinguishability(
            vectors=_PPT_STATES,
            probs=[1 / 2, 1 / 2],
            measurement="ppt",
            subsystems=[0],
            dimensions=[2, 2],
            primal_dual="dual",
            strategy="unambiguous",
        )
