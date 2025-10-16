"""Tests for learnability."""

import numpy as np
import pytest
from importlib import import_module

from toqito.state_opt import learnability
from toqito.states import basis

learnability_module = import_module("toqito.state_opt.learnability")


@pytest.fixture(autouse=True)
def mock_cvxpy_problem_solve(monkeypatch):
    def fake_solve(self, *args, **kwargs):
        self._status = "optimal"
        self._value = 1.0
        return self._value

    monkeypatch.setattr("cvxpy.problems.problem.Problem.solve", fake_solve)


def test_learnability_accepts_vectors_and_matches_reduced():
    e0, e1 = basis(2, 0), basis(2, 1)
    states = [e0, e1, e0 + e1]
    result = learnability(
        states,
        k=1,
        solver="SCS",
        solver_kwargs={"eps": 1e-6, "max_iters": 10_000},
        verify_tolerance=1e-3,
    )
    assert result["reduced_value"] is not None
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert result["reduced_status"] in {"optimal", "optimal_inaccurate"}
    assert np.isclose(result["value"], result["reduced_value"], atol=1e-3)
    assert isinstance(result["measurement_operators"], dict)
    assert isinstance(result["reduced_operators"], dict)


def test_learnability_handles_mixed_states_and_skips_reduced():
    rho_1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_2 = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=np.complex128)
    rho_3 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    result = learnability(
        [rho_1, rho_2, rho_3],
        k=1,
        solver="SCS",
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["reduced_value"] is None
    assert result["value"] >= 0
    assert isinstance(result["measurement_operators"], dict)
    assert result["reduced_operators"] is None


def test_learnability_returns_warning_on_large_gap(monkeypatch):
    states = [basis(2, 0), basis(2, 1)]

    def fake_reduced(*args, **kwargs):
        return 2.0, "optimal", {}

    monkeypatch.setattr(
        learnability_module,
        "_solve_learnability_reduced",
        fake_reduced,
    )

    with pytest.warns(RuntimeWarning):
        learnability(
            states,
            k=1,
            solver="SCS",
            solver_kwargs={"eps": 1e-6, "max_iters": 10_000},
            verify_tolerance=1e-9,
        )


def test_learnability_k_equals_number_of_states():
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(
        states,
        k=2,
        solver="SCS",
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["value"] == pytest.approx(1.0, abs=1e-9)


def test_learnability_verify_reduced_disabled():
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(
        states,
        k=1,
        solver="SCS",
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
        verify_reduced=False,
    )
    assert result["reduced_value"] is None
    assert result["reduced_status"] is None
    assert isinstance(result["measurement_operators"], dict)
    assert result["reduced_operators"] is None


@pytest.mark.parametrize(
    ("states", "k"),
    [
        ([basis(2, 0).flatten(), basis(2, 1).flatten()], 0),
        ([np.eye(2, dtype=np.complex128), np.eye(3, dtype=np.complex128)], 1),
    ],
)
def test_learnability_invalid_inputs_raise(states, k):
    with pytest.raises(ValueError):
        learnability(states, k=k)


@pytest.mark.parametrize(
    "states",
    [
        [
            np.array([[1, 0], [0, -0.25]], dtype=np.complex128),
            np.eye(2, dtype=np.complex128),
        ],
        [
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
            np.eye(2, dtype=np.complex128),
        ],
    ],
)
def test_learnability_state_validation_raises(states):
    with pytest.raises(ValueError):
        learnability(states, k=1)


def test_learnability_pure_density_matrices_extract_vector():
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    result = learnability(
        [rho, rho],
        k=1,
        solver="SCS",
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["reduced_value"] is not None


def test_learnability_empty_state_list_raises():
    with pytest.raises(ValueError):
        learnability([], k=1)


def test_learnability_solver_none_branch():
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(states, k=1, solver=None)
    assert result["status"] == "optimal"


def test_learnability_reduced_invalid_k_raises():
    with pytest.raises(ValueError):
        learnability_module._solve_learnability_reduced(
            np.eye(2, dtype=np.complex128),
            k=0,
            solver=None,
            solver_kwargs=None,
        )


def test_sum_expressions_empty_returns_zero():
    assert learnability_module._sum_expressions([]) == 0.0

