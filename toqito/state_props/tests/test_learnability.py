"""Tests for the learnability helper functions and solver integration."""

from importlib import import_module

import numpy as np
import pytest
import scipy.sparse as sp

from toqito.state_props import learnability
from toqito.states import basis

learnability_module = import_module("toqito.state_props.learnability")


@pytest.fixture(autouse=True)
def mock_cvxpy_problem_solve(monkeypatch):
    """Replace CVXPY's solve method to avoid heavy solver calls in tests."""
    def fake_solve(self, *args, **kwargs):
        self._status = "optimal"
        self._value = 0.0
        return self._value

    monkeypatch.setattr("cvxpy.problems.problem.Problem.solve", fake_solve)


def test_learnability_accepts_vectors_and_matches_reduced():
    """Pure-state inputs should match reduced SDP verification."""
    e0, e1 = basis(2, 0), basis(2, 1)
    states = [e0, e1, e0 + e1]
    result = learnability(
        states,
        k=1,
        solver=None,
        solver_kwargs={"eps": 1e-6, "max_iters": 10_000},
        verify_tolerance=1e-3,
    )
    assert result["reduced_value"] is not None
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert result["reduced_status"] in {"optimal", "optimal_inaccurate"}
    assert np.isclose(result["value"], result["reduced_value"], atol=1e-3)
    assert isinstance(result["measurement_operators"], dict)
    assert isinstance(result["reduced_operators"], dict)
    assert result["total_value"] == pytest.approx(result["value"] * len(states), abs=1e-9)


def test_learnability_handles_mixed_states_and_skips_reduced():
    """Mixed states should skip reduced SDP validation."""
    rho_1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_2 = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=np.complex128)
    rho_3 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    result = learnability(
        [rho_1, rho_2, rho_3],
        k=1,
        solver=None,
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["reduced_value"] is None
    assert result["value"] >= 0
    assert isinstance(result["measurement_operators"], dict)
    assert result["reduced_operators"] is None
    assert result["total_value"] == pytest.approx(result["value"] * 3, abs=1e-9)


def test_learnability_returns_warning_on_large_gap(monkeypatch):
    """Trigger a warning when reduced and general optima differ substantially."""
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
            solver=None,
            solver_kwargs={"eps": 1e-6, "max_iters": 10_000},
            verify_tolerance=1e-9,
        )


def test_learnability_k_equals_number_of_states():
    """K equal to the number of states makes the objective zero."""
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(
        states,
        k=2,
        solver=None,
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["value"] == pytest.approx(0.0, abs=1e-9)


def test_learnability_verify_reduced_disabled():
    """Disabling verify_reduced suppresses the Gram matrix solve."""
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(
        states,
        k=1,
        solver=None,
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
    """Invalid state collections should raise ValueError."""
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
    """Detect invalid density matrices before solving."""
    with pytest.raises(ValueError):
        learnability(states, k=1)


def test_learnability_pure_density_matrices_extract_vector():
    """Pure density matrices should recover reference vectors."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    result = learnability(
        [rho, rho],
        k=1,
        solver=None,
        solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
    )
    assert result["reduced_value"] is not None


def test_learnability_empty_state_list_raises():
    """An empty state list is invalid input."""
    with pytest.raises(ValueError):
        learnability([], k=1)


def test_learnability_solver_none_branch():
    """Solver=None should use CVXPY defaults successfully."""
    states = [basis(2, 0), basis(2, 1)]
    result = learnability(states, k=1, solver=None)
    assert result["status"] == "optimal"


def test_learnability_reduced_invalid_k_raises():
    """Reduced SDP helper validates k bounds."""
    with pytest.raises(ValueError):
        learnability_module._solve_learnability_reduced(
            np.eye(2, dtype=np.complex128),
            k=0,
            solver=None,
            solver_kwargs=None,
        )


def test_sum_expressions_empty_returns_zero():
    """Summing an empty collection returns zero as a sentinel."""
    assert learnability_module._sum_expressions([]) == 0.0


def test_extract_state_vector_handles_column_vector():
    """Column-vector inputs are flattened without diagonalization."""
    column = np.array([[1.0], [0.0]], dtype=np.complex128)
    density = column @ column.conj().T
    vector = learnability_module._extract_state_vector(column, density)
    np.testing.assert_allclose(vector, np.array([1.0, 0.0]))


def test_extract_state_vector_uses_eigendecomposition_for_matrices():
    """Matrix inputs fall back to the dominant eigenvector representation."""
    density = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    vector = learnability_module._extract_state_vector(density, density)
    expected = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(vector, expected)


def test_convert_states_rejects_zero_trace_state():
    """States with zero trace are invalid for the SDP."""
    with pytest.raises(ValueError):
        learnability_module._convert_states(
            [np.zeros((2, 2), dtype=np.complex128)],
            tol=1e-8,
        )


def test_learnability_solve_problem_non_scs_branch():
    """_solve_problem forwards solver keywords for non-SCS solvers."""

    class DummyProblem:
        def __init__(self):
            self.status = "unknown"
            self.calls: list[dict] = []

        def solve(self, *args, **kwargs):
            self.calls.append(kwargs)
            self.status = "optimal"
            return 0.5

    problem = DummyProblem()
    value, status = learnability_module._solve_problem(
        problem,
        solver="ECOS",
        solver_kwargs={"max_iters": 25},
    )
    assert value == 0.5
    assert status == "optimal"
    assert problem.calls == [{"solver": "ECOS", "max_iters": 25}]


def test_learnability_solve_problem_default_solver():
    """_solve_problem uses CVXPY defaults when solver is None."""

    class DummyProblem:
        def __init__(self):
            self.status = "unknown"
            self.calls: list[dict] = []

        def solve(self, *_, **kwargs):
            self.calls.append(kwargs)
            self.status = "optimal"
            return 0.75

    problem = DummyProblem()
    value, status = learnability_module._solve_problem(
        problem,
        solver=None,
        solver_kwargs={"rho": 0.1},
    )
    assert value == 0.75
    assert status == "optimal"
    assert problem.calls == [{"rho": 0.1}]


def test_learnability_solve_problem_routes_to_scs(monkeypatch):
    """_solve_problem delegates to the specialized SCS helper when requested."""
    calls = {}

    def fake_scs(problem, kwargs):  # noqa: ARG001
        calls["invoked"] = kwargs
        return 0.0, "optimal"

    monkeypatch.setattr(
        learnability_module,
        "_solve_problem_with_scs",
        fake_scs,
    )

    value, status = learnability_module._solve_problem(
        object(),
        solver="SCS",
        solver_kwargs={"warm_start": True},
    )
    assert calls == {"invoked": {"warm_start": True}}
    assert value == 0.0
    assert status == "optimal"


def test_convert_states_degrades_to_mixed_vectors():
    """Pure input list with a mixed state clears the candidate vectors."""
    pure_state = np.array([1.0, 0.0], dtype=np.complex128)
    mixed_state = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    densities, vectors = learnability_module._convert_states(
        [pure_state, mixed_state],
        tol=1e-8,
    )
    assert len(densities) == 2
    assert vectors is None


def test_learnability_solve_problem_with_scs_helper():
    """Specialized SCS solver wrapper converts sparse inputs to CSC."""

    class DummyChain:
        def solve_via_data(self, problem, data, warm_start, verbose, solver_opts):
            assert sp.isspmatrix_csc(data[learnability_module.cp_settings.A])
            assert sp.isspmatrix_csc(data[learnability_module.cp_settings.P])
            assert "warm_start" not in solver_opts
            assert "verbose" not in solver_opts
            return {"value": 0.125, "status": "optimal"}

    class DummyProblem:
        def __init__(self):
            self.value = None
            self.status = None

        def get_problem_data(self, solver):
            assert solver is learnability_module.cp.SCS
            data = {
                learnability_module.cp_settings.A: sp.csr_matrix([[1.0]]),
                learnability_module.cp_settings.P: sp.csr_matrix([[1.0]]),
            }
            return data, DummyChain(), {}

        def unpack_results(self, solution, chain, inverse_data):
            self.value = solution["value"]
            self.status = solution["status"]

    problem = DummyProblem()
    value, status = learnability_module._solve_problem_with_scs(
        problem,
        {"warm_start": True, "verbose": True, "max_iters": 5_000},
    )
    assert value == 0.125
    assert status == "optimal"
    assert problem.value == 0.125
    assert problem.status == "optimal"


def test_learnability_solve_problem_with_scs_handles_missing_matrix():
    """SCS helper leaves absent matrices untouched while still returning results."""

    class DummyChain:
        def __init__(self):
            self.called = False
            self.opts = None

        def solve_via_data(self, problem, data, warm_start, verbose, solver_opts):
            self.called = True
            self.opts = solver_opts
            assert sp.isspmatrix_csc(data[learnability_module.cp_settings.A])
            assert data[learnability_module.cp_settings.P] is None
            assert warm_start is False
            assert verbose is False
            return {"status": "optimal", "value": 0.0}

    class DummyProblem:
        def __init__(self):
            self.status = None
            self.value = None
            self.chain = DummyChain()

        def get_problem_data(self, solver):
            assert solver is learnability_module.cp.SCS
            data = {
                learnability_module.cp_settings.A: sp.csr_matrix([[1.0]]),
                learnability_module.cp_settings.P: None,
            }
            return data, self.chain, {}

        def unpack_results(self, solution, chain, inverse_data):
            assert chain is self.chain
            self.status = solution["status"]
            self.value = solution["value"]

    problem = DummyProblem()
    value, status = learnability_module._solve_problem_with_scs(
        problem,
        {"warm_start": False, "verbose": False, "alpha": 0.9},
    )
    assert value == 0.0
    assert status == "optimal"
    assert problem.status == "optimal"
    assert problem.value == 0.0
    assert problem.chain.called is True
    assert problem.chain.opts == {"alpha": 0.9}


def test_is_scs_solver_supports_cvxtag_and_strings():
    """Solver detection recognizes both constants and strings for SCS."""
    assert learnability_module._is_scs_solver(None) is False
    assert learnability_module._is_scs_solver(learnability_module.cp.SCS) is True
    assert learnability_module._is_scs_solver("SCS") is True
    assert learnability_module._is_scs_solver("ECOS") is False


def test_is_scs_solver_strips_and_normalizes_strings():
    """String inputs with extra whitespace still identify the SCS solver."""
    assert learnability_module._is_scs_solver("  scs  ") is True
