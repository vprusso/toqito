"""Tests for factor_width."""

from importlib import import_module

import numpy as np
import pytest
import scipy.sparse as sp

from toqito.matrix_props import factor_width

factor_width_module = import_module("toqito.matrix_props.factor_width")

pytestmark = [
    pytest.mark.filterwarnings("ignore:Converting A to a CSC"),
    pytest.mark.filterwarnings("ignore:Converting P to a CSC"),
    pytest.mark.filterwarnings("ignore:Initializing a Constant with a nested list"),
    pytest.mark.filterwarnings("ignore:Solution may be inaccurate"),
]


@pytest.fixture(scope="module")
def solver_settings():
    """Provide tighter solver tolerances for deterministic tests."""
    return {"eps": 1e-7, "max_iters": 20000}


def test_factor_width_one_for_diagonal(solver_settings):
    """Diagonal matrix with zeros is 1-factorable."""
    mat = np.diag([1.0, 2.0, 0.0])
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is True
    reconstructed = sum(result["factors"])
    np.testing.assert_allclose(reconstructed, mat, atol=1e-5)


def test_factor_width_false_for_rank_one_sum(solver_settings):
    """Rank-one projector should not be 1-factorable."""
    mat = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
    result = factor_width(mat, k=1, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is False


def test_factor_width_accepts_k_equals_dimension():
    """Factor width equals dimension yields trivial decomposition."""
    mat = np.eye(3, dtype=np.complex128)
    result = factor_width(mat, k=3)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], mat)


def test_factor_width_zero_matrix_trivial_case():
    """Zero matrix should always be feasible."""
    mat = np.zeros((2, 2), dtype=np.complex128)
    result = factor_width(mat, k=1)
    assert result["feasible"] is True
    np.testing.assert_allclose(result["factors"][0], np.zeros_like(mat))


def test_factor_width_ball_example(solver_settings):
    """Verify factor width on the ensemble described in the manuscript."""
    d = 4
    k = 2
    x = 0.4
    eps = x / (4 * d)
    u = np.zeros((d, 1), dtype=np.complex128)
    u[:2, 0] = 1 / np.sqrt(2)
    uu = u @ u.conj().T
    mat = (1 - eps / d) * (x * np.eye(d) / d + (1 - x) * uu)
    result = factor_width(mat, k=k, solver="SCS", solver_kwargs=solver_settings)
    assert result["feasible"] is True
    reconstructed = sum(result["factors"])
    np.testing.assert_allclose(reconstructed, mat, atol=1e-5)


def test_factor_width_rank3_example_matches_paper(solver_settings):
    """Rank-3 example from the paper has factor width three but not two."""
    mat = np.array(
        [
            [2, 1, 1, -1],
            [1, 2, 0, 1],
            [1, 0, 2, -1],
            [-1, 1, -1, 2],
        ],
        dtype=np.complex128,
    )

    result_three = factor_width(
        mat,
        k=3,
        solver="SCS",
        solver_kwargs=solver_settings,
    )
    assert result_three["feasible"] is True
    assert result_three["factors"]
    np.testing.assert_allclose(
        sum(result_three["factors"]),
        mat,
        atol=1e-5,
    )

    result_two = factor_width(
        mat,
        k=2,
        solver="SCS",
        solver_kwargs=solver_settings,
    )
    assert result_two["feasible"] is False


def test_factor_width_rejects_non_square_matrix():
    """Non-square input matrices should be rejected."""
    mat = np.ones((2, 3), dtype=np.complex128)
    with pytest.raises(ValueError):
        factor_width(mat, k=1)


def test_factor_width_invalid_k_bounds():
    """The factor width parameter must obey the documented bounds."""
    mat = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError):
        factor_width(mat, k=0)
    with pytest.raises(ValueError):
        factor_width(mat, k=3)


def test_factor_width_requires_positive_semidefinite_input():
    """Indefinite matrices are rejected before the SDP is constructed."""
    mat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    with pytest.raises(ValueError):
        factor_width(mat, k=1)


def test_factor_width_no_support_subspaces(monkeypatch):
    """If no support subspace is found the routine reports infeasibility."""
    mat = np.eye(2, dtype=np.complex128)

    monkeypatch.setattr(
        factor_width_module,
        "_enumerate_support_subspaces",
        lambda *args, **kwargs: [],
    )

    result = factor_width(mat, k=1)
    assert result["feasible"] is False
    assert result["status"] == "no_support_subspace"
    assert result["factors"] is None
    assert result["subspaces"] == []


def test_factor_width_zero_dimensional_subspaces(monkeypatch):
    """Subspaces with zero dimension lead to the no-support-subspace status."""
    mat = np.eye(2, dtype=np.complex128)

    zero_basis = np.zeros((2, 0), dtype=np.complex128)
    monkeypatch.setattr(
        factor_width_module,
        "_enumerate_support_subspaces",
        lambda *args, **kwargs: [zero_basis],
    )

    result = factor_width(mat, k=1)
    assert result["feasible"] is False
    assert result["status"] == "no_support_subspace"
    assert result["factors"] is None


def test_factor_width_variable_without_value_triggers_infeasible(monkeypatch):
    """Variables without a value indicate solver failure despite optimal status."""
    mat = np.eye(2, dtype=np.complex128)

    monkeypatch.setattr(
        factor_width_module,
        "_enumerate_support_subspaces",
        lambda *args, **kwargs: [np.eye(2, dtype=np.complex128)],
    )
    monkeypatch.setattr(
        factor_width_module,
        "_solve_problem",
        lambda *args, **kwargs: factor_width_module.cp.OPTIMAL,
    )

    result = factor_width(mat, k=1)
    assert result["feasible"] is False
    assert result["status"] == factor_width_module.cp.OPTIMAL
    assert result["factors"] is None


def test_complex_real_block_round_trip():
    """Converting to the real block form and back recovers the matrix."""
    mat = np.array([[1 + 2j, 3 - 4j], [5j, -1 - 6j]], dtype=np.complex128)
    block = factor_width_module._complex_to_real_block(mat)
    recovered = factor_width_module._real_block_to_complex(block)
    np.testing.assert_allclose(recovered, mat)


def test_real_block_to_complex_requires_even_dimension():
    """Odd-sized real blocks are rejected."""
    block = np.ones((3, 2))
    with pytest.raises(ValueError):
        factor_width_module._real_block_to_complex(block)


def test_enumerate_support_subspaces_respects_max_zero_limit():
    """The enumeration skips subsets exceeding the maximum zero count."""
    basis = np.eye(2, dtype=np.complex128)
    subspaces = factor_width_module._enumerate_support_subspaces(basis, max_zero_count=0.5, tol=1e-8)
    assert subspaces == []


def test_intersect_with_zero_returns_empty_when_product_is_empty():
    """Intersection that collapses to the zero subspace returns an empty basis."""

    class ZeroIntersectionBasis(np.ndarray):
        """Custom basis whose intersection with any kernel is empty."""

        def __new__(cls, array):
            obj = np.asarray(array, dtype=np.complex128).view(cls)
            return obj

        def __array_finalize__(self, obj):
            # Nothing special to propagate from parent instances.
            return None

        def __matmul__(self, other):  # pragma: no cover - exercised via numpy's matmul
            return np.zeros((self.shape[0], 0), dtype=np.complex128)

    basis = ZeroIntersectionBasis(np.eye(2, dtype=np.complex128))
    result = factor_width_module._intersect_with_zero(basis, index=0, tol=1e-8)
    assert result.shape == (2, 0)
    assert np.all(result == 0)


def test_is_scs_solver_variants():
    """The solver detection helper accepts multiple SCS identifiers."""
    assert factor_width_module._is_scs_solver(None) is False
    assert factor_width_module._is_scs_solver(factor_width_module.cp.SCS) is True
    assert factor_width_module._is_scs_solver(" scs ") is True
    assert factor_width_module._is_scs_solver("ECOS") is False


def test_canonical_key_handles_zero_and_nonzero_bases():
    """Ensure canonical keys distinguish subspaces consistently."""
    zero_key, zero_basis = factor_width_module._canonical_key(
        np.zeros((3, 0), dtype=np.complex128),
        tol=1e-8,
    )
    assert zero_key == b"zero"
    assert zero_basis.shape == (3, 0)

    basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.complex128)
    key1, ortho1 = factor_width_module._canonical_key(basis, tol=1e-8)
    key2, ortho2 = factor_width_module._canonical_key(basis + 1e-10, tol=1e-8)
    assert key1 == key2
    np.testing.assert_allclose(
        ortho1 @ ortho1.conj().T,
        ortho2 @ ortho2.conj().T,
        atol=1e-9,
    )


def test_enumerate_support_subspaces_tracks_max_zero(monkeypatch):
    """Seen entries update their max_zero value when revisited."""
    basis = np.array([[1.0], [0.0], [0.0]], dtype=np.complex128)

    def fake_canonical_key(_, tol):  # noqa: ARG001
        return b"fixed", basis

    def fake_intersect_with_zero(_, idx, tol):  # noqa: ARG001
        return basis

    monkeypatch.setattr(factor_width_module, "_canonical_key", fake_canonical_key)
    monkeypatch.setattr(
        factor_width_module,
        "_intersect_with_zero",
        fake_intersect_with_zero,
    )
    monkeypatch.setattr(
        factor_width_module,
        "_max_support_size",
        lambda *_: 1,
    )

    subspaces = factor_width_module._enumerate_support_subspaces(
        basis,
        max_zero_count=2,
        tol=1e-8,
    )
    assert len(subspaces) == 1
    np.testing.assert_allclose(subspaces[0], basis)


def test_factor_width_solver_returns_components(monkeypatch):
    """Factor width returns assembled factors when the solver succeeds."""
    mat = np.eye(2, dtype=np.complex128)

    def fake_enumerate(range_basis, max_zero_count, tol):  # noqa: ARG001
        return [np.eye(2, dtype=np.complex128)]

    fake_variable = factor_width_module.cp.Variable((4, 4), PSD=True)

    class FakeProblem:
        def __init__(self):
            self.status = factor_width_module.cp.OPTIMAL

        def solve(self, *args, **kwargs):
            return None

    def fake_problem(*args, **kwargs):  # noqa: ARG001
        return FakeProblem()

    monkeypatch.setattr(
        factor_width_module,
        "_enumerate_support_subspaces",
        fake_enumerate,
    )
    monkeypatch.setattr(
        factor_width_module.cp,
        "Variable",
        lambda *args, **kwargs: fake_variable,
    )
    monkeypatch.setattr(
        factor_width_module.cp,
        "Problem",
        lambda *args, **kwargs: fake_problem(),
    )
    monkeypatch.setattr(
        factor_width_module,
        "_solve_problem",
        lambda problem, solver, solver_kwargs: factor_width_module.cp.OPTIMAL,
    )
    monkeypatch.setattr(
        fake_variable,
        "value",
        np.eye(4),
        raising=False,
    )

    result = factor_width_module.factor_width(mat, k=1)
    assert result["feasible"] is True
    assert result["status"] == factor_width_module.cp.OPTIMAL
    assert result["factors"]
    np.testing.assert_allclose(result["factors"][0], np.eye(2))


def test_intersect_with_zero_covers_degenerate_and_generic_cases():
    """Intersection helper handles both trivial and non-trivial kernels."""
    basis = np.eye(2, dtype=np.complex128)
    intersect = factor_width_module._intersect_with_zero(basis, 0, tol=1e-8)
    np.testing.assert_allclose(intersect[:, 0], np.array([0.0, 1.0]))

    one_dim_basis = np.array([[1.0]], dtype=np.complex128)
    intersect_zero = factor_width_module._intersect_with_zero(one_dim_basis, 0, tol=1e-8)
    assert intersect_zero.shape == (1, 0)


def test_max_support_size_counts_active_rows():
    """Support size helper counts rows with norm exceeding tolerance."""
    basis = np.array([[1.0, 0.0], [0.0, 1e-9], [0.0, 0.0]], dtype=np.complex128)
    support = factor_width_module._max_support_size(basis, tol=1e-8)
    assert support == 1


def test_solve_problem_with_scs_converts_sparse_inputs():
    """The specialized SCS solve helper converts matrices to CSC format."""

    class DummyChain:
        def solve_via_data(self, problem, data, warm_start, verbose, solver_opts):
            assert sp.isspmatrix_csc(data[factor_width_module.cp_settings.A])
            assert sp.isspmatrix_csc(data[factor_width_module.cp_settings.P])
            assert "warm_start" not in solver_opts
            assert "verbose" not in solver_opts
            return {"value": 0.25, "status": "optimal"}

    class DummyProblem:
        def __init__(self):
            self.value = None
            self.status = None

        def get_problem_data(self, solver):
            assert solver is factor_width_module.cp.SCS
            data = {
                factor_width_module.cp_settings.A: sp.csr_matrix([[1.0]]),
                factor_width_module.cp_settings.P: sp.csr_matrix([[1.0]]),
            }
            return data, DummyChain(), {}

        def unpack_results(self, solution, chain, inverse_data):
            self.value = solution["value"]
            self.status = solution["status"]

    problem = DummyProblem()
    status = factor_width_module._solve_problem_with_scs(
        problem,
        {"warm_start": True, "verbose": True, "max_iters": 10_000},
    )
    assert status == "optimal"
    assert problem.status == "optimal"
    assert problem.value == 0.25


def test_solve_problem_dispatches_to_default_solver():
    """_solve_problem forwards arguments correctly for non-SCS solvers."""

    class DummyProblem:
        def __init__(self):
            self.status = "unknown"
            self.calls: list[dict] = []

        def solve(self, *args, **kwargs):
            self.calls.append(kwargs)
            self.status = "optimal"
            return 1.0

    problem = DummyProblem()
    status = factor_width_module._solve_problem(
        problem,
        solver="ECOS",
        solver_kwargs={"max_iters": 50},
    )
    assert status == "optimal"
    assert problem.calls == [{"solver": "ECOS", "max_iters": 50}]


def test_solve_problem_defaults_to_cvxpy_when_solver_none():
    """_solve_problem should call the default CVXPY solver when solver is None."""

    class DummyProblem:
        def __init__(self):
            self.status = "unknown"
            self.calls: list[dict] = []

        def solve(self, *_, **kwargs):
            self.calls.append(kwargs)
            self.status = "optimal"
            return 0.0

    problem = DummyProblem()
    status = factor_width_module._solve_problem(
        problem,
        solver=None,
        solver_kwargs={"rho": 0.5},
    )
    assert status == "optimal"
    assert problem.calls == [{"rho": 0.5}]


def test_solve_problem_with_scs_skips_missing_matrix():
    """SCS helper leaves absent matrices untouched while still unpacking."""

    class DummyChain:
        def __init__(self):
            self.called = False
            self.received_opts = None

        def solve_via_data(self, problem, data, warm_start, verbose, solver_opts):
            self.called = True
            self.received_opts = solver_opts
            assert sp.isspmatrix_csc(data[factor_width_module.cp_settings.A])
            assert data[factor_width_module.cp_settings.P] is None
            assert warm_start is False
            assert verbose is False
            return {"status": "optimal", "value": 0.0}

    class DummyProblem:
        def __init__(self):
            self.status = None
            self.value = None
            self.chain = DummyChain()

        def get_problem_data(self, solver):
            assert solver is factor_width_module.cp.SCS
            data = {
                factor_width_module.cp_settings.A: sp.csr_matrix([[1.0]]),
                factor_width_module.cp_settings.P: None,
            }
            return data, self.chain, {}

        def unpack_results(self, solution, chain, inverse_data):
            assert chain is self.chain
            assert solution["status"] == "optimal"
            self.status = solution["status"]
            self.value = solution["value"]

    problem = DummyProblem()
    status = factor_width_module._solve_problem_with_scs(
        problem,
        {"warm_start": False, "verbose": False, "rho": 1.0},
    )
    assert status == "optimal"
    assert problem.status == "optimal"
    assert problem.value == 0.0
    assert problem.chain.called is True
    assert problem.chain.received_opts == {"rho": 1.0}


def test_intersect_with_zero_returns_input_when_basis_empty():
    """Intersection helper returns the same basis when it has no columns."""
    basis = np.zeros((4, 0), dtype=np.complex128)
    result = factor_width_module._intersect_with_zero(basis, 0, tol=1e-8)
    assert result.shape == (4, 0)


def test_max_support_size_zero_columns_returns_zero():
    """Support counter should return zero when the basis has no columns."""
    basis = np.zeros((3, 0), dtype=np.complex128)
    assert factor_width_module._max_support_size(basis, tol=1e-8) == 0
