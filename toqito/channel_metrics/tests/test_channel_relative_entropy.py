"""Tests for channel_relative_entropy."""

import importlib

import cvxpy as cvx
import numpy as np
import pytest
from scipy.linalg import LinAlgError

from toqito.channel_metrics.channel_relative_entropy import channel_relative_entropy
from toqito.channels import depolarizing, pauli_channel
from toqito.perms import swap_operator
from toqito.state_props.integral_relative_entropy import _sandwich_parameters

_CHANNEL_RELATIVE_ENTROPY_MOD = importlib.import_module("toqito.channel_metrics.channel_relative_entropy")


def _dense(mat):
    """Convert a sparse or matrix-like channel representation to an ndarray."""
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def test_sandwich_parameters_for_distinct_channels():
    """Sandwich endpoints should satisfy 0 < mu < lambda."""
    channel_1 = depolarizing(2, 0.2)
    channel_2 = depolarizing(2, 0.4)

    mu, lam = _sandwich_parameters(channel_1, channel_2)

    assert mu > 0
    assert lam > mu


def test_sandwich_parameters_raises_on_eigh_failure(monkeypatch):
    """Failed generalized eigenvalue solve should raise ValueError."""

    def failing_eigh(*args, **kwargs):
        raise LinAlgError("singular pencil")

    monkeypatch.setattr(
        "toqito.state_props.integral_relative_entropy._generalized_eigenvalues",
        failing_eigh,
    )

    with pytest.raises(
        ValueError,
        match="Failed to compute sandwich parameters from generalized eigenvalues",
    ):
        _sandwich_parameters(depolarizing(2, 0.2), depolarizing(2, 0.4))


def test_forwards_solver_and_kwargs(monkeypatch):
    """Solver name and kwargs should be passed to every CVXPY solve."""
    solve_calls = []
    problem_cls = _CHANNEL_RELATIVE_ENTROPY_MOD.cvx.Problem
    original_init = problem_cls.__init__

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_solve = self.solve

        def recording_solve(solver=None, **solve_kwargs):
            solve_calls.append((solver, solve_kwargs))
            return original_solve(solver=solver, **solve_kwargs)

        self.solve = recording_solve

    monkeypatch.setattr(problem_cls, "__init__", wrapped_init)

    channel_relative_entropy(
        depolarizing(2, 1),
        depolarizing(2, 0.2),
        in_dim=2,
        epsilon_dec=0.2,
        solver="SCS",
        eps=1e-6,
        max_iters=50_000,
    )

    assert len(solve_calls) >= 2
    for solver, solve_kwargs in solve_calls:
        assert solver == "SCS"
        assert solve_kwargs["eps"] == 1e-6
        assert solve_kwargs["max_iters"] == 50_000
        assert solve_kwargs["verbose"] is False


def test_identical_channels_zero():
    """Identical channels should give zero in both bounds and mean modes."""
    choi = depolarizing(2, 1)

    lower, upper = channel_relative_entropy(choi, choi, in_dim=2, epsilon_dec=0.2, mean=False)
    avg = channel_relative_entropy(choi, choi, in_dim=2, epsilon_dec=0.2, mean=True)

    assert lower == 0
    assert upper == 0
    assert avg == 0


@pytest.mark.slow
def test_bounds_order_for_distinct_channels():
    """Distinct channels should produce ordered lower and upper bounds."""
    lower, upper = channel_relative_entropy(
        depolarizing(2, 1),
        depolarizing(2, 0.2),
        in_dim=2,
        epsilon_dec=0.2,
        mean=False,
    )

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert upper >= lower - 1e-8


def test_raises_mismatched_choi_shapes():
    """Mismatched Choi dimensions should raise."""
    with pytest.raises(ValueError, match="equal dimension"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(4, 0.2),
            in_dim=2,
            epsilon_dec=0.2,
        )


def test_raises_non_square_choi():
    """Non-square Choi matrices should raise."""
    bad = np.array([[1, 2, 3], [4, 5, 6]], dtype=complex)

    with pytest.raises(ValueError, match="must be square"):
        channel_relative_entropy(bad, bad, in_dim=1, epsilon_dec=0.2)


def test_raises_bad_in_dim():
    """Choi dimensions not divisible by in_dim should raise."""
    with pytest.raises(ValueError, match="divisible by in_dim"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(2, 0.4),
            in_dim=3,
            epsilon_dec=0.2,
        )


def test_channel_1_must_be_quantum_channel():
    """A non-quantum first argument should raise a clear ValueError."""
    bad_channel = np.array([[1.0, 2.0, 3.0, 4.0]] * 4, dtype=complex)
    good_channel = np.eye(4, dtype=complex) / 2

    with pytest.raises(ValueError, match="channel_1 is a quantum channel"):
        channel_relative_entropy(
            bad_channel,
            good_channel,
            in_dim=2,
            epsilon_dec=0.2,
        )


def test_channel_2_must_be_completely_positive():
    """A non-CP second argument should raise a clear ValueError."""
    good_channel = np.eye(4, dtype=complex) / 2
    non_cp_channel = swap_operator(2).astype(complex)

    with pytest.raises(ValueError, match="channel_2 is completely positive"):
        channel_relative_entropy(
            good_channel,
            non_cp_channel,
            in_dim=2,
            epsilon_dec=0.2,
        )


def test_raises_bad_hamiltonian_shape():
    """Hamiltonian must match the input dimension."""
    with pytest.raises(ValueError, match="Hamiltonian"):
        channel_relative_entropy(
            depolarizing(2, 0.2),
            depolarizing(2, 0.4),
            in_dim=2,
            epsilon_dec=0.2,
            hamiltonian=np.zeros((3, 3), dtype=complex),
        )


def test_raises_degenerate_integral_bounds(monkeypatch):
    """Non-positive mu or lambda <= mu should raise a clear ValueError."""
    channel_1 = depolarizing(2, 0.2)
    channel_2 = depolarizing(2, 0.4)

    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (0.0, 1.0),
    )
    with pytest.raises(ValueError, match="0 < mu < lambda"):
        channel_relative_entropy(channel_1, channel_2, in_dim=2, epsilon_dec=0.2)

    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (2.0, 1.0),
    )
    with pytest.raises(ValueError, match="0 < mu < lambda"):
        channel_relative_entropy(channel_1, channel_2, in_dim=2, epsilon_dec=0.2)


def test_mean_mode_returns_scalar():
    """Mean mode should return the midpoint of the bounds."""
    lower, upper = channel_relative_entropy(
        depolarizing(2, 1),
        depolarizing(2, 0.2),
        in_dim=2,
        epsilon_dec=0.2,
        mean=False,
    )
    avg = channel_relative_entropy(
        depolarizing(2, 1),
        depolarizing(2, 0.2),
        in_dim=2,
        epsilon_dec=0.2,
        mean=True,
    )

    assert avg == pytest.approx((lower + upper) / 2)


def test_raises_when_lower_sdp_fails(monkeypatch):
    """A failed lower-bound solve should raise RuntimeError."""
    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (0.1, 2.0),
    )

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            self.status = cvx.INFEASIBLE if FakeProblem.created == 1 else cvx.OPTIMAL

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(_CHANNEL_RELATIVE_ENTROPY_MOD.cvx, "Problem", FakeProblem)

    with pytest.raises(RuntimeError, match="Lower-bound SDP failed"):
        channel_relative_entropy(depolarizing(2, 0.2), depolarizing(2, 0.4), in_dim=2, epsilon_dec=0.2)


def test_raises_when_upper_sdp_fails(monkeypatch):
    """A failed upper-bound solve should raise RuntimeError."""
    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (0.1, 2.0),
    )

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            self.status = cvx.OPTIMAL if FakeProblem.created == 1 else cvx.INFEASIBLE

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(_CHANNEL_RELATIVE_ENTROPY_MOD.cvx, "Problem", FakeProblem)

    with pytest.raises(RuntimeError, match="Upper-bound SDP failed"):
        channel_relative_entropy(depolarizing(2, 0.2), depolarizing(2, 0.4), in_dim=2, epsilon_dec=0.2)


def test_warns_on_optimal_inaccurate_lower(monkeypatch):
    """OPTIMAL_INACCURATE on the lower SDP should warn."""
    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (0.1, 2.0),
    )

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            self.status = cvx.OPTIMAL_INACCURATE if FakeProblem.created == 1 else cvx.OPTIMAL

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(_CHANNEL_RELATIVE_ENTROPY_MOD.cvx, "Problem", FakeProblem)

    with pytest.warns(UserWarning, match="Lower-bound SDP returned OPTIMAL_INACCURATE"):
        channel_relative_entropy(depolarizing(2, 0.2), depolarizing(2, 0.4), in_dim=2, epsilon_dec=0.2)


def test_warns_on_optimal_inaccurate_upper(monkeypatch):
    """OPTIMAL_INACCURATE on the upper SDP should warn."""
    monkeypatch.setattr(
        _CHANNEL_RELATIVE_ENTROPY_MOD,
        "_sandwich_parameters",
        lambda rho, sigma: (0.1, 2.0),
    )

    class FakeProblem:
        created = 0

        def __init__(self, objective, constraints):
            self.value = 1.0
            FakeProblem.created += 1
            self.status = cvx.OPTIMAL if FakeProblem.created == 1 else cvx.OPTIMAL_INACCURATE

        def solve(self, **kwargs):
            pass

    FakeProblem.created = 0
    monkeypatch.setattr(_CHANNEL_RELATIVE_ENTROPY_MOD.cvx, "Problem", FakeProblem)

    with pytest.warns(UserWarning, match="Upper-bound SDP returned OPTIMAL_INACCURATE"):
        channel_relative_entropy(depolarizing(2, 0.2), depolarizing(2, 0.4), in_dim=2, epsilon_dec=0.2)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("param_p", "expected_mean"),
    [
        (0.010, 2.920),
        (0.026, 2.364),
        (0.050, 1.967),
        (0.072, 1.751),
        (0.091, 1.619),
        (0.100, 1.570),
    ],
)
def test_channel_relative_entropy_paper_example(param_p: float, expected_mean: float):
    """Mean estimate matches the paper's qubit dephasing-vs-depolarizing example."""
    # Paper example:
    # N_deph(rho) = 0.4 rho + 0.6 sigma_z rho sigma_z
    # M_dep(rho) = (1 - 3p/4) rho + p/4 (X rho X + Y rho Y + Z rho Z)
    channel_1 = _dense(pauli_channel(np.array([0.4, 0.0, 0.0, 0.6])))
    channel_2 = _dense(pauli_channel(np.array([1 - 3 * param_p / 4, param_p / 4, param_p / 4, param_p / 4])))

    lower, upper = channel_relative_entropy(channel_1, channel_2, in_dim=2, mean=False)
    avg = (lower + upper) / 2

    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert upper >= lower - 1e-8
    assert avg == pytest.approx(expected_mean, abs=2e-2)
