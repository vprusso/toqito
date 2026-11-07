"""Tests for the building blocks shared by nonlocal and extended nonlocal games."""

import cvxpy
import numpy as np
import pytest

from toqito.nonlocal_games._common import best_two_outcome_measurements


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_two_outcome_measurements_matches_sdp(seed):
    """The closed-form two-outcome step attains the value of Bob's semidefinite program with valid measurements."""
    rng = np.random.default_rng(seed)
    num_inputs, dim = 3, 4
    coeffs = rng.normal(size=(num_inputs, 2, dim, dim)) + 1j * rng.normal(size=(num_inputs, 2, dim, dim))
    coeffs = (coeffs + coeffs.conj().transpose(0, 1, 3, 2)) / 2

    value, measurements = best_two_outcome_measurements(coeffs)

    for y in range(num_inputs):
        for b in range(2):
            np.testing.assert_allclose(measurements[y, b], measurements[y, b].conj().T, atol=1e-12)
            assert np.linalg.eigvalsh(measurements[y, b]).min() > -1e-12
        np.testing.assert_allclose(measurements[y].sum(axis=0), np.eye(dim), atol=1e-12)
    assert value == pytest.approx(np.einsum("ybij,ybji->", coeffs, measurements).real)

    povms = {(y, b): cvxpy.Variable((dim, dim), hermitian=True) for y in range(num_inputs) for b in range(2)}
    objective = sum(cvxpy.real(cvxpy.trace(coeffs[y, b] @ povms[y, b])) for (y, b) in povms)
    constraints = [var >> 0 for var in povms.values()]
    constraints += [povms[y, 0] + povms[y, 1] == np.eye(dim) for y in range(num_inputs)]
    sdp_value = cvxpy.Problem(cvxpy.Maximize(objective), constraints).solve(solver=cvxpy.SCS, eps=1e-9)
    assert value == pytest.approx(sdp_value, abs=1e-6)
