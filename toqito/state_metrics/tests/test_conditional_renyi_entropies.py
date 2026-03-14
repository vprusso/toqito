"""Tests for conditional Rényi entropies."""

import numpy as np
import pytest
from toqito.state_metrics.conditional_renyi_entropies import (
    petz_conditional_entropy_downarrow,
    petz_conditional_entropy_uparrow,
    sandwiched_conditional_entropy_downarrow,
)
from toqito.states import bell, max_mixed
from toqito.state_props import von_neumann_entropy

# Test matrices
RHO_BELL = bell(0) @ bell(0).conj().T
RHO_MAX_MIXED = max_mixed(4, is_sparse=False)
RHO_PURE = np.zeros((4, 4))
RHO_PURE[0, 0] = 1.0
RHO_MIXED = np.diag([0.5, 0.5, 0, 0])
RHO_INVALID = np.array([[0.5, 0, 0, 0], [0, 0.4, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0]])

@pytest.mark.parametrize(
    "rho_AB, alpha, expected",
    [
        # Bell state, alpha=1, conditional entropy should be 0
        (RHO_BELL, 1, 0.0),
        # Pure state, alpha=1, conditional entropy should be 0
        (RHO_PURE, 1, 0.0),
        # Mixed state, alpha=1
        (RHO_MIXED, 1, von_neumann_entropy(RHO_MIXED)),
        # Maximally mixed, alpha=1
        (RHO_MAX_MIXED, 1, von_neumann_entropy(RHO_MAX_MIXED)),
        # Bell state, alpha=2 (collision entropy)
        (RHO_BELL, 2, pytest.approx(0.0, abs=1e-6)),
        # Mixed state, alpha=2
        (RHO_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        # Maximally mixed, alpha=2
        (RHO_MAX_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        # Pure state, alpha=0
        (RHO_PURE, 0, pytest.approx(0.0, abs=1e-6)),
        # Bell state, alpha=0
        (RHO_BELL, 0, pytest.approx(0.0, abs=1e-6)),
        # Maximally mixed, alpha=0
        (RHO_MAX_MIXED, 0, pytest.approx(0.0, abs=1e-6)),
        # Bell state, alpha=np.inf
        (RHO_BELL, np.inf, pytest.approx(0.0, abs=1e-6)),
    ]
)
def test_petz_conditional_entropy_downarrow(rho_AB, alpha, expected):
    np.testing.assert_allclose(petz_conditional_entropy_downarrow(rho_AB, alpha), expected, atol=1e-5)


@pytest.mark.parametrize(
    "rho_AB, alpha, expected",
    [
        (RHO_BELL, 1, 0.0),
        (RHO_PURE, 1, 0.0),
        (RHO_MIXED, 1, von_neumann_entropy(RHO_MIXED)),
        (RHO_MAX_MIXED, 1, von_neumann_entropy(RHO_MAX_MIXED)),
        (RHO_BELL, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_MAX_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_PURE, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_BELL, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_MAX_MIXED, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_BELL, np.inf, pytest.approx(0.0, abs=1e-6)),
    ]
)
def test_petz_conditional_entropy_uparrow(rho_AB, alpha, expected):
    np.testing.assert_allclose(petz_conditional_entropy_uparrow(rho_AB, alpha), expected, atol=1e-5)


@pytest.mark.parametrize(
    "rho_AB, alpha, expected",
    [
        (RHO_BELL, 1, 0.0),
        (RHO_PURE, 1, 0.0),
        (RHO_MIXED, 1, von_neumann_entropy(RHO_MIXED)),
        (RHO_MAX_MIXED, 1, von_neumann_entropy(RHO_MAX_MIXED)),
        (RHO_BELL, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_MAX_MIXED, 2, pytest.approx(0.0, abs=1e-6)),
        (RHO_PURE, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_BELL, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_MAX_MIXED, 0, pytest.approx(0.0, abs=1e-6)),
        (RHO_BELL, np.inf, pytest.approx(0.0, abs=1e-6)),
    ]
)
def test_sandwiched_conditional_entropy_downarrow(rho_AB, alpha, expected):
    np.testing.assert_allclose(sandwiched_conditional_entropy_downarrow(rho_AB, alpha), expected, atol=1e-5)


@pytest.mark.parametrize(
    "func, rho_AB, alpha",
    [
        (petz_conditional_entropy_downarrow, RHO_INVALID, 1),
        (petz_conditional_entropy_uparrow, RHO_INVALID, 1),
        (sandwiched_conditional_entropy_downarrow, RHO_INVALID, 1),
    ]
)
def test_invalid_density_matrix(func, rho_AB, alpha):
    with pytest.raises(ValueError):
        func(rho_AB, alpha)
