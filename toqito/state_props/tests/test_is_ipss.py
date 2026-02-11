"""Test iterative_product_state_subtraction."""

import numpy as np
import pytest

from toqito.state_props.iterative_product_state_subtraction import (
    iterative_product_state_subtraction,
    verify_separable_decomposition,
)


def _check_trace_conservation(rho, decomp, residual, tol=1e-6):
    """Verify trace is conserved across decomposition and residual."""
    total_weight = sum(float(w) for w, _ in decomp)
    trace_resid = float(np.real(np.trace(residual)))
    trace_rho = float(np.real(np.trace(rho)))
    assert abs(total_weight + trace_resid - trace_rho) < tol


def _check_projectors(decomp, tol=1e-8):
    """Verify projectors are Hermitian, PSD, and approximately rank-1."""
    for _, P in decomp:
        # Hermitian
        assert np.allclose(P, P.conj().T, atol=tol)
        # PSD
        eigs = np.linalg.eigvalsh(P)
        assert np.min(eigs) >= -tol
        # Approximately rank-1
        assert eigs[-1] > 0.99
        assert np.max(np.abs(eigs[:-1])) < 1e-6


@pytest.mark.parametrize(
    "rho, dim, max_iter, expected_separable, min_residual_norm",
    [
        # Product state |00⟩
        (
            np.diag([1.0, 0.0, 0.0, 0.0]).astype(complex),
            [2, 2],
            None,
            True,
            None,
        ),
        # Classical mixture of |00⟩ and |11⟩
        (
            np.diag([0.6, 0.0, 0.0, 0.4]).astype(complex),
            [2, 2],
            None,
            True,
            None,
        ),
        # Bell state |Φ+⟩ - entangled
        (
            np.outer(
                np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
                np.array([1, 0, 0, 1], dtype=complex).conj() / np.sqrt(2),
            ),
            [2, 2],
            100,
            False,
            0.5,
        ),
        # Qubit-qutrit product state
        (
            np.outer(
                np.kron(np.array([1, 0], dtype=complex), np.array([0, 0, 1], dtype=complex)),
                np.kron(np.array([1, 0], dtype=complex), np.array([0, 0, 1], dtype=complex)).conj(),
            ),
            [2, 3],
            None,
            True,
            None,
        ),
    ],
)
def test_iterative_product_state_subtraction(rho, dim, max_iter, expected_separable, min_residual_norm):
    """Test function works as expected for valid inputs."""
    kwargs = {"rho": rho, "dim": dim}
    if max_iter is not None:
        kwargs["max_iterations"] = max_iter

    is_sep, decomp, residual = iterative_product_state_subtraction(**kwargs)

    assert is_sep == expected_separable

    if expected_separable:
        assert verify_separable_decomposition(rho, decomp, atol=1e-6)
        assert np.linalg.norm(residual, "fro") < 1e-6
        _check_trace_conservation(rho, decomp, residual)
        _check_projectors(decomp)
    elif min_residual_norm is not None:
        assert np.linalg.norm(residual, "fro") > min_residual_norm


def test_werner_state_entangled():
    """Werner state with p=0.5 should be detected as entangled."""
    p = 0.5
    psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    bell_dm = np.outer(psi_minus, psi_minus.conj())
    rho = p * bell_dm + (1 - p) * np.eye(4, dtype=complex) / 4

    is_sep, _, _ = iterative_product_state_subtraction(rho, [2, 2], max_iterations=200)
    assert not is_sep


def test_dimension_mismatch():
    """Dimension mismatch should raise ValueError."""
    rho = np.eye(4, dtype=complex) / 4
    with pytest.raises(ValueError):
        iterative_product_state_subtraction(rho, [2, 3])


def test_non_bipartite():
    """Non-bipartite systems should raise ValueError."""
    rho = np.eye(8, dtype=complex) / 8
    with pytest.raises(ValueError):
        iterative_product_state_subtraction(rho, [2, 2, 2])


def test_invalid_trace():
    """Matrix with trace != 1 should raise ValueError."""
    rho = np.eye(4, dtype=complex)
    with pytest.raises(ValueError):
        iterative_product_state_subtraction(rho, [2, 2])
