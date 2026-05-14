"""Tests for operator_relative_entropy_epi_cone."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re
from importlib import import_module
from unittest.mock import patch

import cvxpy
import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power, inv, logm

from toqito.matrix_ops.operator_relative_entropy_epi_cone import (
    operator_relative_entropy_epi_cone,
)

_ore_module = import_module("toqito.matrix_ops.operator_relative_entropy_epi_cone")


def _rand_psd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    mat = (mat + mat.conj().T) / 2
    mat = mat / np.trace(mat)
    return mat


def _d_op_reference(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    sa = fractional_matrix_power(mat_a, 0.5)
    mid = sa @ inv(mat_b) @ sa
    return sa @ logm(mid) @ sa


@pytest.mark.parametrize("dim", [3, 5, 8])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_operator_relative_entropy_epi_cone_trace_minimum(
    dim: int, mk: int, apx: int, hermitian: bool
):
    """Minimize ``trace(TAU)`` and compare ``TAU`` to the operator relative entropy reference."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    mat_b = _rand_psd_normalized(dim, seed + 1, hermitian=hermitian)
    dop = _d_op_reference(mat_a, mat_b)

    a_c = cvxpy.Constant(mat_a)
    b_c = cvxpy.Constant(mat_b)
    if hermitian:
        TAU = cvxpy.Variable((dim, dim), hermitian=True)
    else:
        TAU = cvxpy.Variable((dim, dim), symmetric=True)

    cons = operator_relative_entropy_epi_cone(
        a_c,
        b_c,
        TAU,
        m=mk,
        k=mk,
        e=np.eye(dim),
        apx=apx,
        hermitian=hermitian,
    )
    obj = cvxpy.trace(TAU)
    if hermitian:
        obj = cvxpy.real(obj)
    prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)
    prob.solve(solver=cvxpy.SCS, verbose=False)

    assert prob.status in {cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE}, prob.status
    assert TAU.value is not None

    err = (TAU.value - dop) / np.linalg.norm(dop)
    eig_err = np.linalg.eigvalsh((err + err.conj().T) / 2).real
    assert float(np.min(apx * eig_err)) >= -5e-4
    if mk >= 3:
        assert float(np.linalg.norm(err, ord="fro")) <= 2e-2


def test_operator_relative_entropy_epi_cone_x_not_2d():
    """Reject ``X`` that is not two-dimensional."""
    with pytest.raises(ValueError, match=re.escape("X must be 2D.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.array([1.0, 2.0])),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 2), symmetric=True),
        )


def test_operator_relative_entropy_epi_cone_x_not_square():
    """Reject non-square ``X``."""
    with pytest.raises(ValueError, match=re.escape("X must be square.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Variable((2, 3)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 2), symmetric=True),
        )


def test_operator_relative_entropy_epi_cone_y_not_2d():
    """Reject ``Y`` that is not two-dimensional."""
    with pytest.raises(ValueError, match=re.escape("Y must be 2D.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.array([1.0, 2.0])),
            cvxpy.Variable((2, 2), symmetric=True),
        )


def test_operator_relative_entropy_epi_cone_y_not_square():
    """Reject non-square ``Y``."""
    with pytest.raises(ValueError, match=re.escape("Y must be square.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 3)),
            cvxpy.Variable((2, 2), symmetric=True),
        )


def test_operator_relative_entropy_epi_cone_tau_not_2d():
    """Reject ``TAU`` that is not two-dimensional."""
    with pytest.raises(ValueError, match=re.escape("TAU must be 2D.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.array([1.0, 2.0])),
        )


def test_operator_relative_entropy_epi_cone_tau_not_square():
    """Reject non-square ``TAU``."""
    with pytest.raises(ValueError, match=re.escape("TAU must be square.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 3)),
        )


def test_operator_relative_entropy_epi_cone_y_shape_mismatch():
    """Reject ``Y`` whose shape does not match ``X``."""
    with pytest.raises(ValueError, match=re.escape("Y must have the same shape as X.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(3)),
            cvxpy.Variable((2, 2), symmetric=True),
        )


def test_operator_relative_entropy_epi_cone_e_not_2d():
    """Reject ``e`` that is not two-dimensional."""
    with pytest.raises(ValueError, match=re.escape("e must be 2D.")):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 2), symmetric=True),
            e=np.array([1.0, 2.0, 3.0]),
        )


def test_operator_relative_entropy_epi_cone_e_row_mismatch():
    """Reject ``e`` whose row count does not match ``X``."""
    with pytest.raises(
        ValueError, match=re.escape("The number of rows of e must match X.")
    ):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 2), symmetric=True),
            e=np.eye(3, 2),
        )


def test_operator_relative_entropy_epi_cone_tau_r_mismatch():
    """Reject ``TAU`` when its size does not match ``e.shape[1]``."""
    with pytest.raises(
        ValueError, match=re.escape("TAU must be r x r with r = e.shape[1].")
    ):
        operator_relative_entropy_epi_cone(
            cvxpy.Constant(np.eye(2)),
            cvxpy.Constant(np.eye(2)),
            cvxpy.Variable((2, 2), symmetric=True),
            e=np.eye(2, 1),
        )


@pytest.mark.parametrize("bad_m", [0, -1])
def test_operator_relative_entropy_epi_cone_invalid_m(bad_m: int):
    """Reject ``m`` below 1 for the epigraph cone."""
    n = 2
    X = cvxpy.Constant(np.eye(n))
    Y = cvxpy.Constant(np.eye(n))
    TAU = cvxpy.Variable((n, n), symmetric=True)
    with pytest.raises(ValueError, match=re.escape("m and k must be at least 1.")):
        operator_relative_entropy_epi_cone(X, Y, TAU, m=bad_m, k=1)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_operator_relative_entropy_epi_cone_invalid_k(bad_k: int):
    """Reject ``k`` below 1 for the epigraph cone."""
    n = 2
    X = cvxpy.Constant(np.eye(n))
    Y = cvxpy.Constant(np.eye(n))
    TAU = cvxpy.Variable((n, n), symmetric=True)
    with pytest.raises(ValueError, match=re.escape("m and k must be at least 1.")):
        operator_relative_entropy_epi_cone(X, Y, TAU, m=1, k=bad_k)


@pytest.mark.parametrize("bad_apx", [2, -2, 99])
def test_operator_relative_entropy_epi_cone_invalid_apx(bad_apx: int):
    """Reject ``apx`` outside ``{-1, 0, 1}``."""
    n = 2
    X = cvxpy.Constant(np.eye(n))
    Y = cvxpy.Constant(np.eye(n))
    TAU = cvxpy.Variable((n, n), symmetric=True)
    with pytest.raises(ValueError, match=re.escape("apx must be either -1, 0, or 1.")):
        operator_relative_entropy_epi_cone(X, Y, TAU, apx=bad_apx)


def test_operator_relative_entropy_epi_cone_zero_quadrature_weight():
    """Raise when a mocked Gauss-Legendre weight is zero."""
    n = 2
    X = cvxpy.Constant(np.eye(n))
    Y = cvxpy.Constant(np.eye(n))
    TAU = cvxpy.Variable((n, n), symmetric=True)
    with patch.object(
        _ore_module,
        "_gauss_legendre",
        return_value=(np.array([0.5, 0.5]), np.array([1.0, 0.0])),
    ):
        with pytest.raises(
            ValueError, match=re.escape("Quadrature weight must be positive.")
        ):
            operator_relative_entropy_epi_cone(X, Y, TAU, m=2, apx=0)


@pytest.mark.parametrize("bad_m", [0, -3])
def test_gauss_legendre_invalid_m(bad_m: int):
    """Reject ``m`` below 1 in ``_gauss_legendre``."""
    with pytest.raises(ValueError, match=re.escape("m must be at least 1.")):
        _ore_module._gauss_legendre(bad_m)


def test_gauss_legendre_single_node() -> None:
    """``m == 1`` returns the midpoint rule on ``[0, 1]``."""
    nodes, weights = _ore_module._gauss_legendre(1)
    np.testing.assert_allclose(nodes, np.array([0.5]))
    np.testing.assert_allclose(weights, np.array([1.0]))


@pytest.mark.parametrize("bad_m", [0, -1])
def test_gauss_radau_invalid_m(bad_m: int):
    """Reject ``m`` below 1 in ``_gauss_radau``."""
    with pytest.raises(ValueError, match=re.escape("m must be at least 1.")):
        _ore_module._gauss_radau(bad_m, 0)


def test_gauss_radau_invalid_endpoint():
    """Reject ``endpoint`` outside ``{0, 1}`` in ``_gauss_radau``."""
    with pytest.raises(ValueError, match=re.escape("endpoint must be either 0 or 1.")):
        _ore_module._gauss_radau(3, 2)
