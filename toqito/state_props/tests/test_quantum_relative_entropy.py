"""Tests for quantum_relative_entropy."""

# Adapted from CVXQUAD (https://github.com/hfawzi/cvxquad), BSD-2-Clause.
# Original implementation by Fawzi, Saunderson, et al.

import re

import cvxpy
import numpy as np
import pytest
from scipy.linalg import logm

from toqito.matrix_props import is_positive_semidefinite
from toqito.state_props.quantum_relative_entropy import quantum_relative_entropy


def _rand_psd(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    """Random PSD matrix."""
    rng = np.random.default_rng(seed)
    if hermitian:
        g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    else:
        g = rng.standard_normal((dim, dim))
    mat = g @ g.conj().T + 1e-1 * np.eye(dim, dtype=g.dtype)
    return (mat + mat.conj().T) / 2


def _rand_psd_normalized(dim: int, seed: int, *, hermitian: bool) -> np.ndarray:
    mat = _rand_psd(dim, seed, hermitian=hermitian)
    return mat / np.trace(mat)


def _affine_fixed_at(mat: np.ndarray, *, hermitian: bool) -> cvxpy.Expression:
    """``Constant(A) + W - W`` with ``W.value = 0`` (algebraically ``A``)."""
    n = mat.shape[0]
    if hermitian:
        w_var = cvxpy.Variable((n, n), hermitian=True)
        w_var.value = np.zeros((n, n), dtype=np.complex128)
    else:
        w_var = cvxpy.Variable((n, n), symmetric=True)
        w_var.value = np.zeros((n, n))
    expr = cvxpy.Constant(mat) + w_var - w_var
    assert expr.is_affine() and not expr.is_constant()
    return expr


def _qre_sdp_fixed_y(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    *,
    m: int,
    k: int,
    apx: int,
    hermitian: bool,
    space_optimized: bool = False,
) -> float:
    """Like CVXQUAD ``minimize quantum_rel_entr(X,B)`` with ``X == A``."""
    return quantum_relative_entropy(
        _affine_fixed_at(mat_a, hermitian=hermitian),
        cvxpy.Constant(mat_b),
        m=m,
        k=k,
        apx=apx,
        space_optimized=space_optimized,
    )


def _qre_sdp_fixed_x(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    *,
    m: int,
    k: int,
    apx: int,
    hermitian: bool,
    space_optimized: bool = False,
) -> float:
    """Like CVXQUAD ``minimize quantum_rel_entr(A,Y)`` with ``Y == B``."""
    return quantum_relative_entropy(
        cvxpy.Constant(mat_a),
        _affine_fixed_at(mat_b, hermitian=hermitian),
        m=m,
        k=k,
        apx=apx,
        space_optimized=space_optimized,
    )


def _qre_sdp_fixed_xy(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    *,
    m: int,
    k: int,
    apx: int,
    hermitian: bool,
    space_optimized: bool = False,
) -> float:
    """Like CVXQUAD ``minimize quantum_rel_entr(X,Y)`` with ``X == A``, ``Y == B``."""
    return quantum_relative_entropy(
        _affine_fixed_at(mat_a, hermitian=hermitian),
        _affine_fixed_at(mat_b, hermitian=hermitian),
        m=m,
        k=k,
        apx=apx,
        space_optimized=space_optimized,
    )


@pytest.mark.parametrize("space_optimized", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mk", [1, 3])
@pytest.mark.parametrize("apx", [-1, 0, 1])
@pytest.mark.parametrize("hermitian", [False, True])
def test_quantum_relative_entropy(
    dim: int,
    mk: int,
    apx: int,
    hermitian: bool,
    space_optimized: bool,
):
    """Like CVXQUAD ``test_quantum_rel_entr``: three SDP branches vs numeric reference."""
    if mk == 1 and apx == 0:
        pytest.skip("CVXQUAD skips (m,k)=(1,1) with Pade apx=0.")

    seed = dim * 100_003 + mk * 17 + (apx + 1) * 3 + int(hermitian)
    mat_a = _rand_psd_normalized(dim, seed, hermitian=hermitian)
    mat_b = _rand_psd_normalized(dim, seed + 1, hermitian=hermitian)

    assert is_positive_semidefinite(np.asarray(mat_a, dtype=np.complex128))
    assert is_positive_semidefinite(np.asarray(mat_b, dtype=np.complex128))

    dab = quantum_relative_entropy(mat_a, mat_b)
    val1 = _qre_sdp_fixed_y(
        mat_a,
        mat_b,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
        space_optimized=space_optimized,
    )
    val2 = _qre_sdp_fixed_x(
        mat_a,
        mat_b,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
        space_optimized=space_optimized,
    )
    val12 = _qre_sdp_fixed_xy(
        mat_a,
        mat_b,
        m=mk,
        k=mk,
        apx=apx,
        hermitian=hermitian,
        space_optimized=space_optimized,
    )

    if abs(dab) < 1e-12:
        assert abs(val1 - dab) <= 1e-4
        assert abs(val2 - dab) <= 1e-4
        assert abs(val12 - dab) <= 1e-4
        return

    err1 = (val1 - dab) / abs(dab)
    err2 = (val2 - dab) / abs(dab)
    err12 = (val12 - dab) / abs(dab)

    if apx != 0:
        assert apx * err1 >= -5e-4, err1
        assert apx * err2 >= -5e-4, err2
        if not space_optimized:
            assert apx * err12 >= -5e-4, err12

    if mk >= 3:
        assert abs(err1) <= 1e-2, err1
        assert abs(err2) <= 1e-2, err2
        if not space_optimized:
            assert abs(err12) <= 1e-2, err12

    if space_optimized:
        assert abs(err12) <= 1.5e-1, err12


def test_quantum_relative_entropy_commuting_reference():
    """Diagonal ``X``, ``Y`` share eigenbasis; match ``tr(X(log X - log Y))``."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.4, 0.6])
    ref = float(np.real(np.trace(mat_x @ (logm(mat_x) - logm(mat_y)))))
    got = quantum_relative_entropy(mat_x, mat_y)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_x_branch():
    """Constant ``mat_x`` uses ``ln_quantum_entropy`` + ``trace_matrix_log``."""
    n = 2
    rng = np.random.default_rng(21)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    expr_x = cvxpy.Constant(mat_a)
    got = quantum_relative_entropy(expr_x, mat_b, m=3, k=3, apx=0)
    want = quantum_relative_entropy(mat_a, mat_b, m=3, k=3, apx=0)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_x_constant_y_branch():
    """Constant ``mat_x`` and constant ``mat_y`` both use numeric ``.value`` for ``trace_matrix_log``."""
    n = 2
    rng = np.random.default_rng(27)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    got = quantum_relative_entropy(cvxpy.Constant(mat_a), cvxpy.Constant(mat_b), m=3, k=3, apx=0)
    want = quantum_relative_entropy(mat_a, mat_b, m=3, k=3, apx=0)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def test_quantum_relative_entropy_constant_x_mat_y_bad_type():
    """Reject non-array / non-CVXPY ``mat_y`` on the constant-``X`` branch."""
    with pytest.raises(
        ValueError,
        match=re.escape("mat_y must be a numpy array or a cvxpy expression"),
    ):
        quantum_relative_entropy(cvxpy.Constant(np.eye(2)), [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("space_optimized", [False, True])
def test_quantum_relative_entropy_constant_x_affine_y_branch(space_optimized: bool):
    """Constant ``mat_x`` with affine ``mat_y`` uses ``trace_matrix_log`` on ``mat_y``."""
    n = 2
    rng = np.random.default_rng(31)
    g = rng.standard_normal((n, n))
    mat_a = g @ g.T + np.eye(n)
    mat_a = (mat_a + mat_a.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    expr_x = cvxpy.Constant(mat_a)
    expr_y = _affine_fixed_at(mat_b, hermitian=False)
    got = quantum_relative_entropy(expr_x, expr_y, m=3, k=3, apx=0, space_optimized=space_optimized)
    want = quantum_relative_entropy(mat_a, mat_b, m=3, k=3, apx=0)
    np.testing.assert_allclose(float(got), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("space_optimized", [False, True])
def test_quantum_relative_entropy_constant_y_branch(space_optimized: bool):
    """Constant ``mat_y`` uses affine-``X`` branch with ``logm(Y)``."""
    n = 3
    rng = np.random.default_rng(23)
    g = rng.standard_normal((n, n))
    a0 = g @ g.T + 0.3 * np.eye(n)
    mat_a = (a0 + a0.T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    w_var = cvxpy.Variable((n, n), symmetric=True)
    w_var.value = np.zeros((n, n))
    mat_x = cvxpy.Constant(mat_a) + w_var - w_var
    got = quantum_relative_entropy(
        mat_x,
        cvxpy.Constant(mat_b),
        m=3,
        k=3,
        apx=1,
        space_optimized=space_optimized,
    )
    want = quantum_relative_entropy(mat_a, mat_b, m=3, k=3, apx=1)
    np.testing.assert_allclose(float(got), want, rtol=5e-4, atol=1e-6)


@pytest.mark.parametrize("space_optimized", [False, True])
def test_quantum_relative_entropy_joint_affine_hermitian(space_optimized: bool):
    """Hermitian joint-affine branch vs numeric reference."""
    n = 2
    rng = np.random.default_rng(29)
    g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h0 = g @ g.conj().T + 0.4 * np.eye(n)
    mat_a = (h0 + h0.conj().T) / 2
    mat_a = mat_a / np.trace(mat_a)
    mat_b = np.eye(n) / n
    val = _qre_sdp_fixed_xy(
        mat_a,
        mat_b,
        m=3,
        k=3,
        apx=-1,
        hermitian=True,
        space_optimized=space_optimized,
    )
    ref = quantum_relative_entropy(mat_a, mat_b)
    if space_optimized:
        np.testing.assert_allclose(float(val), ref, rtol=5e-2, atol=1e-4)
    else:
        np.testing.assert_allclose(float(val), ref, rtol=5e-4, atol=1e-6)


def test_quantum_relative_entropy_support_inclusion_failure():
    """Numeric path rejects ``im(X) not subseteq im(Y)``."""
    mat_x = np.array([[1.0, 0.0], [0.0, 0.0]])
    mat_y = np.array([[0.0, 0.0], [0.0, 1.0]])
    with pytest.raises(
        ValueError,
        match=re.escape("D(X||Y) is infinity because im(X) is not contained in im(Y)"),
    ):
        quantum_relative_entropy(mat_x, mat_y)


class TestQuantumRelativeEntropyValueErrors:
    """``ValueError`` paths in ``quantum_relative_entropy``."""

    def test_mat_x_wrong_type(self):
        """Reject ``mat_x`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a numpy array or a cvxpy expression"),
        ):
            quantum_relative_entropy([[1.0, 0.0], [0.0, 1.0]], np.eye(2))

    def test_mat_y_wrong_type(self):
        """Reject ``mat_y`` that is not a numpy array or CVXPY expression."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_y must be a numpy array or a cvxpy expression"),
        ):
            quantum_relative_entropy(np.eye(2), [[1.0, 0.0], [0.0, 1.0]])

    def test_numpy_x_cvxpy_y_not_both_expressions(self):
        """Reject mixing a numpy ``mat_x`` with a non-constant CVXPY ``mat_y``."""
        n = 2
        y_var = cvxpy.Variable((n, n), symmetric=True)
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x and mat_y must be numpy arrays or cvxpy expressions"),
        ):
            quantum_relative_entropy(np.eye(n), y_var)

    def test_shape_mismatch(self):
        """Reject ``mat_x`` and ``mat_y`` with different shapes."""
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x and mat_y must have the same shape"),
        ):
            quantum_relative_entropy(np.eye(2), np.eye(3))

    def test_mat_x_not_2d(self):
        """Reject non-2D ``mat_x``."""
        with pytest.raises(ValueError, match=re.escape("mat_x must be 2D.")):
            quantum_relative_entropy(np.array([1.0, 0.0]), np.eye(2))

    def test_mat_y_not_square(self):
        """Reject non-square ``mat_y``."""
        with pytest.raises(ValueError, match=re.escape("mat_y must be square.")):
            quantum_relative_entropy(np.eye(2), np.zeros((2, 3)))

    def test_m_invalid(self):
        """Reject non-positive quadrature count ``m``."""
        with pytest.raises(ValueError, match=re.escape("m must be a positive integer")):
            quantum_relative_entropy(np.eye(2), np.eye(2), m=0)

    def test_k_invalid(self):
        """Reject non-positive square-root count ``k``."""
        with pytest.raises(ValueError, match=re.escape("k must be a positive integer")):
            quantum_relative_entropy(np.eye(2), np.eye(2), k=0)

    def test_apx_invalid(self):
        """Reject approximation flag outside ``{-1, 0, 1}``."""
        with pytest.raises(ValueError, match=re.escape("apx must be -1, 0, or 1")):
            quantum_relative_entropy(np.eye(2), np.eye(2), apx=2)

    def test_space_optimized_invalid(self):
        """Reject non-boolean ``space_optimized``."""
        with pytest.raises(ValueError, match=re.escape("space_optimized must be a boolean")):
            quantum_relative_entropy(np.eye(2), np.eye(2), space_optimized="yes")

    def test_mat_x_not_psd(self):
        """Reject non-PSD numeric ``mat_x``."""
        mat_x = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a positive semidefinite matrix"),
        ):
            quantum_relative_entropy(mat_x, np.eye(2))

    def test_mat_y_not_psd_numeric(self):
        """Reject non-PSD numeric ``mat_y``."""
        mat_y = np.diag([1.0, -0.5])
        with pytest.raises(
            ValueError,
            match=re.escape("mat_y must be a positive semidefinite matrix"),
        ):
            quantum_relative_entropy(np.eye(2), mat_y)

    def test_mat_x_not_hermitian(self):
        """Reject non-Hermitian numeric ``mat_x``."""
        mat_x = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match=re.escape("mat_x must be a Hermitian matrix")):
            quantum_relative_entropy(mat_x, np.eye(2))

    def test_mat_y_not_hermitian(self):
        """Reject non-Hermitian numeric ``mat_y``."""
        mat_y = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match=re.escape("mat_y must be a Hermitian matrix")):
            quantum_relative_entropy(np.eye(2), mat_y)

    def test_constant_x_no_value(self):
        """Reject constant ``mat_x`` with no ``.value``."""
        n = 2
        p_x = cvxpy.Parameter((n, n), symmetric=True)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_x as a numpy.ndarray."
            ),
        ):
            quantum_relative_entropy(p_x, np.eye(n))

    def test_constant_x_branch_mat_y_no_value(self):
        """Reject constant ``mat_y`` with no ``.value`` on the constant-``X`` branch."""
        n = 2
        p_x = cvxpy.Parameter((n, n), symmetric=True)
        p_x.value = np.eye(n)
        p_y = cvxpy.Parameter((n, n), symmetric=True)
        with pytest.raises(
            ValueError,
            match=re.escape("mat_y has no numeric value; pass a numpy.ndarray or set `.value`."),
        ):
            quantum_relative_entropy(p_x, p_y)


@pytest.mark.parametrize("space_optimized", [False, True])
class TestQuantumRelativeEntropyAffineValueErrors:
    """``ValueError`` paths on affine CVXPY branches."""

    def test_mat_y_not_psd_constant_y_branch(self, space_optimized: bool):
        """Reject non-PSD constant ``mat_y`` on the constant-``Y`` branch."""
        n = 2
        mat_x = _affine_fixed_at(np.eye(n), hermitian=False)
        bad_y = np.diag([1.0, -0.5])
        with pytest.raises(
            ValueError,
            match=re.escape("mat_y must be a positive semidefinite matrix"),
        ):
            quantum_relative_entropy(
                mat_x,
                cvxpy.Constant(bad_y),
                space_optimized=space_optimized,
            )

    def test_constant_y_no_value(self, space_optimized: bool):
        """Reject constant ``mat_y`` with no ``.value``."""
        n = 2
        p_y = cvxpy.Parameter((n, n), symmetric=True)
        mat_x = _affine_fixed_at(np.eye(n), hermitian=False)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Constant CVXPY expression has no numeric value; set parameter `.value` "
                "or pass mat_y as a numpy.ndarray."
            ),
        ):
            quantum_relative_entropy(mat_x, p_y, space_optimized=space_optimized)

    def test_not_affine(self, space_optimized: bool):
        """Reject non-affine CVXPY inputs on the joint branch."""
        n = 2
        x_var = cvxpy.Variable((n, n), symmetric=True)
        y_var = cvxpy.Variable((n, n), symmetric=True)
        x_var.value = np.eye(n)
        y_var.value = np.eye(n)
        with pytest.raises(
            ValueError,
            match="mat_x and mat_y must be affine CVXPY expressions",
        ):
            quantum_relative_entropy(
                x_var @ x_var,
                y_var,
                space_optimized=space_optimized,
            )

    def test_joint_affine_no_initial_value(self, space_optimized: bool):
        """Reject joint-affine inputs with no initial ``.value`` for PSD checks."""
        n = 2
        t_x = cvxpy.Variable()
        t_y = cvxpy.Variable()
        with pytest.raises(
            ValueError,
            match="need numeric initial values",
        ):
            quantum_relative_entropy(
                t_x * np.eye(n),
                t_y * np.eye(n),
                space_optimized=space_optimized,
            )

    def test_constant_y_branch_affine_x_no_initial_value(self, space_optimized: bool):
        """Reject affine ``mat_x`` with no ``.value`` on the constant-``Y`` branch."""
        n = 2
        t = cvxpy.Variable()
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x has no numeric value; pass a numpy.ndarray or set `.value`."),
        ):
            quantum_relative_entropy(
                t * np.eye(n),
                cvxpy.Constant(np.eye(n)),
                space_optimized=space_optimized,
            )

    def test_affine_x_not_psd_at_value_constant_y(self, space_optimized: bool):
        """Reject affine ``mat_x`` that is not PSD at ``.value`` (constant-``Y`` branch)."""
        n = 2
        t = cvxpy.Variable()
        t.value = -1.0
        with pytest.raises(
            ValueError,
            match=re.escape("mat_x must be a positive semidefinite matrix"),
        ):
            quantum_relative_entropy(
                t * np.eye(n),
                cvxpy.Constant(np.eye(n)),
                space_optimized=space_optimized,
            )

    def test_joint_affine_x_not_psd_at_value(self, space_optimized: bool):
        """Reject joint-affine ``mat_x`` that is not PSD at ``.value``."""
        n = 2
        t_x = cvxpy.Variable()
        t_y = cvxpy.Variable()
        t_x.value = -1.0
        t_y.value = 1.0
        with pytest.raises(
            ValueError,
            match="mat_x must be positive semidefinite",
        ):
            quantum_relative_entropy(
                t_x * np.eye(n),
                t_y * np.eye(n),
                space_optimized=space_optimized,
            )

    def test_joint_affine_y_not_psd_at_value(self, space_optimized: bool):
        """Reject joint-affine ``mat_y`` that is not PSD at ``.value``."""
        n = 2
        t_x = cvxpy.Variable()
        t_y = cvxpy.Variable()
        t_x.value = 1.0
        t_y.value = -1.0
        with pytest.raises(
            ValueError,
            match="mat_y must be positive semidefinite",
        ):
            quantum_relative_entropy(
                t_x * np.eye(n),
                t_y * np.eye(n),
                space_optimized=space_optimized,
            )


def test_quantum_relative_entropy_numpy_still_works():
    """Numeric numpy path is unaffected by the guard."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    result = quantum_relative_entropy(mat_x, mat_y)
    assert np.isfinite(result)
    assert result > 0


def test_quantum_relative_entropy_constant_cvxpy_still_works():
    """Constant CVXPY expressions (no free variables) must not be rejected."""
    mat_x = np.diag([0.7, 0.3])
    mat_y = np.diag([0.6, 0.4])
    result = quantum_relative_entropy(cvxpy.Constant(mat_x), cvxpy.Constant(mat_y))
    assert np.isfinite(result)
    assert result > 0
