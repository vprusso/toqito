"""Test is_separable."""
import numpy as np

from toqito.channels import partial_trace
from toqito.matrix_props import is_density
from toqito.rand import random_density_matrix
from toqito.state_props.is_separable import is_separable
from toqito.states import basis, bell, isotropic, tile


def test_non_positive_semidefinite_matrix():
    """Ensure separability of non-positive semidefinite matrix is invalid."""
    with np.testing.assert_raises(ValueError):
        state = np.array([[-1, -1], [-1, -1]])
        is_separable(state)


def test_psd_matrix_local_dim_one():
    """Every positive semidefinite matrix is separable when one of the local dimensions is 1."""
    np.testing.assert_equal(is_separable(np.identity(2)), True)


def test_invalid_dim_parameter():
    """The dimension of the state must evenly divide the length of the state."""
    with np.testing.assert_raises(ValueError):
        dim = 3
        rho = isotropic(dim, 1 / (dim + 1))
        is_separable(rho, dim + 1)


def test_entangled_ppt_criterion():
    """Determined to be entangled via the PPT criterion."""
    rho = bell(0) * bell(0).conj().T
    np.testing.assert_equal(is_separable(rho), False)


def test_ppt_small_dimensions():
    """Determined to be separable via sufficiency of the PPT criterion in small dimensions."""
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    psi = 1 / np.sqrt(3) * e_0 + 1 / np.sqrt(3) * e_1 + 1 / np.sqrt(3) * e_2

    e_0, e_1 = basis(2, 0), basis(2, 1)
    phi = np.kron((1 / np.sqrt(2) * e_0 + 1 / np.sqrt(2) * e_1), psi)
    sigma = phi * phi.conj().T
    np.testing.assert_equal(is_separable(sigma), True)


def test_ppt_low_rank():
    """Determined to be separable via the operational criterion for low-rank operators."""
    m = 6
    n = m
    rho = random_density_matrix(m)
    u, s, v_h = np.linalg.svd(rho)
    rho_cut = u[:, : m - 1] @ np.diag(s[: m - 1]) @ v_h[: m - 1]
    rho_cut = rho_cut / np.trace(rho_cut)
    pt_state_alice = partial_trace(rho_cut, [1], [3, 2])

    np.testing.assert_equal(is_density(rho_cut), True)
    np.testing.assert_equal(is_density(np.array(pt_state_alice)), True)
    np.testing.assert_equal(
        np.linalg.matrix_rank(rho_cut) + np.linalg.matrix_rank(pt_state_alice)
        <= 2 * m * n - m - n + 2,
        True,
    )
    # TODO
    # np.testing.assert_equal(is_separable(rho), True)


def test_entangled_realignment_criterion():
    """Determined to be entangled via the realignment criterion."""
    # Construct bound entangled state:
    # :math:`\rho = \frac{1}{4} \mathbb{I}_3 \otimes \mathbb{I}_3 - \sum_{i=0}^4 | \psi_i \rangle \langle \psi_i |`
    rho = np.identity(9)
    for i in range(5):
        rho = rho - tile(i) * tile(i).conj().T
    rho = rho / 4
    np.testing.assert_equal(is_density(rho), True)
    np.testing.assert_equal(is_separable(rho), False)


def test_entangled_cross_norm_realignment_criterion():
    """Determined to be entangled by using Theorem 1 and Remark 1 of Quantum Inf. Comput., 3:193-202, 2003."""
    p_var, a_var, b_var = 0.4, 0.8, 0.64
    rho = np.array(
        [
            [p_var * a_var ** 2, 0, 0, p_var * a_var * b_var],
            [0, (1 - p_var) * a_var ** 2, (1 - p_var) * a_var * b_var, 0],
            [0, (1 - p_var) * a_var * b_var, (1 - p_var) * a_var ** 2, 0],
            [p_var * a_var * b_var, 0, 0, p_var * a_var ** 2],
        ]
    )
    np.testing.assert_equal(is_separable(rho), False)
