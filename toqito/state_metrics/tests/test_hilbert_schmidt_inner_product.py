"""Tests for hilbert_schmidt."""

import numpy as np

from toqito.matrices.hadamard import hadamard
from toqito.matrices.pauli import pauli
from toqito.rand.random_unitary import random_unitary
from toqito.state_metrics import hilbert_schmidt_inner_product


def test_hilbert_schmidt_inner_product_hadamard_hadamard():
    r"""Test Hilbert-Schmidt inner product between an unitary and itself.

    Output should return the dimension of the space the operator acts on
    """
    hadamard_mat = hadamard(1)
    hs_ip = hilbert_schmidt_inner_product(hadamard_mat, hadamard_mat)
    np.testing.assert_equal(np.isclose(hs_ip, 2), True)


def test_hilbert_schmidt_inner_product_is_conjugate_symmetric():
    r"""Test Hilbert-Schmidt inner product is conjugate symmetric for two matrices."""
    random_mat_1 = random_unitary(2)
    random_mat_2 = random_unitary(2)
    hs_ip_1_2 = hilbert_schmidt_inner_product(random_mat_1, random_mat_2)
    hs_ip_2_1 = hilbert_schmidt_inner_product(random_mat_2, random_mat_1)
    np.testing.assert_equal(np.isclose(hs_ip_1_2, np.conj(hs_ip_2_1)), True)


def test_hilbert_schmidt_inner_product_linearity():
    r"""Test Hilbert-Schmidt inner product acts linearly."""
    rand_unitary = random_unitary(2)
    random_hermitian_operator = rand_unitary + np.conj(rand_unitary.T)
    b_mat_1 = pauli("I")
    b_mat_2 = 2 * b_mat_1
    beta_1 = 0.3
    beta_2 = 0.8
    lhs = beta_1 * hilbert_schmidt_inner_product(
        random_hermitian_operator, b_mat_1
    ) + beta_2 * hilbert_schmidt_inner_product(random_hermitian_operator, b_mat_2)
    rhs = hilbert_schmidt_inner_product(random_hermitian_operator, beta_1 * b_mat_1 + beta_2 * b_mat_2)
    np.testing.assert_equal(np.isclose(lhs, rhs), True)
