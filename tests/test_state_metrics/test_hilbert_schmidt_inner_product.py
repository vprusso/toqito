"""Tests for hilbert_schmidt."""
import numpy as np
from sympy import betainc
from toqito.matrices.hadamard import hadamard
from toqito.matrices.pauli import pauli
from toqito.random.random_unitary import random_unitary

from toqito.state_metrics import hilbert_schmidt_inner_product


def test_hilbert_schmidt_inner_product_hadamard_hadamard():
    r"""Test Hilbert-Schmidt inner product between an unitary and itself should
    return the dimension of the space the operator acts on"""

    H = hadamard(1)
    hs_ip = hilbert_schmidt_inner_product(H, H)
    np.testing.assert_equal(np.isclose(hs_ip, 2), True)
    
def test_hilbert_schmidt_inner_product_is_conjugate_symmetric():
    r"""Test Hilbert-Schmidt inner product is conjugate symmetric for two matrices"""

    A = random_unitary(2)
    B = random_unitary(2)
    hs_ip_A_B = hilbert_schmidt_inner_product(A, B)
    hs_ip_B_A = hilbert_schmidt_inner_product(B, A)
    print(hs_ip_A_B, hs_ip_B_A)
    np.testing.assert_equal(np.isclose(hs_ip_A_B, np.conj(hs_ip_B_A)), True)

def test_hilbert_schmidt_inner_product_linearity():
  r"""Test Hilbert-Schmidt inner product acts linearly"""
  rand_unitary = random_unitary(2)
  random_hermitian_operator = rand_unitary + np.conj(rand_unitary.T)
  B_1 = pauli("I")
  B_2 = 2*B_1
  beta_1 = 0.3
  beta_2 = 0.8
  LHS = beta_1 * hilbert_schmidt_inner_product(random_hermitian_operator, B_1) + \
    beta_2 * hilbert_schmidt_inner_product(random_hermitian_operator, B_2)
  RHS = hilbert_schmidt_inner_product(random_hermitian_operator, beta_1 * B_1 + beta_2 * B_2)
  assert np.allclose(LHS, RHS)
    
if __name__ == "__main__":
    np.testing.run_module_suite()