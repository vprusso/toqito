"""Tests for hilbert_schmidt."""
import numpy as np
from toqito.matrices.hadamard import hadamard
from toqito.matrices.pauli import pauli

from toqito.state_metrics import hilbert_schmidt_inner_product
from toqito.states import bell


def test_hilbert_schmidt_same_operator():
    r"""Test Hilbert-Schmidt distance between an unitary and itself"""

    H = hadamard(1)

    hs_ip = hilbert_schmidt_inner_product(H, H)

    np.testing.assert_equal(np.isclose(hs_ip, 1), True)


# def test_hilbert_schmidt_non_density_matrix():
#     r"""Test Hilbert-Schmidt distance on non-density matrix."""
#     rho = np.array([[1, 2], [3, 4]])
#     sigma = np.array([[5, 6], [7, 8]])

#     with np.testing.assert_raises(ValueError):
#         hilbert_schmidt(rho, sigma)


if __name__ == "__main__":
    np.testing.run_module_suite()