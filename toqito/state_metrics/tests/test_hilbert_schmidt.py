"""Tests for hilbert_schmidt."""
import numpy as np

from toqito.state_metrics import hilbert_schmidt
from toqito.states import bell


def test_hilbert_schmidt_bell():
    r"""Test Hilbert-Schmidt distance on two Bell states."""

    rho = bell(0) * bell(0).conj().T
    sigma = bell(3) * bell(3).conj().T

    res = hilbert_schmidt(rho, sigma)

    np.testing.assert_equal(np.isclose(res, 1), True)


def test_hilbert_schmidt_non_density_matrix():
    r"""Test Hilbert-Schmidt distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        hilbert_schmidt(rho, sigma)


if __name__ == "__main__":
    np.testing.run_module_suite()
