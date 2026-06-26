"""Tests for hilbert_schmidt."""

import numpy as np

from toqito.state_metrics import hilbert_schmidt
from toqito.states import bell


def test_hilbert_schmidt_bell():
    r"""Test Hilbert-Schmidt distance on two Bell states."""
    rho = bell(0) @ bell(0).conj().T
    sigma = bell(3) @ bell(3).conj().T

    res = hilbert_schmidt(rho, sigma)

    # The two Bell states are orthogonal pure states, so rho - sigma has eigenvalues +1 and -1
    # and Tr((rho - sigma)^2) = 2.
    np.testing.assert_allclose(res, 2)


def test_hilbert_schmidt_matches_trace_formula():
    r"""Hilbert-Schmidt distance equals Tr((rho - sigma)^2) for non-commuting mixed states."""
    rho = np.array([[0.6, 0.2], [0.2, 0.4]])
    sigma = np.array([[0.4, 0.3], [0.3, 0.6]])

    res = hilbert_schmidt(rho, sigma)

    diff = rho - sigma
    np.testing.assert_allclose(res, np.real(np.trace(diff @ diff)))


def test_hilbert_schmidt_non_density_matrix():
    r"""Test Hilbert-Schmidt distance on non-density matrix."""
    rho = np.array([[1, 2], [3, 4]])
    sigma = np.array([[5, 6], [7, 8]])

    with np.testing.assert_raises(ValueError):
        hilbert_schmidt(rho, sigma)
