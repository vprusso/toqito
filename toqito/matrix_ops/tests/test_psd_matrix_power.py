"""Tests for psd_matrix_power."""

import numpy as np

from toqito.matrix_ops import psd_matrix_power


def test_psd_matrix_power_full_rank_square_root():
    """The square root of a diagonal PSD matrix is the elementwise square root."""
    mat = np.diag([4.0, 9.0])
    np.testing.assert_allclose(psd_matrix_power(mat, 0.5), np.diag([2.0, 3.0]))


def test_psd_matrix_power_negative_power_on_support():
    """A negative power acts on the support and leaves the kernel at zero."""
    mat = np.diag([4.0, 0.0])
    np.testing.assert_allclose(psd_matrix_power(mat, -1.0), np.diag([0.25, 0.0]))


def test_psd_matrix_power_complex_hermitian():
    """A power of a complex Hermitian PSD matrix matches conjugation by its eigenvectors."""
    angle = np.array([[1.0, 1j], [-1j, 1.0]])
    powered = psd_matrix_power(angle, 2.0)
    np.testing.assert_allclose(powered, angle @ angle, atol=1e-10)
