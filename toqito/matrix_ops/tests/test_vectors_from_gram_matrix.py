"""Test vectors_from_gram_matrix."""

import numpy as np
import pytest

from toqito.matrix_ops import vectors_from_gram_matrix


@pytest.mark.parametrize(
    "gram, expected_result",
    [
        # Gram matrix is identity matrix.
        (
            np.identity(4),
            [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])],
        ),
    ],
)
def test_vectors_from_gram_matrix(gram, expected_result):
    """Test able to extract vectors from Gram matrix."""
    vectors = vectors_from_gram_matrix(gram)
    np.testing.assert_allclose(vectors, expected_result)


@pytest.mark.parametrize(
    "gram",
    [
        # Positive-definite real Gram matrix (Cholesky path).
        np.array([[2, -1], [-1, 2]], dtype=float),
        # Positive-definite complex Hermitian Gram matrix (Cholesky path).
        np.array([[2, 1j], [-1j, 2]], dtype=complex),
    ],
)
def test_vectors_from_gram_matrix_reconstructs(gram):
    """The returned vectors reconstruct a positive-definite Gram matrix."""
    mat = np.array(vectors_from_gram_matrix(gram))
    np.testing.assert_allclose(mat @ mat.conj().T, gram, atol=1e-8)


@pytest.mark.parametrize(
    "gram",
    [
        # Rank-deficient (PSD but not PD) real Gram matrix: Cholesky fails, eigendecomposition fallback is used.
        np.array([[1, 1], [1, 1]], dtype=float),
        # Singular trine Gram matrix (eigenvalues 0, 3/2, 3/2).
        np.array([[1, -1 / 2, -1 / 2], [-1 / 2, 1, -1 / 2], [-1 / 2, -1 / 2, 1]], dtype=complex),
    ],
)
def test_vectors_from_gram_matrix_not_psd(gram):
    """Vectors from a non-positive-definite (but PSD) Gram matrix reconstruct that matrix."""
    with pytest.warns(UserWarning):
        vectors = vectors_from_gram_matrix(gram)
    mat = np.array(vectors)
    np.testing.assert_allclose(mat @ mat.conj().T, gram, atol=1e-8)


@pytest.mark.parametrize(
    "gram",
    [
        # Non-square matrix.
        (np.array([[1, 2], [4, 5], [7, 8]])),
    ],
)
def test_vectors_from_gram_matrix_invalid_input(gram):
    """Test function works as expected for an invalid input."""
    with pytest.raises(np.linalg.LinAlgError):
        vectors_from_gram_matrix(gram)
