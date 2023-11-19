"""Test vectors_from_gram_matrix."""
import numpy as np
import pytest

from toqito.matrix_ops import vectors_from_gram_matrix


def test_vectors_from_gram_matrix():
    """Test able to extract vectors from Gram matrix."""
    gram = np.identity(4)
    vectors = vectors_from_gram_matrix(gram)

    assert np.allclose(vectors[0], np.array([1, 0, 0, 0]))
    assert np.allclose(vectors[1], np.array([0, 1, 0, 0]))
    assert np.allclose(vectors[2], np.array([0, 0, 1, 0]))
    assert np.allclose(vectors[3], np.array([0, 0, 0, 1]))


def test_vectors_from_gram_matrix_not_psd():
    """Test when matrix is not positive semidefinite."""
    gram = np.array([[1, -1 / 2, -1 / 2], [-1 / 2, 1, -1 / 2], [-1 / 2, -1 / 2, 1]], dtype=complex)

    vectors = vectors_from_gram_matrix(gram)

    assert np.allclose(vectors[0][0], 1)
    assert np.allclose(vectors[1][0], -1 / 2)
    assert np.allclose(vectors[2][0], -1 / 2)


@pytest.mark.parametrize("gram", [
    # Non-square matrix.
    (np.array([[1, 2], [4, 5], [7, 8]])),
    # Non-symmetric matrix.
    (np.array([[1, 2], [3, 4]])),
])
def test_vectors_from_gram_matrix_invalid_input(gram):
    with np.testing.assert_raises(np.linalg.LinAlgError):
        vectors_from_gram_matrix(gram)