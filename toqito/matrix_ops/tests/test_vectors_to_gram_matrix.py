"""Test vectors_to_gram_matrix."""
import numpy as np

from toqito.states import trine
from toqito.matrix_ops import vectors_to_gram_matrix


def test_vectors_to_gram_matrix():
    """Test able to construct Gram matrix from vectors."""
    gram = vectors_to_gram_matrix(trine())
    expected_gram = np.array([[1, -1 / 2, -1 / 2], [-1 / 2, 1, -1 / 2], [-1 / 2, -1 / 2, 1]])
    assert np.allclose(gram, expected_gram)
