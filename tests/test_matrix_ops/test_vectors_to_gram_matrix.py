"""Test vectors_to_gram_matrix."""
import numpy as np

from toqito.matrices import standard_basis
from toqito.matrix_ops import vectors_to_gram_matrix


def test_vectors_to_gram_matrix():
    """Test able to construct Gram matrix from vectors."""
    e_0, e_1 = standard_basis(2)
    trine = [
        e_0,
        1/2 * (-e_0 + np.sqrt(3) * e_1),
        -1/2 * (e_0 + np.sqrt(3) * e_1),
    ]
    gram = vectors_to_gram_matrix(trine)
    expected_gram = np.array([
        [1, -1/2, -1/2],
        [-1/2, 1, -1/2],
        [-1/2, -1/2, 1]
    ])
    assert np.allclose(gram, expected_gram)


if __name__ == "__main__":
    np.testing.run_module_suite()
