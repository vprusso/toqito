"""Tests for is_orthonormal."""
import numpy as np
from toqito.state_props.is_mutually_orthogonal import is_mutually_orthogonal


def is_orthonormal(vectors: list[np.ndarray]) -> bool:
    """Check if the vectors are orthonormal.

    :param vectors: A list of `np.ndarray` 1-by-n vectors.
    :return: True if vectors are orthonormal; False otherwise.
    """
    return is_mutually_orthogonal(vectors) and np.allclose(
        np.dot(vectors, vectors.T), np.eye(vectors.shape[0])
    )
