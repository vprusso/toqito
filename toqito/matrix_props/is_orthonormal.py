"""Tests for is_orthonormal."""
import numpy as np

from toqito.state_props.is_mutually_orthogonal import is_mutually_orthogonal


def is_orthonormal(vectors: list[np.ndarray]) -> bool:
    r"""Check if the vectors are orthonormal.

    Examples
    ========
    The following vectors are an example of an orthonormal set of
    vectors in :math:`\mathbb{R}^3`.

    .. math::
        \begin{pmatrix}
            1 \\ 0 \\ 1
        \end{pmatrix}, \quad
        \begin{pmatrix}
            1 \\ 1 \\ 0
        \end{pmatrix}, \quad \text{and} \quad
        \begin{pmatrix}
            0 \\ 0 \\ 1
        \end{pmatrix}

    To check these are a known set of orthonormal vectors:
    >>> import numpy as np
    >>> from toqito.matrix_props import is_orthonormal
    >>> v_1 = np.array([1, 0, 0])
    >>> v_2 = np.array([0, 1, 0])
    >>> v_3 = np.array([0, 0, 1])
    >>> v = np.array([v_1, v_2, v_3])
    >>> is_orthonormal(v)
    True

    :param vectors: A list of `np.ndarray` 1-by-n vectors.
    :return: True if vectors are orthonormal; False otherwise.

    """
    return is_mutually_orthogonal(vectors) and np.allclose(
        np.dot(vectors, vectors.T), np.eye(vectors.shape[0])
    )
