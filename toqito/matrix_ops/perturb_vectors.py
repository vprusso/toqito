"""Perturb vectors is used to add a small random number to each element of a vector.

A random value is added sampled from a normal distribution scaled by `eps`.
"""

import numpy as np


def perturb_vectors(vectors: list[np.ndarray], eps: float = 0.1) -> list[np.ndarray]:
    """Perturb the vectors by adding a small random number to each element.

    :param vectors: List of vectors to perturb.
    :param eps: Amount by which to perturb vectors.
    :return: Resulting list of perturbed vectors by a factor of epsilon.

    Example:
    ==========

    .. jupyter-execute::

     import numpy as np
     from toqito.matrix_ops import perturb_vectors

     vectors = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

     perturb_vectors(vectors, eps=0.1)

    """
    perturbed_vectors: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        if eps == 0:
            perturbed_vectors.append(v)
        else:
            perturbed_vectors.append(v + np.random.randn(v.shape[0]) * eps)

            # Normalize the vectors after perturbing them.
            perturbed_vectors[i] = perturbed_vectors[i] / np.linalg.norm(perturbed_vectors[i])
    return np.array(perturbed_vectors)
