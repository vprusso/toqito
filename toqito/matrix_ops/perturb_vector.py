"""Perturb vectors is used to add a small random number/noise to each element of a vector."""

import numpy as np


def perturb_vectors(vectors: list[np.ndarray], eps: float = 0.1) -> list[np.ndarray]:
    """Perturb the vectors by adding a small random number to each element.

    :param vectors: List of vectors to perturb.
    :param eps: Amount by which to perturb vectors.
    :return: Resulting list of perturbed vectors by a factor of epsilon.
    """
    perturbed_vectors: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        perturbed_vectors.append(v + np.random.randn(v.shape[0]) * eps)

        # Normalize the vectors after perturbing them.
        perturbed_vectors[i] = perturbed_vectors[i] / np.linalg.norm(perturbed_vectors[i])
    return np.array(perturbed_vectors)
