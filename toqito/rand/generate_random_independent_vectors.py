"""Utilities for generating random linearly independent vectors."""

import numpy as np

from toqito.matrix_props import is_linearly_independent


def generate_random_independent_vectors(
    num_vectors: int,
    dim: int,
    is_real: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a set of random linearly independent vectors.

    This function generates a collection of random vectors that are guaranteed
    to be linearly independent.

    Examples
    =========
    >>> from toqito.rand import generate_random_independent_vectors
    >>> vecs = generate_random_independent_vectors(num_vectors=2, dim=3, seed=42)
    >>> vecs.shape
    (3, 2)

    :param num_vectors: Number of independent vectors to generate.
    :param dim: Dimension of the vector space.
    :param is_real: Whether vectors are real-valued. Defaults to True.
    :param seed: Optional random seed for reproducibility.
    :return: A (dim x num_vectors) matrix whose columns are linearly independent.

    """
    if num_vectors > dim:
        raise ValueError("Cannot have more independent vectors than the dimension of the space.")

    rng = np.random.default_rng(seed)
    max_tries = 1000

    for _ in range(max_tries):
        if is_real:
            mat = rng.standard_normal((dim, num_vectors))
        else:
            mat = rng.standard_normal((dim, num_vectors)) + 1j * rng.standard_normal(
                (dim, num_vectors)
            )

        vectors: list[np.ndarray] = [mat[:, i] for i in range(mat.shape[1])]

        if is_linearly_independent(vectors):
            return mat

    raise RuntimeError(
        "Failed to generate linearly independent vectors after multiple attempts."
    )
