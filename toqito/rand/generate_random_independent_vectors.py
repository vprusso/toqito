"""Utilities for generating random linearly independent vectors."""

import numpy as np

from toqito.matrix_props import is_linearly_independent


def generate_random_independent_vectors(
    num_vectors: int,
    dim: int,
    is_real: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate random linearly independent vectors.

    Generates a collection of random vectors that are guaranteed
    to be linearly independent.

    Examples
    ==========
    Demonstrate generation of random independent vectors.

    .. jupyter-execute::

        from toqito.rand import generate_random_independent_vectors
        vecs = generate_random_independent_vectors(num_vectors=2, dim=3, seed=42)
        vecs.shape

    References
    ==========
    .. footbibliography::


    :param num_vectors: Number of independent vectors to generate.
    :param dim: Dimension of the vector space.
    :param is_real: Whether vectors are real-valued (default is True).
    :param seed: Random seed for reproducibility.
    :raises ValueError: If ``num_vectors`` is greater than ``dim``.
    :raises RuntimeError: If independent vectors cannot be generated.
    :return: A ``(dim x num_vectors)`` matrix with independent columns.

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
