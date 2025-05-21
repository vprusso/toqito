"""Generates a set of linearly independent random vectors."""

import numpy as np


def generate_random_independent_vectors(num_vectors: np.ndarray, dim: int, is_real: bool = True) -> list[np.ndarray]:
    r"""Generate a set of linearly independent random vectors.

    This function generates a random collection of linearly independent (possibly complex) vectors.

    Examples
    ==========
    generate_ran

    :param num_vectors: The number of vectors to generate.
    :param dim: The dimension of the vector space.
    :param is_real: Boolean denoting whether the returned vector will have all real entries or not.
                    Default is :code:`False`.
    :return: A (dim x num_vectors) matrix whose columns are the generated independent vectors.

    """
    if num_vectors > dim:
        raise ValueError("Cannot have more independent vectors than the dimension of the space.")

    # Keep generating until we get a matrix with independent columns.
    while True:
        if is_real:
            # Generate a random real matrix.
            A = np.random.randn(dim, num_vectors)
        else:
            # Generate a random complex matrix: real + i*imag.
            A = np.random.randn(dim, num_vectors) + 1j * np.random.randn(dim, num_vectors)

        # Check that the rank equals num_vectors.
        if np.linalg.matrix_rank(A) == num_vectors:
            return A
