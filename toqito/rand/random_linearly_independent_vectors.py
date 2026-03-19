"""Generates a set of random linearly independent vectors."""

import numpy as np


def random_linearly_independent_vectors(
    num_vectors: int,
    dim: int,
    is_real: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a set of linearly independent random vectors.

    This function generates a random collection of linearly independent vectors, useful for
    quantum state distinguishability scenarios where the states must be linearly independent.

    The vectors are sampled from the standard normal distribution. For complex vectors,
    both real and imaginary parts are sampled independently. The function repeatedly
    samples until a full-rank matrix is obtained (which occurs with probability 1 for
    continuous distributions).

    Args:
        num_vectors: The number of vectors to generate. Must not exceed `dim`.
        dim: The dimension of the vector space.
        is_real: If `True` (default), generate real vectors. If `False`, generate complex vectors.
        seed: A seed used to instantiate numpy's random number generator.

    Returns:
        A `(dim, num_vectors)` matrix whose columns are the generated linearly independent vectors.

    Raises:
        ValueError: If `num_vectors` exceeds `dim`.

    Examples:
        Generate 3 real linearly independent vectors in a 4-dimensional space.

        ```python exec="1" source="above" result="text"
        from toqito.rand import random_linearly_independent_vectors
        from toqito.matrix_props import is_linearly_independent

        vectors = random_linearly_independent_vectors(3, 4, seed=42)
        print(f"Shape: {vectors.shape}")
        print(f"Linearly independent: {is_linearly_independent(list(vectors.T))}")
        ```

        Generate 2 complex linearly independent vectors in a 3-dimensional space.

        ```python exec="1" source="above" result="text"
        from toqito.rand import random_linearly_independent_vectors
        from toqito.matrix_props import is_linearly_independent

        vectors = random_linearly_independent_vectors(2, 3, is_real=False, seed=7)
        print(f"Shape: {vectors.shape}")
        print(f"Linearly independent: {is_linearly_independent(list(vectors.T))}")
        ```

    """
    if num_vectors > dim:
        raise ValueError("Cannot have more independent vectors than the dimension of the space.")

    rng = np.random.default_rng(seed)

    while True:
        if is_real:
            mat = rng.standard_normal((dim, num_vectors))
        else:
            mat = rng.standard_normal((dim, num_vectors)) + 1j * rng.standard_normal((dim, num_vectors))

        if np.linalg.matrix_rank(mat) == num_vectors:
            return mat
