"""Generate a random set of linearly independent vectors."""

import numpy as np

from toqito.matrix_props import is_linearly_independent


def random_linearly_independent_vectors(
    num_vectors: int,
    dim: int,
    is_real: bool = True,
    seed: int | None = None,
    max_tries: int = 1_000,
) -> np.ndarray:
    r"""Generate random linearly independent vectors.

    This function repeatedly samples random vectors (real or complex) and checks
    their linear independence until a valid set is found or a maximum number of
    attempts is reached.

    Random vectors are drawn from a standard normal distribution. Independence is
    verified using :func:`toqito.matrix_props.is_linearly_independent`. If a valid
    set of vectors cannot be generated within ``max_tries`` attempts, a
    ``RuntimeError`` is raised.

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
        raise ValueError(f"Number of vectors {num_vectors} cannot be greater than dimension {dim}")

    rng = np.random.default_rng(seed)

    for _ in range(max_tries):
        if is_real:
            mat = rng.standard_normal((dim, num_vectors))
        else:
            mat = rng.standard_normal((dim, num_vectors)) + 1j * rng.standard_normal((dim, num_vectors))

        vectors: list[np.ndarray] = [mat[:, i] for i in range(mat.shape[1])]

        if is_linearly_independent(vectors):
            return mat

    raise RuntimeError(f"Failed to generate linearly independent vectors after {max_tries} attempts.")
