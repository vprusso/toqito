"""Calculates the vectors associated to a Gram matrix."""

import warnings

import numpy as np


def vectors_from_gram_matrix(gram: np.ndarray) -> list[np.ndarray]:
    r"""Obtain the corresponding ensemble of states from the Gram matrix [@wikipediagram].

    The function attempts to compute the Cholesky decomposition of the given Gram matrix. If the matrix is positive
    definite, the (lower-triangular) Cholesky factor is returned. If the matrix is not positive definite, the function
    falls back to a Hermitian eigendecomposition and returns vectors whose Gram matrix equals the positive-semidefinite
    part of the input.

    In both cases the returned list of vectors ``v`` satisfies ``np.array(v) @ np.array(v).conj().T == gram`` whenever
    ``gram`` is positive semidefinite.

    Args:
        gram: A square, symmetric (Hermitian) matrix representing the Gram matrix.

    Returns:
        A list of vectors (np.ndarray) corresponding to the ensemble of states.

    Raises:
        LinAlgError: If the Gram matrix is not square.

    Examples:
        Example of a positive definite matrix:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import vectors_from_gram_matrix

        gram_matrix = np.array([[2, -1], [-1, 2]])
        vectors = vectors_from_gram_matrix(gram_matrix)

        print(vectors)
        ```

        Example of a matrix that is not positive definite:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_ops import vectors_from_gram_matrix

        gram_matrix = np.array([[1, 1], [1, 1]])
        vectors = vectors_from_gram_matrix(gram_matrix)

        print(vectors)
        ```

    """
    dim = gram.shape[0]
    if gram.shape[0] != gram.shape[1]:
        raise np.linalg.LinAlgError("The Gram matrix must be square.")

    # If the matrix is positive definite, we can use the Cholesky decomposition. The rows of the lower-triangular factor
    # are the requested vectors, since ``gram == decomp @ decomp.conj().T``.
    try:
        decomp = np.linalg.cholesky(gram)
        return [decomp[i, :] for i in range(dim)]
    # Otherwise, fall back to a Hermitian eigendecomposition. ``eigh`` (not ``eig``) is correct here because a Gram
    # matrix is Hermitian; it returns real eigenvalues and orthonormal eigenvectors.
    except np.linalg.LinAlgError:
        warnings.warn("Matrix is not positive definite. Using eigendecomposition as alternative.")
        eig_vals, eig_vecs = np.linalg.eigh(gram)
        # Clip eigenvalues that are negative (numerical noise, or a genuinely non-PSD input) to zero so the factor is
        # well-defined. The reconstructed Gram matrix is then the positive-semidefinite part of the input.
        factor = eig_vecs @ np.diag(np.sqrt(np.clip(eig_vals, 0, None)))
        return [factor[i, :] for i in range(dim)]
