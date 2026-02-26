"""Calculates the Gram matrix from a list of vectors."""

import numpy as np


def vectors_to_gram_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    r"""Construct the Gram matrix from a list of vectors or density matrices [@WikiGram].

    The Gram matrix is a matrix of inner products. This function automatically detects whether the inputs
    are vectors (pure states) or density matrices (mixed states) and computes the appropriate Gram matrix.

    For vectors |ψᵢ⟩: G[i, j] = ⟨ψᵢ|ψⱼ⟩
    For density matrices ρᵢ: G[i, j] = Tr(ρᵢ ρⱼ)

    Examples:
    Example with real vectors:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_ops import vectors_to_gram_matrix

    vectors = [np.array([1, 2]), np.array([3, 4])]
    gram_matrix = vectors_to_gram_matrix(vectors)

    print(gram_matrix)
    ```

    Example with complex vectors:

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_ops import vectors_to_gram_matrix

    vectors = [np.array([1+1j, 2+2j]), np.array([3+3j, 4+4j])]
    gram_matrix = vectors_to_gram_matrix(vectors)

    print(gram_matrix)
    ```

    Example with density matrices (mixed states):

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrix_ops import vectors_to_gram_matrix

    # Two mixed states
    rho1 = 0.7 * np.array([[1., 0.], [0., 0.]]) + 0.3 * np.eye(2) / 2
    rho2 = 0.7 * np.array([[0., 0.], [0., 1.]]) + 0.3 * np.eye(2) / 2
    states = [rho1, rho2]

    gram_matrix = vectors_to_gram_matrix(states)
    print(gram_matrix)
    ```

    Raises:
        ValueError: If the vectors are not all of the same shape.

    Args:
        vectors: A list of vectors (1D/column arrays for pure states) or density matrices (2D arrays for mixed states).

    Returns:
        The Gram matrix with entries G[i,j] = ⟨vᵢ|vⱼ⟩ for vectors or Tr(ρᵢρⱼ) for density matrices.

    """
    # Check that all vectors are of the same shape
    if not all(v.shape == vectors[0].shape for v in vectors):
        raise ValueError("All vectors must be of the same shape.")

    first_input = vectors[0]

    # Check if inputs are vectors (1D or column vectors) or density matrices (2D with d > 1)
    if first_input.ndim == 1 or (first_input.ndim == 2 and first_input.shape[1] == 1):
        # Pure states: use standard Gram matrix construction
        # Stack vectors into a matrix
        stacked_vectors = np.column_stack(vectors)
        # Compute Gram matrix using vectorized operations
        return np.dot(stacked_vectors.conj().T, stacked_vectors)
    else:
        # Mixed states: compute Tr(ρᵢ ρⱼ)
        n = len(vectors)
        gram = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                gram[i, j] = np.trace(vectors[i] @ vectors[j])
        return gram
