"""Computes the mutual coherence for a list of 1D numpy arrays."""

import numpy as np


def mutual_coherence(vectors: list[np.ndarray]) -> float | np.floating:
    r"""Calculate the mutual coherence of a collection of input vectors.

    The mutual coherence of a collection of input vectors is defined as the maximum
    absolute value of the inner product between any two distinct vectors, divided by the
    product of their norms [@wikipediamutualcoherence]. It provides a measure of how
    similar the vectors are to each other.

    Args:
        vectors: A list of 1D numpy arrays.

    Returns:
        The mutual coherence of the collection of input vectors.

    Raises:
        ValueError: If arrays in list are not 1D.
        TypeError: If input is not a list.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props.mutual_coherence import mutual_coherence
        example_A = [np.array([1, 0]), np.array([0, 1])]
        print("Result for example_A = ", mutual_coherence(example_A))

        # An example with a larger set of vectors
        example_B = [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0])]
        print("Result for example_B = ", mutual_coherence(example_B))
        ```

    """
    # Check if the input is a valid list of 1D numpy arrays.
    if not isinstance(vectors, list):
        raise TypeError("Input must be a list of 1D numpy arrays.")
    if not all(isinstance(vec, np.ndarray) and vec.ndim == 1 for vec in vectors):
        raise ValueError("All elements in the list must be 1D numpy arrays.")

    # Convert input into a 2D numpy array. Use a complex dtype so complex-valued vectors are not truncated.
    vectors = np.column_stack(vectors).astype(complex)

    # Normalize the vectors, guarding against zero vectors (which would divide by zero).
    norms = np.linalg.norm(vectors, axis=0)
    if np.any(np.isclose(norms, 0)):
        raise ValueError("Vectors must be nonzero to compute mutual coherence.")
    vectors /= norms

    # Calculate the inner product between all pairs of columns.
    inner_products = np.abs(np.conj(vectors.T) @ vectors)

    # Set diagonal elements to zero (only respecting distinct vectors).
    np.fill_diagonal(inner_products, 0)

    return inner_products.max()
