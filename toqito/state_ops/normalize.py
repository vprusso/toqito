"""Normalize quantum state vectors."""

from typing import Sequence

import numpy as np


def normalize(vector: Sequence[complex] | np.ndarray, *, tol: float = 1e-8) -> np.ndarray:
    r"""Return a normalized copy of the input state vector.

    The input may be a one-dimensional array or a column/row vector. A zero vector raises `ValueError`.

    Examples:
        ```python exec="1" source="above"
        import numpy as np
        from toqito.state_ops import normalize

        v = np.array([1, 1], dtype=np.complex128)
        print(normalize(v))
        ```


    Raises:
        ValueError: If the input is not vector-shaped or has vanishing norm.

    Args:
        vector: State vector expressed as a 1D array or column/row vector.
        tol: Numerical tolerance used to detect zero-norm inputs.

    Returns:
        Normalized vector as a 1D NumPy array.

    """
    array = np.asarray(vector, dtype=np.complex128)

    if array.ndim == 1:
        flattened = array
    elif array.ndim == 2 and 1 in array.shape:
        flattened = array.reshape(-1)
    else:
        raise ValueError("normalize expects a vector or column/row matrix input.")

    norm = np.linalg.norm(flattened)
    if np.isclose(norm, 0.0, atol=tol):
        raise ValueError("Cannot normalize a zero vector.")

    return flattened / norm
