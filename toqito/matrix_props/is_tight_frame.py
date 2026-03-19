"""Check if a set of vectors constitutes a tight frame."""

import numpy as np


def is_tight_frame(vectors: list[np.ndarray], tol: float = 1e-8) -> bool:
    r"""Check if a set of vectors constitutes a tight frame.

    A set of vectors \(\{v_1, v_2, \ldots, v_n\}\) in \(\mathbb{C}^d\) forms a
    *tight frame* if there exists a constant \(A > 0\) such that

    \[
        \sum_{i=1}^{n} v_i v_i^* = A \cdot I_d
    \]

    where \(I_d\) is the \(d \times d\) identity matrix. The constant \(A\) is
    called the *frame bound* and equals \(n/d\) when the vectors all have unit norm.

    Args:
        vectors: A list of 1D numpy arrays (vectors) of the same dimension.
        tol: Numerical tolerance for the comparison. Default 1e-8.

    Returns:
        `True` if the vectors form a tight frame; `False` otherwise.

    Raises:
        ValueError: If the list of vectors is empty or vectors have inconsistent dimensions.

    Examples:
        The standard basis vectors in \(\mathbb{R}^2\) form a tight frame.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_tight_frame

        e0 = np.array([1, 0])
        e1 = np.array([0, 1])
        print(is_tight_frame([e0, e1]))
        ```

        The Mercedes-Benz (trine) vectors form a tight frame in \(\mathbb{R}^2\).

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.matrix_props import is_tight_frame

        v0 = np.array([0, 1])
        v1 = np.array([np.sqrt(3)/2, -1/2])
        v2 = np.array([-np.sqrt(3)/2, -1/2])
        print(is_tight_frame([v0, v1, v2]))
        ```

    """
    if not vectors:
        raise ValueError("At least one vector must be provided.")

    dim = vectors[0].shape[0]
    if any(v.shape[0] != dim for v in vectors):
        raise ValueError("All vectors must have the same dimension.")

    # Compute the frame operator: S = sum_i v_i v_i^*
    frame_op = np.zeros((dim, dim), dtype=complex)
    for v in vectors:
        v_col = v.reshape(-1, 1)
        frame_op += v_col @ v_col.conj().T

    # A tight frame requires S = A * I for some scalar A > 0.
    # The frame bound A is trace(S) / d.
    frame_bound = np.real(np.trace(frame_op)) / dim

    if frame_bound < tol:
        return False

    return np.allclose(frame_op, frame_bound * np.eye(dim), atol=tol)
