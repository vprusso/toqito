"""Check if a quantum state is LDOI (Local Diagonal Orthogonal Invariant)."""

import numpy as np

from toqito.channels import ldot_channel


def is_ldoi(mat: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Check if a quantum state is LDOI (Local Diagonal Orthogonal Invariant).

    A matrix \(A \in \mathcal{L}(\mathcal{X} \otimes \mathcal{Y})\) is called *local diagonal
    orthogonal invariant* (LDOI) if \(\Phi_O(A) = A\), where \(\Phi_O\) is the local
    diagonal orthogonal twirl map defined as:

    \[
        \Phi_O(A) = \frac{1}{2^n} \sum_{O \in \text{DO}(\mathcal{X})} (O \otimes O) A (O \otimes O)
    \]

    where \(\text{DO}(\mathcal{X})\) is the set of diagonal matrices with entries \(\pm 1\).

    LDOI states include many important families such as Werner states, isotropic states, X-states,
    and mixtures of Dicke states. This function efficiently checks the LDOI property using the
    standard basis representation.

    Examples:
        X-states are examples of 2-qubit LDOI states:

        ```python exec="1" source="above"
        from toqito.matrix_props import is_ldoi
        import numpy as np

        # Example X-state
        x_state = np.array([[1, 0, 0, 2],
                             [0, 3, 4, 0],
                             [0, 5, 6, 0],
                             [7, 0, 0, 8]])
        print(is_ldoi(x_state))
        ```

        All diagonal states are LDOI:

        ```python exec="1" source="above"
        from toqito.matrix_props import is_ldoi
        import numpy as np

        diagonal_state = np.diag([1, 2, 3, 4])
        print(is_ldoi(diagonal_state))
        ```

        Non-LDOI states return False:

        ```python exec="1" source="above"
        from toqito.matrix_props import is_ldoi
        import numpy as np

        # Random non-LDOI state
        non_ldoi = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]])
        print(is_ldoi(non_ldoi))
        ```

    Args:
        mat: A matrix representing a quantum state.
        rtol: Relative tolerance parameter (default: 1e-05).
        atol: Absolute tolerance parameter (default: 1e-08).

    Returns:
        True if the matrix is LDOI, False otherwise.

    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    # A matrix is LDOI if Φ_O(mat) = mat
    ldot_result = ldot_channel(mat)
    return np.allclose(ldot_result, mat, rtol=rtol, atol=atol)
