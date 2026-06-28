"""Determine if a list of matrices are POVM elements."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_povm(mat_list: list[np.ndarray], rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""Determine if a list of matrices constitute a valid set of POVMs [@wikipediapovm].

    A valid set of measurements are defined by a set of positive semidefinite operators

    \[
        \{P_a : a \in \Gamma\} \subset \text{Pos}(\mathcal{X}),
    \]

    indexed by the alphabet \(\Gamma\) of measurement outcomes satisfying the constraint that

    \[
        \sum_{a \in \Gamma} P_a = I_{\mathcal{X}}.
    \]

    Args:
        mat_list: A list of matrices.
        rtol: The relative tolerance parameter (default 1e-05).
        atol: The absolute tolerance parameter (default 1e-08).

    Returns:
        Return `True` if set of matrices constitutes a set of measurements, and `False` otherwise.

    Examples:
        Consider the following matrices:

        \[
            M_0 =
            \begin{pmatrix}
                1 & 0 \\
                0 & 0
            \end{pmatrix}
            \quad \text{and} \quad
            M_1 =
            \begin{pmatrix}
                0 & 0 \\
                0 & 1
            \end{pmatrix}.
        \]

        Our function indicates that this set of operators constitute a set of
        POVMs.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.measurement_props import is_povm

        meas_1 = np.array([[1, 0], [0, 0]])
        meas_2 = np.array([[0, 0], [0, 1]])
        meas = [meas_1, meas_2]

        print(is_povm(meas))
        ```

        We may also use the `random_povm` function from `|toqito⟩`, and can verify that a
        randomly generated set satisfies the criteria for being a POVM set.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.rand import random_povm
        from toqito.measurement_props import is_povm

        dim, num_inputs, num_outputs = 2, 2, 2
        measurements = random_povm(dim, num_inputs, num_outputs)

        print(is_povm([measurements[:, :, 0, 0], measurements[:, :, 0, 1]]))
        ```

        Alternatively, the following matrices

        \[
            M_0 =
            \begin{pmatrix}
                1 & 2 \\
                3 & 4
            \end{pmatrix}
            \quad \text{and} \quad
            M_1 =
            \begin{pmatrix}
                5 & 6 \\
                7 & 8
            \end{pmatrix},
        \]

        do not constitute a POVM set.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.measurement_props import is_povm

        non_meas_1 = np.array([[1, 2], [3, 4]])
        non_meas_2 = np.array([[5, 6], [7, 8]])
        non_meas = [non_meas_1, non_meas_2]

        print(is_povm(non_meas))
        ```

    """
    if len(mat_list) == 0:
        raise ValueError("A POVM must contain at least one measurement operator.")

    dim = mat_list[0].shape[0]
    for mat in mat_list:
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] != dim:
            raise ValueError("All POVM elements must be square matrices of the same dimension.")

    mat_sum = np.zeros((dim, dim), dtype=complex)
    for mat in mat_list:
        # Each measurement in the set must be positive semidefinite.
        if not is_positive_semidefinite(mat, rtol=rtol, atol=atol):
            return False
        mat_sum += mat
    # Summing all the measurements from the set must be equal to the identity.
    if not np.allclose(np.identity(dim), mat_sum, rtol=rtol, atol=atol):
        return False
    return True
