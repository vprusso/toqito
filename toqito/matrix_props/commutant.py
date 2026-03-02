"""Module for computing the commutant of a set of matrices."""

import numpy as np
from scipy.linalg import null_space


def commutant(A: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    r"""Compute an orthonormal basis for the commutant algebra [@planetmathcommutant].

    Given a matrix \(A\) or a set of matrices \(\mathcal{A} = \{A_1, A_2, \dots\}\),
    this function determines an orthonormal basis (with respect to the Hilbert-Schmidt inner product)
    for the algebra of matrices that commute with every matrix in \(\mathcal{A}\).

    The commutant of a single matrix \(A \in \mathbb{C}^{n \times n}\) consists of all matrices
    \(X \in \mathbb{C}^{n \times n}\) satisfying:

    \[
        A X = X A.
    \]

    More generally, for a set of matrices \(\mathcal{A} = \{A_1, A_2, \dots\}\), the commutant
    consists of all matrices \(X\) satisfying:

    \[
        A_i X = X A_i \quad \forall A_i \in \mathcal{A}.
    \]

    This condition can be rewritten in vectorized form as:

    \[
        (A_i \otimes I - I \otimes A_i^T) \text{vec}(X) = 0, \quad \forall A_i \in \mathcal{A}.
    \]

    where \(\text{vec}(X)\) denotes the column-wise vectorization of \(X\).
    The null space of this equation provides a basis for the commutant.

    This implementation is based on [@qetlablink].

    Examples:
        Consider the following set of matrices:

        \[
            A_1 = \begin{pmatrix}
                    1 & 0 \\
                    0 & -1
                \end{pmatrix}, \quad
            A_2 = \begin{pmatrix}
                    0 & 1 \\
                    1 & 0
                \end{pmatrix}
        \]

        The commutant consists of matrices that commute with both \(A_1\) and \(A_2\).

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import commutant

        A1 = np.array([[1, 0], [0, -1]])
        A2 = np.array([[0, 1], [1, 0]])

        basis = commutant([A1, A2])

        print(basis)
        ```


        Now, consider a single matrix:

        \[
            A = \begin{pmatrix}
                    1 & 1 \\
                    0 & 1
                \end{pmatrix}
        \]

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_props import commutant

        A = np.array([[1, 1], [0, 1]])

        basis = commutant(A)

        for i, basis_ in enumerate(basis):
           print(f"basis{ i} :\n{basis_} \n")
        ```

    Args:
        A: A single matrix of the form np.ndarray or a list of square matrices of the same dimension.

    Returns:
        A list of matrices forming an orthonormal basis for the commutant.

    """
    # Handle list of matrices.
    if isinstance(A, list):
        # Convert to 3D array.
        A = np.stack(A, axis=0)
    else:
        # Ensure it's a 3D array.
        A = np.expand_dims(A, axis=0)
    # Extract number of operators and dimension.
    num_ops, dim, _ = A.shape

    # Construct the commutant condition (A ⊗ I - I ⊗ A^T) vec(X) = 0.
    comm_matrices = [np.kron(A[i], np.eye(dim)) - np.kron(np.eye(dim), A[i].T) for i in range(num_ops)]

    # Stack into a 2D matrix for null_space computation.
    comm_matrix = np.vstack(comm_matrices) if len(comm_matrices) > 1 else comm_matrices[0]

    # Compute null space.
    null_basis = null_space(comm_matrix)  # Basis vectors for commuting matrices

    # Reshape each basis vector into a matrix of size (dim x dim).
    return [null_basis[:, i].reshape((dim, dim)) for i in range(null_basis.shape[1])]
