"""Module for computing the commutant of a set of matrices."""

import numpy as np
from scipy.linalg import null_space


def commutant(A: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    r"""Compute an orthonormal basis (in the Hilbert-Schmidt inner product).

    For the algebra of matrices that commute with each matrix in A.


    The commutant condition is given by:

    .. math::
        A X = X A \quad \forall A \in \mathcal{A}

    This can be rewritten as:

    .. math::
        (A \otimes I - I \otimes A^T) \text{vec}(X) = 0

    where :math:`\text{vec}(X)` denotes the vectorization of :math:`X`. The null space of this
    equation provides a basis for the commutant.

    Parameters
    ==========
    A : np.ndarray or list[np.ndarray]
        A single matrix or a list of square matrices of the same dimension.

    Returns
    =======
    list[np.ndarray]
        A list of matrices forming an orthonormal basis for the commutant.

    See Also
    ========
    scipy.linalg.null_space

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """
    if isinstance(A, list):  # Handle list of matrices
        A = np.stack(A, axis=0)  # Convert to 3D array
    else:
        A = np.expand_dims(A, axis=0)  # Ensure it's a 3D array
    num_ops, dim, _ = A.shape  # Extract number of operators and dimension

    # Construct the commutant condition (A ⊗ I - I ⊗ A^T) vec(X) = 0
    comm_matrices = []
    for i in range(num_ops):
        op = np.kron(A[i], np.eye(dim)) - np.kron(np.eye(dim), A[i].T)
        comm_matrices.append(op)

    # Stack into a 2D matrix for null_space computation
    comm_matrix = np.vstack(comm_matrices) if len(comm_matrices) > 1 else comm_matrices[0]

    # Compute null space
    null_basis = null_space(comm_matrix)  # Basis vectors for commuting matrices
    print(null_basis)
    # Reshape each basis vector into a matrix of size (dim x dim)
    commutant_basis = [null_basis[:, i].reshape((dim, dim)) for i in range(null_basis.shape[1])]

    return commutant_basis
