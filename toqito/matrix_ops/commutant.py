"""Module for computing the commutant of a set of matrices."""

import numpy as np
from scipy.linalg import null_space


def commutant(A: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    r"""Compute an orthonormal basis for the commutant algebra.

    Given a set of matrices A, this function determines an orthonormal basis 
    (with respect to the Hilbert-Schmidt inner product) for the algebra of matrices 
    that commute with every matrix in A.


    The commutant condition is given by:

    .. math::
        A X = X A \quad \forall A \in \mathcal{A}.

    This can be rewritten as:

    .. math::
        (A \otimes I - I \otimes A^T) \text{vec}(X) = 0.

    where :math:`\text{vec}(X)` denotes the vectorization of :math:`X`. The null space of this
    equation provides a basis for the commutant.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param A: A single matrix of the form np.ndarray or a list of square matrices of the same dimension.
    :return: A list of matrices forming an orthonormal basis for the commutant.
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
    comm_matrices = [
        np.kron(A[i], np.eye(dim)) - np.kron(np.eye(dim), A[i].T)
        for i in range(num_ops)
    ]

    # Stack into a 2D matrix for null_space computation.
    comm_matrix = np.vstack(comm_matrices) if len(comm_matrices) > 1 else comm_matrices[0]

    # Compute null space.
    null_basis = null_space(comm_matrix)  # Basis vectors for commuting matrices

    # Reshape each basis vector into a matrix of size (dim x dim).
    return [null_basis[:, i].reshape((dim, dim)) for i in range(null_basis.shape[1])]

