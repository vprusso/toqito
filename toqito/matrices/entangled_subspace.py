"""Produces a basis of an r-entangled subspace."""
import numpy as np
from scipy import sparse


def entangled_subspace(
    dim: int, local_dim: int | list[int], r: int = 1
) -> sparse.csr_matrix:
    """Produce a basis of an r-entangled subspace.

    A subspace is called r-entangled if every non-zero vector within it has Schmidt
    rank at least r+1. This function generates a basis for such a subspace.

    Examples
    ==========
    >>> from toqito.matrices import entangled_subspace
    >>> entangled_subspace(dim=2, local_dim=3)  # doctest: +NORMALIZE_WHITESPACE
    <Compressed Sparse Row sparse matrix of dtype 'complex128'
            with 5 stored elements and shape (9, 2)>

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim: The dimension of the subspace.
    :param local_dim: The dimensions of the local systems. If a single integer
                     is provided, it is assumed that both local dimensions are equal.
    :param r: The entanglement parameter. The resulting subspace will be r-entangled.
             Default: 1 (i.e., the resulting subspace is just entangled).
    :return: Matrix whose columns form a basis of an r-entangled subspace of the requested
            dimension.

    """
    # Convert local_dim to a list if it was provided as a scalar.
    if isinstance(local_dim, int):
        local_dim = [local_dim, local_dim]
    elif len(local_dim) == 1:
        local_dim = [local_dim[0], local_dim[0]]

    # Entangled subspaces only exist in some dimensions.
    if dim > (local_dim[0] - r) * (local_dim[1] - r):
        raise ValueError("No r-entangled subspace of this dimension exists with the given parameters.")

    m = min(local_dim)
    prod_local_dim = local_dim[0] * local_dim[1]

    # Create a Vandermonde matrix for basis vectors.
    # In MATLAB: V = fliplr(vander(1:m)).
    vander_matrix = np.vander(np.arange(1, m + 1), increasing=True)

    # Initialize the output matrix.
    e_matrix = sparse.csr_matrix((prod_local_dim, dim), dtype=complex)

    # Now construct E matrix.
    ct = 0
    for k in range(1, m - r + 1):  # Loop through columns of the Vandermonde matrix.
        for j in range(r + 1 - local_dim[1], local_dim[0] - r):  # Loop through relevant diagonals.
            # Calculate length of this diagonal.
            if j < 0:
                ell = min(local_dim[1] + j, local_dim[0])
            else:
                ell = min(local_dim[1], local_dim[0] - j)

            if k <= ell - r:
                # Get the Vandermonde column elements for this diagonal.
                d_vec = vander_matrix[:ell, k-1]

                # Create a sparse matrix with these elements on the diagonal.
                t_matrix = sparse.csr_matrix((local_dim[1], local_dim[0]), dtype=complex)

                # Construct the diagonal matrix.
                if j < 0:
                    # Diagonal below main diagonal.
                    diag_indices = np.arange(ell)
                    row_indices = diag_indices - j
                    col_indices = diag_indices
                else:
                    # Diagonal above main diagonal.
                    diag_indices = np.arange(ell)
                    row_indices = diag_indices
                    col_indices = diag_indices + j

                t_matrix[row_indices, col_indices] = d_vec

                # Reshape to a column vector.
                e_matrix[:, ct] = sparse.csr_matrix(t_matrix.reshape((prod_local_dim, 1)))

                ct += 1
                if ct >= dim:  # We've found enough columns.
                    return e_matrix

    return e_matrix
