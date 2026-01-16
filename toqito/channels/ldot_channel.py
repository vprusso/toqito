"""Local diagonal orthogonal twirl channel."""

import itertools

import numpy as np


def ldot_channel(mat: np.ndarray, efficient: bool = True) -> np.ndarray:
    r"""Apply the local diagonal orthogonal twirl (LDOT) channel to a matrix.

    The LDOT channel projects a matrix onto the subspace of local diagonal
    orthogonal invariant (LDOI) matrices. It is defined as:

    .. math::
        \Phi_O(A) = \frac{1}{2^n} \sum_{O \in \text{DO}(\mathcal{X})} (O \otimes O) A (O \otimes O)

    where :math:`\text{DO}(\mathcal{X})` is the set of :math:`n \times n` diagonal matrices with
    diagonal entries equal to :math:`\pm 1`.

    The LDOT channel has the following properties:

    - It is a quantum channel (completely positive and trace-preserving)
    - It is self-adjoint: :math:`\Phi_O^* = \Phi_O`
    - It preserves PPT and separability
    - It is an orthogonal projection onto the LDOI subspace

    The efficient implementation works directly in the computational basis, zeroing out the entries
    that average to zero under the twirl. This keeps the complexity polynomial in :math:`n` instead
    of the exponential :math:`O(2^n)` for the brute-force approach.

    Examples
    ==========

    Apply LDOT channel to project an arbitrary matrix onto LDOI subspace:

    .. jupyter-execute::

        from toqito.channels import ldot_channel
        import numpy as np

        # Arbitrary 2-qubit matrix
        mat = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
        ldoi_projection = ldot_channel(mat)
        print(ldoi_projection)

    The LDOT channel is idempotent (applying it twice gives the same result):

    .. jupyter-execute::

        from toqito.channels import ldot_channel
        import numpy as np

        mat = np.random.rand(4, 4)
        once = ldot_channel(mat)
        twice = ldot_channel(once)
        np.allclose(once, twice)

    References
    ==========
    .. footbibliography::

    :param mat: A square matrix of dimension :math:`n^2 \times n^2` representing a bipartite
                operator on :math:`\mathcal{X} \otimes \mathcal{Y}` where
                :math:`\mathcal{X} = \mathcal{Y} = \mathbb{C}^n`.
    :param efficient: If True, use the efficient O(n²) standard basis implementation. If False,
                      use the brute-force O(2ⁿ) implementation (useful for verification).
    :return: The LDOI projection of the input matrix.

    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square.")

    if efficient:
        return _ldot_channel_standard_basis(mat)
    return _ldot_channel_brute_force(mat)


def _ldot_channel_standard_basis(mat: np.ndarray) -> np.ndarray:
    """Efficient implementation by zeroing forbidden entries in the computational basis."""
    dim_sq = mat.shape[0]
    dim = int(np.sqrt(dim_sq))
    if dim * dim != dim_sq:
        raise ValueError("Input dimension must be a perfect square.")

    result = np.zeros_like(mat, dtype=np.complex128)
    for i in range(dim):
        for j in range(dim):
            row = i * dim + j
            for k in range(dim):
<<<<<<< HEAD
                for target in range(dim):
                    col = k * dim + target
                    if (
                        (i == j and k == target)
                        or (i == k and j == target)
                        or (i == target and j == k)
                    ):
=======
                for col_subsystem_idx in range(dim):
                    col = k * dim + col_subsystem_idx
                    same_diag = i == j and k == col_subsystem_idx
                    same_indices = i == k and j == col_subsystem_idx
                    swapped_indices = i == col_subsystem_idx and j == k
                    if both_on_diagonal or same_indices or swapped_indices:
>>>>>>> 8cb272c06dd3fdc06802e9112dbf1dc4cb15291e
                        result[row, col] = mat[row, col]

    return result


def _ldot_channel_brute_force(mat: np.ndarray) -> np.ndarray:
    """Brute-force O(2ⁿ) implementation by averaging over all diagonal ±1 matrices.

    This implementation is provided for verification purposes and to match the mathematical
    definition exactly. For practical use, the efficient implementation should be preferred.
    """
    dim = int(np.sqrt(mat.shape[0]))

    # Generate all possible diagonal ±1 matrices
    result = np.zeros((dim**2, dim**2), dtype=np.complex128)

    for diag_entries in itertools.product([-1, 1], repeat=dim):
        diag_mat = np.diag(diag_entries)
        conjugation = np.kron(diag_mat, diag_mat)
        result += conjugation @ mat @ conjugation

    return result / (2**dim)
