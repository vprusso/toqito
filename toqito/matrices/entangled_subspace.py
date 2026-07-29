"""Generate a basis for an entangled subspace."""

import numpy as np
from scipy.linalg import qr


def entangled_subspace(
    dim: int,
    local_dim: int | list[int],
    r: int = 1,
) -> np.ndarray:
    r"""Generate a basis for an r-entangled subspace.

    Constructs an orthonormal basis for a subspace of
    \(\mathbb{C}^{d_A} \otimes \mathbb{C}^{d_B}\) in which every vector has
    Schmidt rank at least \(r + 1\). Such subspaces are useful for constructing
    entanglement witnesses and bound entangled states.

    An r-entangled subspace of the requested dimension exists if and only if

    \[
        \text{dim} \leq (d_A - r)(d_B - r).
    \]

    The construction uses Vandermonde matrices placed along diagonals, following
    the method from QETLAB [@qetlablink].

    Args:
        dim: The dimension of the entangled subspace (number of basis vectors).
        local_dim: Local dimensions of the subsystems. If an integer, both
            subsystems have the same dimension. If a list `[d_A, d_B]`,
            the subsystems have dimensions `d_A` and `d_B`.
        r: Entanglement parameter (default 1). Every vector in the subspace
            will have Schmidt rank at least `r + 1`.

    Returns:
        A `(d_A * d_B, dim)` matrix whose columns form an orthonormal basis for
        the r-entangled subspace.

    Raises:
        ValueError: If no r-entangled subspace of the requested dimension exists.

    Examples:
        Generate a 1-entangled subspace of dimension 1 in a 3x3 system.

        ```python exec="1" source="above" result="text"
        from toqito.matrices import entangled_subspace

        E = entangled_subspace(1, 3)
        print(f"Shape: {E.shape}")
        print(f"Orthonormal: {abs(E[:, 0] @ E[:, 0].conj() - 1) < 1e-10}")
        ```

        Generate a 1-entangled subspace of dimension 4 in a 4x4 system.

        ```python exec="1" source="above" result="text"
        from toqito.matrices import entangled_subspace

        E = entangled_subspace(4, 4)
        print(f"Shape: {E.shape}")
        ```

    """
    if isinstance(local_dim, int):
        local_dim = [local_dim, local_dim]

    d_a, d_b = local_dim[0], local_dim[1]
    max_dim = (d_a - r) * (d_b - r)

    if dim > max_dim:
        raise ValueError(
            f"No {r}-entangled subspace of dimension {dim} exists. "
            f"Maximum dimension is ({d_a} - {r}) * ({d_b} - {r}) = {max_dim}."
        )

    m = min(d_a, d_b)
    prod_dim = d_a * d_b

    # Vandermonde matrix: V[i, k] = (i+1)^k
    V = np.fliplr(np.vander(np.arange(1, m + 1)))

    k_grid, j_grid = np.meshgrid(
        np.arange(m - r),
        np.arange(r + 1 - d_b, d_a - r),
        indexing="ij",
    )
    k_values = k_grid.ravel()
    j_values = j_grid.ravel()

    diag_lens = np.where(
        j_values >= 0,
        np.minimum(d_b, d_a - j_values),
        np.minimum(d_b + j_values, d_a),
    )
    valid = k_values < diag_lens - r
    k_values = k_values[valid][:dim]
    j_values = j_values[valid][:dim]
    diag_lens = diag_lens[valid][:dim]

    diag_positions = np.arange(m)
    position_grid = np.broadcast_to(diag_positions, (len(k_values), m))
    entry_mask = position_grid < diag_lens[:, np.newaxis]
    row_offsets = np.where(j_values >= 0, j_values, -j_values * d_a)
    row_indices = position_grid * (d_a + 1) + row_offsets[:, np.newaxis]
    column_indices = np.broadcast_to(
        np.arange(len(k_values))[:, np.newaxis],
        row_indices.shape,
    )
    values = V[position_grid, k_values[:, np.newaxis]]

    E = np.zeros((prod_dim, len(k_values)))
    E[row_indices[entry_mask], column_indices[entry_mask]] = values[entry_mask]

    # Orthonormalize via QR.
    Q, _ = qr(E, mode="economic")
    return Q
