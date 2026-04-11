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

    E = np.zeros((prod_dim, dim))

    ct = 0
    for k in range(m - r):
        for j in range(r + 1 - d_b, d_a - r):
            # Length of the j-th diagonal of a (d_b x d_a) matrix.
            if j >= 0:
                diag_len = min(d_b, d_a - j)
            else:
                diag_len = min(d_b + j, d_a)

            if k < diag_len - r:
                D = V[:diag_len, k]

                # Place D on the j-th diagonal of a (d_b x d_a) matrix.
                T = np.zeros((d_b, d_a))
                if j >= 0:
                    for i in range(diag_len):
                        T[i, i + j] = D[i]
                else:
                    for i in range(diag_len):
                        T[i - j, i] = D[i]

                E[:, ct] = T.reshape(prod_dim)

                ct += 1
                if ct >= dim:
                    # Orthonormalize via QR.
                    Q, _ = qr(E, mode='economic')
                    return Q

    # Orthonormalize via QR.
    Q, _ = qr(E[:, :ct], mode='economic')
    return Q
