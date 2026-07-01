"""Weighted Gram matrix of a set of isometry channels."""

import numpy as np

from toqito.matrix_props import is_isometry


def channel_gram_matrix(
    isometries: list[np.ndarray],
    sigma: np.ndarray | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    r"""Compute the \(\sigma\)-weighted Gram matrix of a set of isometry channels.

    For isometry channels \(\Xi_i(X) = V_i X V_i^*\) with \(V_i \in \text{U}(\mathcal{X}, \mathcal{Y})\)
    (that is, \(V_i^* V_i = \mathbb{I}_{\mathcal{X}}\)) and a density operator \(\sigma \in
    \text{D}(\mathcal{X})\), the \(\sigma\)-weighted Gram matrix is

    \[
        G(\sigma)_{i,j} = \text{Tr}(V_j^* V_i \sigma).
    \]

    This equals \(\langle \varphi_i | \varphi_j \rangle\) for the output states \(|\varphi_i\rangle =
    (V_i \otimes \mathbb{I})|\psi\rangle\) obtained from any purification \(|\psi\rangle\) with
    \(\text{Tr}_{\mathcal{Z}}(|\psi\rangle\langle\psi|) = \sigma\). The choice \(\sigma =
    \mathbb{I}_{\mathcal{X}} / \dim(\mathcal{X})\) (the default) corresponds to the maximally
    entangled input, for which \(G(\sigma)\) is the Gram matrix of the pure Choi states. Unlike the
    Hilbert-Schmidt inner product \(\langle J(\Xi_i), J(\Xi_j) \rangle = |\text{Tr}(V_i^* V_j)|^2\) of
    the Choi matrices, this weighted Gram matrix is linear in \(V_j^* V_i\) and carries no dimension
    factors or squaring of overlaps.

    Args:
        isometries: A list of isometries \(V_i\), each a `(dim_out, dim_in)` array satisfying
            \(V_i^* V_i = \mathbb{I}\). A unitary channel is provided by its unitary. All isometries
            must share the same input and output dimensions.
        sigma: A `(dim_in, dim_in)` density operator. If omitted, the maximally mixed state
            \(\mathbb{I} / \dim_{\text{in}}\) is used.
        tol: Tolerance for the isometry check \(V_i^* V_i = \mathbb{I}\). Default is `1e-8`.

    Returns:
        The `(n, n)` complex weighted Gram matrix \(G(\sigma)\).

    Raises:
        ValueError: If fewer than one isometry is provided.
        ValueError: If the isometries have mismatched dimensions.
        ValueError: If any provided operator is not an isometry.
        ValueError: If `sigma` does not match the input dimension.

    Examples:
        With the default maximally entangled input, the weighted Gram matrix of the Pauli unitaries
        is the identity, since the Pauli operators are orthogonal with respect to
        \(\langle A, B \rangle = \text{Tr}(A^* B) / \dim\):

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_ops import channel_gram_matrix
        paulis = [
            np.eye(2),
            np.array([[0, 1], [1, 0]]),
            np.array([[0, -1j], [1j, 0]]),
            np.array([[1, 0], [0, -1]]),
        ]
        print(np.round(channel_gram_matrix(paulis).real, 6))
        ```

    """
    if len(isometries) < 1:
        raise ValueError("At least one isometry is required.")

    dim_out, dim_in = isometries[0].shape
    for mat in isometries:
        if mat.shape != (dim_out, dim_in):
            raise ValueError("All isometries must share the same input and output dimensions.")
        if not is_isometry(mat, atol=tol):
            raise ValueError("Each operator must be an isometry (V^* V = I).")

    if sigma is None:
        sigma = np.eye(dim_in) / dim_in
    elif sigma.shape != (dim_in, dim_in):
        raise ValueError("sigma must be a (dim_in, dim_in) density operator.")

    n = len(isometries)
    gram = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            gram[i, j] = np.trace(isometries[j].conj().T @ isometries[i] @ sigma)
    return gram
