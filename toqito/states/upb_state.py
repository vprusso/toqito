"""Bound entangled state on the orthogonal complement of an unextendible product basis."""

import numpy as np


def upb_state(vecs: list[np.ndarray]) -> np.ndarray:
    r"""Produce the bound entangled state associated with an unextendible product basis [@bennett1999unextendible].

    Given an unextendible product basis (UPB) \(\{|\psi_i\rangle\}_{i=1}^k\) of
    \(\mathbb{C}^{d_A} \otimes \mathbb{C}^{d_B}\), the state

    \[
        \rho = \frac{1}{d_A d_B - k} \left( \mathbb{I} - \sum_{i=1}^k |\psi_i\rangle\langle\psi_i| \right)
    \]

    is the normalized projector onto the orthogonal complement of the basis. Each
    \(|\psi_i\rangle\langle\psi_i|\) is a product projector, so \(\rho\) is invariant under partial
    transposition and hence PPT, while its range contains no product vector, so it is entangled by the
    range criterion. It is therefore bound entangled.

    The function itself only requires the vectors to be orthonormal. Whether they form a UPB, and hence
    whether the output is entangled, can be checked with
    `toqito.state_props.is_unextendible_product_basis`.

    Args:
        vecs: Orthonormal column vectors forming the product basis.

    Returns:
        The normalized projector onto the orthogonal complement of the span of `vecs`.

    Raises:
        ValueError: If the vectors are not orthonormal or span the whole space.

    Examples:
        The Tile UPB on \(\mathbb{C}^3 \otimes \mathbb{C}^3\) gives a 9-by-9 state of rank 4 that is PPT.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import tile, upb_state
        from toqito.state_props import is_ppt
        rho = upb_state([tile(i) for i in range(5)])
        print(np.linalg.matrix_rank(rho), is_ppt(rho, dim=[3, 3]))
        ```

    """
    mat = np.column_stack([np.asarray(vec).reshape(-1) for vec in vecs])
    dim, num_vecs = mat.shape
    if not np.allclose(mat.conj().T @ mat, np.identity(num_vecs)):
        raise ValueError("The vectors must be orthonormal.")
    if num_vecs >= dim:
        raise ValueError("The vectors span the whole space, so the orthogonal complement is empty.")
    return (np.identity(dim) - mat @ mat.conj().T) / (dim - num_vecs)
