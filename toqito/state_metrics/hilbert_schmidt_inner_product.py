"""Hilbert-Schmidt Inner Product refers to the inner product between two Hilbert-Schmidt operators."""

import numpy as np


def hilbert_schmidt_inner_product(a_mat: np.ndarray, b_mat: np.ndarray) -> complex:
    r"""Compute the Hilbert-Schmidt inner product between two matrices [@WikiHilbSchOp].

    The Hilbert-Schmidt inner product between `a_mat` and `b_mat` is defined as

    \[
        HS = (A|B) = Tr[A^\dagger B]
    \]

    where \(|B\rangle = \text{vec}(B)\) and \(\langle A|\) is the dual vector to \(|A \rangle\).

    Note: This function has been adapted from [@Rigetti_2022_Forest].

    Examples:

    One may consider taking the Hilbert-Schmidt distance between two Hadamard matrices.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.matrices import hadamard
    from toqito.state_metrics import hilbert_schmidt_inner_product
    
    h = hadamard(1)
    
    print(np.around(hilbert_schmidt_inner_product(h, h), decimals=2))
    ```

    Args:
        a_mat: An input matrix provided as a numpy array.
        b_mat: An input matrix provided as a numpy array.

    Returns:
        The Hilbert-Schmidt inner product between `a_mat` and `b_mat`.

    """
    return np.trace(a_mat.conj().T @ b_mat)
