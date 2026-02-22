"""Produces the generalized Gell-Mann operator matrices."""

import numpy as np


def gen_gell_mann(ind_1: int, ind_2: int, dim: int) -> np.ndarray:
    r"""Produce a generalized Gell-Mann operator [@WikiGellMann].

    Construct a `dim`-by-`dim` Hermitian operator. These matrices
    span the entire space of `dim`-by-`dim` matrices as
    `ind_1` and `ind_2` range from 0 to `dim-1`, inclusive,
    and they generalize the Pauli operators when `dim = 2` and the
    Gell-Mann operators when `dim = 3`.

    Examples:

    The generalized Gell-Mann matrix for `ind_1 = 0`, `ind_2 = 1`
    and `dim = 2` is given as

    \[
        G_{0, 1, 2} = \begin{pmatrix}
                         0 & 1 \\
                         1 & 0
                      \end{pmatrix}.
    \]

    This can be obtained in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    from toqito.matrices import gen_gell_mann
    
    print(gen_gell_mann(ind_1=0, ind_2=1, dim=2))
    ```

    The generalized Gell-Mann matrix `ind_1 = 2`, `ind_2 = 3`, and
    `dim = 4` is given as

    \[
        G_{2, 3, 4} = \begin{pmatrix}
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1 \\
                        0 & 0 & 1 & 0
                      \end{pmatrix}.
    \]

    This can be obtained in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    from toqito.matrices import gen_gell_mann
    
    gen_gell_mann(ind_1=2, ind_2=3, dim=4)
    ```

    Args:
        ind_1: A non-negative integer from 0 to `dim-1` (inclusive).
        ind_2: A non-negative integer from 0 to `dim-1` (inclusive).
        dim: The dimension of the Gell-Mann operator.

    Returns:
        The generalized Gell-Mann operator as an array.

    """
    if ind_1 == ind_2:
        if ind_1 == 0:
            gm_op = np.eye(dim)
        else:
            scalar = np.sqrt(2 / (ind_1 * (ind_1 + 1)))
            diag = np.ones((ind_1,))
            diag = np.append(diag, -ind_1)
            diag = scalar * np.append(diag, np.zeros((dim - ind_1 - 1)))

            gm_op = np.diag(diag)

    else:
        e_mat = np.zeros((dim, dim))
        e_mat[ind_1, ind_2] = 1
        if ind_1 < ind_2:
            gm_op = e_mat + e_mat.conj().T
        else:
            gm_op = 1j * e_mat - 1j * e_mat.conj().T

    return gm_op
