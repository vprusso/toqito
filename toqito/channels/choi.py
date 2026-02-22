"""Generates the Choi channel."""

import numpy as np

from toqito.states import max_entangled


def choi(a_var: int = 1, b_var: int = 1, c_var: int = 0) -> np.ndarray:
    r"""Produce the Choi channel or one of its generalizations [@Choi_1992_Generalized].

    The *Choi channel* is a positive map on 3-by-3 matrices that is capable of detecting some
    entanglement that the transpose map is not.

    The standard Choi channel defined with `a=1`, `b=1`, and `c=0` is the Choi
    matrix of the positive map defined in [@Choi_1992_Generalized]. Many of these maps are capable of detecting
    PPT entanglement.

    Examples:

    The standard Choi channel is given as

    \[
        \Phi_{1, 1, 0} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 1
        \end{pmatrix}
    \]

    We can generate the Choi channel in `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import choi
    
    print(choi())
    ```

    The reduction channel is the map \(R\) defined by:

    \[
        R(X) = \text{Tr}(X) \mathbb{I} - X.
    \]

    The matrix correspond to this is given as

    \[
        \Phi_{0, 1, 1} =
        \begin{pmatrix}
            0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0
        \end{pmatrix}
    \]

    The reduction channel is the Choi channel that arises when `a = 0` and when `b =
    c = 1`. We can obtain this matrix using `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.channels import choi
    
    print(choi(0, 1, 1))
    ```

    !!! See Also
        [reduction][toqito.channels.reduction.reduction]

    Args:
        a_var: Default integer for standard Choi map.
        b_var: Default integer for standard Choi map.
        c_var: Default integer for standard Choi map.

    Returns:
        The Choi channel (or one of its  generalizations).

    """
    psi = max_entangled(3, False, False)
    return np.diag([a_var + 1, c_var, b_var, b_var, a_var + 1, c_var, c_var, b_var, a_var + 1]) - psi @ psi.conj().T
