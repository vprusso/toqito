"""Checks if the matrix is a density matrix."""

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_density(mat: np.ndarray) -> bool:
    r"""Check if matrix is a density matrix [@WikiDen].

    A matrix is a density matrix if its trace is equal to one and it has the
    property of being positive semidefinite (PSD).

    Examples:

    Consider the Bell state:

    \[
        u = \frac{1}{\sqrt{2}} |00 \rangle + \frac{1}{\sqrt{2}} |11 \rangle.
    \]

    Constructing the matrix \(\rho = u u^*\) defined as

    \[
        \rho = \frac{1}{2} \begin{pmatrix}
                                1 & 0 & 0 & 1 \\
                                0 & 0 & 0 & 0 \\
                                0 & 0 & 0 & 0 \\
                                1 & 0 & 0 & 1
                           \end{pmatrix}
    \]

    our function indicates that this is indeed a density operator as the trace
    of \(\rho\) is equal to \(1\) and the matrix is positive
    semidefinite.

    ```python exec="1" source="above"
    from toqito.matrix_props import is_density
    from toqito.states import bell
    import numpy as np
    rho = bell(0) @ bell(0).conj().T
    print(is_density(rho))
    ```

    Alternatively, the following example matrix \(\sigma\) defined as

    \[
        \sigma = \frac{1}{2} \begin{pmatrix}
                                1 & 2 \\
                                3 & 1
                             \end{pmatrix}
    \]

    does satisfy \(\text{Tr}(\sigma) = 1\), however fails to be positive
    semidefinite, and is therefore not a density operator. This can be
    illustrated using `|toqito⟩` as follows.

    ```python exec="1" source="above"
    import numpy as np
    from toqito.states import bell
    from toqito.matrix_props import is_density
    
    sigma = 1/2 * np.array([[1, 2], [3, 1]])
    
    print(is_density(sigma))
    ```

    Args:
        mat: Matrix to check.

    Returns:
        Return `True` if matrix is a density matrix, and `False` otherwise.

    """
    return bool(is_positive_semidefinite(mat) and np.isclose(np.trace(mat), 1))
