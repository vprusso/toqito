"""Determine if one vector or matrix majorizes another."""

import numpy as np


def majorizes(a_var: np.ndarray | list[int], b_var: np.ndarray | list[int]) -> bool:
    r"""Determine if one vector or matrix majorizes another [@WikiMajorization].

    Given \(a, b \in \mathbb{R}^d\), we say that \(a\) **weakly majorizes** (or dominates)
    \(b\) from below if and only if

    \[
        \sum_{i=1}^k a_i^{\downarrow} \geq \sum_{i=1}^k b_i^{\downarrow}
    \]

    for all \(k \in \{1, \ldots, d\}\).

    This function was adapted from the QETLAB package.

    Examples:
        Simple example illustrating that the vector \((3, 0, 0)\) majorizes the vector
        \((1, 1, 1)\).

        ```python exec="1" source="above"
        from toqito.matrix_props import majorizes

        print(majorizes([3, 0, 0], [1, 1, 1]))
        ```


        The majorization criterion says that every separable state
        \(\rho \in \text{D}(\mathcal{A} \otimes \mathcal{B})\) is such that
        \(\text{Tr}_{\mathcal{B}}(\rho)\) majorizes
        \(\text{Tr}_{\mathcal{A}}(\rho)\).

        ```python exec="1" source="above"
        from toqito.matrix_props import majorizes
        from toqito.states import max_entangled
        from toqito.matrix_ops import partial_trace

        v_vec = max_entangled(3)
        rho = v_vec @ v_vec.conj().T

        print(majorizes(partial_trace(rho, [1]), rho))
        ```

    Args:
        a_var: Matrix or vector provided as list or np.array.
        b_var: Matrix or vector provided as list or np.array.

    Returns:
        Return `True` if `a_var` majorizes `b_var` and `False` otherwise.

    """
    # If input if provided as list, convert to np.array.
    if isinstance(a_var, list):
        a_var = np.array(a_var)
    if isinstance(b_var, list):
        b_var = np.array(b_var)

    # If matrix, obtain singular values for majorization.
    if len(a_var.shape) == 1:
        a_var = np.sort(a_var)[::-1]
    # Otherwise, just sort in descending order.
    else:
        _, a_var, _ = np.linalg.svd(a_var)

    # Do the same for second input argument.
    if len(b_var.shape) == 1:
        b_var = np.sort(b_var)[::-1]
    else:
        _, b_var, _ = np.linalg.svd(b_var)

    la_var = len(a_var)
    lb_var = len(b_var)

    # If different length vectors, pad with zeros.
    if la_var < lb_var:
        a_var = np.pad(a_var, (0, lb_var - la_var), "constant")
    elif lb_var < la_var:
        b_var = np.pad(b_var, (0, la_var - lb_var), "constant")

    cta = 0
    ctb = -np.linalg.norm(a_var) * np.finfo(float).eps ** (3 / 4)

    # Check for majorization.
    for k, _ in enumerate(a_var):
        cta = cta + a_var[k]
        ctb = ctb + b_var[k]
        if cta < ctb:
            return False
    return True
