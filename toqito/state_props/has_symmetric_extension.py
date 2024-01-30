"""Determine whether there exists a symmetric extension for a given quantum state."""


import numpy as np
from picos import partial_trace

from toqito.matrix_props import is_positive_semidefinite
from toqito.state_opt import symmetric_extension_hierarchy
from toqito.state_props import is_ppt


def has_symmetric_extension(
    rho: np.ndarray,
    level: int = 2,
    dim: np.ndarray | int = None,
    ppt: bool = True,
    tol: float = 1e-4,
) -> bool:
    r"""Determine whether there exists a symmetric extension for a given quantum state.

    For more information, see :cite:`Doherty_2002_Distinguishing`.

    Determining whether an operator possesses a symmetric extension at some level :code:`level`
    can be used as a check to determine if the operator is entangled or not.

    This function was adapted from QETLAB.

    Examples
    ==========

    2-qubit symmetric extension:

    In :cite:`Chen_2014_Symmetric`, it was shown that a 2-qubit state :math:`\rho_{AB}` has a
    symmetric extension if and only if

    .. math::
        \text{Tr}(\rho_B^2) \geq \text{Tr}(\rho_{AB}^2) - 4 \sqrt{\text{det}(\rho_{AB})}.

    This closed-form equation is much quicker to check than running the semidefinite program.

    >>> import numpy as np
    >>> from toqito.state_props import has_symmetric_extension
    >>> rho = np.array([[1, 0, 0, -1],
    ...                 [0, 1, 1/2, 0],
    ...                 [0, 1/2, 1, 0],
    ...                 [-1, 0, 0, 1]])
    >>> # Show the closed-form equation holds
    >>> np.trace(np.linalg.matrix_power(partial_trace(rho, 1), 2)) >= np.trace(rho**2) - 4 * np.sqrt(np.linalg.det(rho))
    True
    >>> # Now show that the `has_symmetric_extension` function recognizes this case.
    >>> has_symmetric_extension(rho)
    True

    Higher qubit systems:

    Consider a density operator corresponding to one of the Bell states.

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                            1 & 0 & 0 & 1 \\
                            0 & 0 & 0 & 0 \\
                            0 & 0 & 0 & 0 \\
                            1 & 0 & 0 & 1
                           \end{pmatrix}

    To make this state over more than just two qubits, let's construct the following state

    .. math::
        \sigma = \rho \otimes \rho.

    As the state :math:`\sigma` is entangled, there should not exist a symmetric extension at some
    level. We see this being the case for a relatively low level of the hierachy.

    >>> import numpy as np
    >>> from toqito.states import bell
    >>> from toqito.state_props import has_symmetric_extension
    >>>
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = np.kron(rho, rho)
    >>> has_symmetric_extension(sigma)
    False

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If dimension does not evenly divide matrix length.
    :param rho: A matrix or vector.
    :param level: Level of the hierarchy to compute.
    :param dim: The default has both subsystems of equal dimension.
    :param ppt: If :code:`True`, this enforces that the symmetric extension must be PPT.
    :param tol: Tolerance when determining whether a symmetric extension exists.
    :return: :code:`True` if :code:`mat` has a symmetric extension; :code:`False` otherwise.

    """
    len_mat = rho.shape[1]

    # Set default dimension if none was provided.
    if dim is None:
        dim = int(np.round(np.sqrt(len_mat)))

    # Allow the user to enter in a single integer for dimension.
    if isinstance(dim, int):
        dim = np.array([dim, len_mat / dim])  # pylint: disable=redefined-variable-type
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len_mat * np.finfo(float).eps:
            raise ValueError(
                "If `dim` is a scalar, it must evenly divide the length of the matrix."
            )
        dim[1] = int(np.round(dim[1]))

    dim = np.int_(dim)

    dim_x, dim_y = int(dim[0]), int(dim[1])  # pylint: disable=unsubscriptable-object
    # In certain situations, we don't need semidefinite programming.
    if level == 1 or len_mat <= 6 and ppt:
        if not ppt:
            # In some cases, the problem is *really* trivial.
            return is_positive_semidefinite(rho)

        # In this case, all they asked for is a 1-copy PPT symmetric extension
        # (i.e., they're asking if the state is PPT).
        return is_ppt(rho, 2, dim) and is_positive_semidefinite(rho)

    # In the 2-qubit case, an analytic formula is known for whether or not a state has a
    # (2-copy, non-PPT) symmetric extension that is much faster to use than semidefinite
    # programming [CJKLZB14]_.
    if level == 2 and not ppt and dim_x == 2 and dim_y == 2:
        return np.trace(np.linalg.matrix_power(partial_trace(rho, [0]), 2)) >= np.trace(
            np.linalg.matrix_power(rho, 2)
        ) - 4 * np.sqrt(np.linalg.det(rho))

    # Otherwise, use semidefinite programming to find a symmetric extension.
    # If the optimal value of the symmetric extension hierarchy is equal to 1,
    # this indicates that there does not exist a symmetric extension at
    # level :code:`level`.
    return not np.isclose(
        (1 - min(symmetric_extension_hierarchy([rho], probs=None, level=level), 1)),
        0,
        atol=tol,
    )
