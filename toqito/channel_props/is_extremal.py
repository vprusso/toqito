"""Determines whether a quantum channel is extremal."""

import numpy as np
from numpy.linalg import matrix_rank

from toqito.channel_ops.choi_to_kraus import choi_to_kraus


def is_extremal(phi: np.ndarray | list[np.ndarray | list[np.ndarray]], tol: float = 1e-9) -> bool:
    r"""Determine whether a quantum channel is extremal.

    (Section 2.2.4: Extremal Channels from :footcite:`Watrous_2018_TQI`).

    Theorem 2.31 in :footcite:`Watrous_2018_TQI` provides the characterization of extremal
    quantum channels as a channel :math:`\Phi` is an extreme point of the convex set
    of quantum channels if and only if the collection:

    .. math::
        \{ A_i^\dagger A_j \}_{i,j=1}^{r}

    is linearly independent.

    The channel can be provided in one of the following representations:

    - A Choi matrix, representing the quantum channel in the Choi representation. It will
      be converted internally to a set of Kraus operators.
    - A list of Kraus operators, representing the channel in Kraus form.
    - A nested list of Kraus operators, which will be flattened automatically.

    Examples
    ==========

    The following demonstrates an example of an extremal quantum channel from Example 2.33
    in :footcite:`Watrous_2018_TQI`.

    .. jupyter-execute::

     import numpy as np
     from toqito.channel_props import is_extremal
     kraus_ops = [
         (1 / np.sqrt(6)) * np.array([[2, 0], [0, 1], [0, 1], [0, 0]]),
         (1 / np.sqrt(6)) * np.array([[0, 0], [1, 0], [1, 0], [0, 2]])
     ]

     is_extremal(kraus_ops)

    References
    ==========
    .. footbibliography::


    :param phi: The quantum channel, which may be given as a Choi matrix or a list of Kraus operators.
    :param tol: Tolerance value for numerical precision in rank computation.
    :type phi: list[numpy.ndarray] | list[list[numpy.ndarray]] | numpy.ndarray
    :raises ValueError: If the input is neither a valid list of Kraus operators nor a Choi matrix.
    :return: True if the channel is extremal; False otherwise.

    """
    # If input is a Choi matrix, convert to a (flat) list of Kraus operators.
    if isinstance(phi, np.ndarray):
        kraus_ops = choi_to_kraus(phi)
    elif isinstance(phi, list):
        # If the first element is a list, assume nested list of Kraus operators.
        if len(phi) == 0:
            raise ValueError("The channel must contain at least one Kraus operator.")
        if isinstance(phi[0], list):
            # Flatten the nested list.
            kraus_ops = [op for sublist in phi for op in sublist if isinstance(op, np.ndarray)]
        elif all(isinstance(op, np.ndarray) for op in phi):
            kraus_ops = phi
        else:
            raise ValueError("Channel must be a list (or nested list) of Kraus operators.")
    else:
        raise ValueError("Channel must be a list of Kraus operators or a Choi matrix.")

    # Check that we have at least one Kraus operator.
    if not kraus_ops:
        raise ValueError("The channel must contain at least one Kraus operator.")

    r = len(kraus_ops)

    # A single Kraus operator (e.g., a unitary channel) is always extremal.
    if r == 1:
        return True

    # Compute the set {A_i^† A_j} for every pair (i, j).
    flattened_products = [np.dot(A.conj().T, B).flatten() for A in kraus_ops for B in kraus_ops]

    # Form a matrix whose columns are these vectorized operators.
    M = np.column_stack(flattened_products)

    # The channel is extremal if and only if the operators {A_i^† A_j} are linearly independent,
    # i.e. the rank of M equals r^2.
    return bool(matrix_rank(M, tol=tol) == r * r)
