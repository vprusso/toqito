"""Generates and applies Pauli Channel to a matrix."""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from toqito.channel_ops import kraus_to_choi
from toqito.helper import update_odometer
from toqito.matrices import pauli


def pauli_channel(
    prob: Union[float, np.ndarray] = 0,
    kraus_ops: bool = False,
    input_mat: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Generate and apply a Pauli channel to a matrix.

    Generates the Choi matrix of a Pauli channel with given probabilities and optionally applies it
    to an input matrix. The Pauli channel is defined by the set of Pauli operators weighted by
    the probability vector. For a given probability vector :math:`(p_1, \ldots, p_{4^Q})`, the
    channel is defined as:

    .. math::
       \Phi(\rho) = \sum_{i=1}^{4^Q} p_i P_i \rho P_i^*

    where :math:`P_i` are Pauli operators and :math:`Q` is the number of qubits.
    If :code:`prob` is a scalar, it generates a random :code:`prob`-qubit Pauli channel.
    The length of the probability vector (if provided) must be :math:`4^Q` for some
    integer :math:`Q` (number of qubits).

    Examples
    ========

    Generate a random single-qubit Pauli channel:

    >>> from toqito.channels import pauli_channel
    >>> choi_matrix = pauli_channel(prob = 1)

    Apply a specific two-qubit Pauli channel to an input matrix:

    >>> from toqito.channels import pauli_channel
    >>> import numpy as np
    >>> _, output = pauli_channel(
    ...     prob=np.array([0.1, 0.2, 0.3, 0.4]), input_mat=np.eye(2)
    ... )  # doctest: +NORMALIZE_WHITESPACE
    >>> print(output)  # doctest: +NORMALIZE_WHITESPACE
    [[1.+0.j 0.+0.j]
     [0.+0.j 1.+0.j]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param prob: Probability vector for Pauli operators. If scalar, generates random probabilities
                 for `Q=prob` qubits. Default is 0 (single qubit with random probabilities).
    :param kraus_ops: Flag to return Kraus operators. Default is False.
    :param input_mat: Optional input matrix to apply the channel to. Default is ``None`.
    :raises ValueError: If probabilities are negative or don't sum to 1.
    :raises ValueError: If length of probability vector is not `4^Q` for some integer `Q`.
    :return: The Choi matrix of the channel. If input_mat is provided, also returns the output matrix.
             If kraus_ops is True, returns Kraus operators as well.

    """
    if not isinstance(prob, np.ndarray):
        if np.isscalar(prob):
            q = int(prob)
            len_p = 4**q
            prob = np.random.rand(len_p)
            prob /= np.sum(prob)

        else:
            prob = np.array(prob)

    if np.any(prob < 0) or not np.isclose(np.sum(prob), 1):
        raise ValueError("Probabilities must be non-negative and sum to 1.")

    len_p = len(prob)
    q = int(np.round(np.log2(len_p) / 2))

    if len_p != 4**q:
        raise ValueError("The length of the probability vector must be 4^Q for some integer Q (number of qubits).")

    Phi = sparse.csc_matrix((4**q, 4**q), dtype=complex)

    kraus_operators = []
    ind = np.zeros(q, dtype=int)

    for j in range(len_p):
        pauli_op = pauli(ind.tolist())
        kraus_operators.append(np.sqrt(prob[j]) * pauli_op)
        Phi += prob[j] * kraus_to_choi([[pauli_op, pauli_op.conj().T]])
        ind = update_odometer(ind, 4 * np.ones(q, dtype=int))

    if input_mat is not None:
        output_mat = np.zeros_like(input_mat, dtype=complex)
        for kraus in kraus_operators:
            output_mat += kraus @ input_mat @ kraus.conj().T
    else:
        output_mat = None

    if kraus_ops:
        if input_mat is not None:
            return Phi, output_mat, kraus_operators
        else:
            return Phi, kraus_operators
    elif input_mat is not None:
        return Phi, output_mat
    else:
        return Phi
