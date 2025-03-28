"""Converts Kraus operators into the corressponding quantum channel (i.e. superoperator)."""

import numpy as np

from toqito.matrix_ops import tensor


def kraus_to_channel(
    kraus_list: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    r"""Convert a collection of Kraus operators into the corresponding quantum channel (superoperator).

    (Section: Kraus Representations of :cite:`Watrous_2018_TQI`).

    This function computes the superoperator representation of a quantum channel from its Kraus representation.
    Given a list of Kraus operators \(\{A_i, B_i\}\), the superoperator \(\mathcal{E}\) is computed as:

    \[
    \mathcal{E}(\rho) = \sum_i B_i \rho A_i^\dagger
    \]

    The resulting quantum channel can be applied to density matrices by reshaping them into vectorized form.

    Examples
    ========

    Constructing a simple quantum channel from Kraus operators:

    >>> import numpy as np
    >>> from toqito.channel_ops import kraus_to_channel
    >>> kraus_1 = np.array([[1, 0], [0, 0]])
    >>> kraus_2 = np.array([[0, 1], [0, 0]])
    >>> kraus_list = [(kraus_1, kraus_1), (kraus_2, kraus_2)]
    >>> kraus_to_channel(kraus_list)
    array([[1, 0, 0, 1],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])

    See Also
    ========
    choi_to_kraus, kraus_to_choi

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param kraus_list: List of tuples (A, B) where A and B are Kraus operators as numpy arrays.
    :return: The superoperator as a numpy array.

    """
    super_op = sum(tensor(B, A.conj()) for A, B in kraus_list)
    return super_op
