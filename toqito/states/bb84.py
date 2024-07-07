"""BB84 basis states."""

import numpy as np

from toqito.matrices import standard_basis


def bb84() -> np.ndarray:
    r"""Obtain the BB84 basis states :cite:`WikiBB84`.

    The BB84 basis states are defined as

    .. math::
        |0\rangle := \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \\
        |1\rangle := \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad \\
        |+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \\
        |-\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}.

    Examples
    ==========
    The BB84 basis states can be obtained in :code:`toqito` as follows in the form of a list of
    arrays.

    >>> from toqito.states import bb84
    >>> bb84()
    [[array([[1.],
           [0.]]), array([[0.],
           [1.]])], [array([[0.70710678],
           [0.70710678]]), array([[ 0.70710678],
           [-0.70710678]])]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :return: The four BB84 basis states.

    """
    # Computational basis states |0>, |1>:
    e_0, e_1 = standard_basis(2)
    # Plus/minus basis |+>, |->
    e_p, e_m = (e_0 + e_1) / np.sqrt(2), (e_0 - e_1) / np.sqrt(2)
    return [[e_0, e_1], [e_p, e_m]]
