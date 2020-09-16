"""Determine if a list of matrices are POVM elements."""
from typing import List
import numpy as np

from toqito.matrix_props import is_positive_semidefinite


def is_povm(mat_list: List[np.ndarray]) -> bool:
    r"""
    Determine if a list of matrices constitute a valid set of POVMs [WikPOVM]_.

    A valid set of measurements are defined by a set of positive semidefinite operators

    .. math::
         \{P_a : a \in \Gamma\} \subset \text{Pos}(\mathcal{X}),

    indexed by the alphabet :math:`\Gamma` of measurement outcomes satisfying the constraint that

    .. math::
        \sum_{a \in \Gamma} P_a = I_{\mathcal{X}}.

    Examples
    ==========

    Consider the following matrices:

    .. math::
        M_0 =
        \begin{pmatrix}
            1 & 0 \\
            0 & 0
        \end{pmatrix}, \qquad
        M_1 =
        \begin{pmatrix}
            0 & 0 \\
            0 & 1
        \end{pmatrix}

    our function indicates that this set of operators constitute a set of POVMs.

    >>> from toqito.measurement_props import is_povm
    >>> import numpy as np
    >>> meas_1 = np.array([[1, 0], [0, 0]])
    >>> meas_2 = np.array([[0, 0], [0, 1]])
    >>> meas = [meas_1, meas_2]
    >>> is_povm(meas)
    True

    We may also use the :code:`random_povm` function from :code:`toqito`, and can verify that a
    randomly generated set satisfies the criteria for being a POVM set.

    >>> from toqito.measurement_props import is_povm
    >>> from toqito.random import random_povm
    >>> import numpy as np
    >>> dim, num_inputs, num_outputs = 2, 2, 2
    >>> measurements = random_povm(dim, num_inputs, num_outputs)
    >>> is_povm([measurements[:, :, 0, 0], measurements[:, :, 0, 1]])
    True

    Alternatively, the following matrices

    .. math::
        M_0 =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}, \qquad
        M_1 =
        \begin{pmatrix}
            5 & 6 \\
            7 & 8
        \end{pmatrix}

    does not constitute a POVM set.

    >>> from toqito.measurement_props import is_povm
    >>> import numpy as np
    >>> non_meas_1 = np.array([[1, 2], [3, 4]])
    >>> non_meas_2 = np.array([[5, 6], [7, 8]])
    >>> non_meas = [non_meas_1, non_meas_2]
    >>> is_povm(non_meas)
    False

    References
    ==========
    .. [WikPOVM] Wikipedia: POVM
        https://en.wikipedia.org/wiki/POVM

    :param mat_list: A list of matrices.
    :return: Return :code:`True` if set of matrices constitutes a set of
             measurements, and :code:`False` otherwise.
    """
    dim = mat_list[0].shape[0]

    mat_sum = np.zeros((dim, dim), dtype=complex)
    for mat in mat_list:
        # Each measurement in the set must be positive semidefinite.
        if not is_positive_semidefinite(mat):
            return False
        mat_sum += mat
    # Summing all the measurements from the set must be equal to the identity.
    if not np.allclose(np.identity(dim), mat_sum):
        return False
    return True
