"""Maximally mixed states are states which are formed as a uniform mixture of states in an orthonormal basis.

The density matrix of a maximally mixed state is directly proportional to the identity matrix.
"""

import numpy as np
from scipy.sparse import dia_array, eye_array


def max_mixed(dim: int, is_sparse: bool = False) -> [np.ndarray, dia_array]:
    r"""Produce the maximally mixed state :footcite:`Aaronson_2018_MaxMixed`.

    Produces the maximally mixed state on of :code:`dim` dimensions. The maximally mixed state is defined as

    .. math::
        \omega = \frac{1}{d} \begin{pmatrix}
                        1 & 0 & \ldots & 0 \\
                        0 & 1 & \ldots & 0 \\
                        \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & \ldots & 1
                    \end{pmatrix},

    or equivalently, it is defined as

    .. math::
        \omega = \frac{\mathbb{I}}{\text{dim}(\mathcal{X})}

    for some complex Euclidean space :math:`\mathcal{X}`. The maximally mixed state is sometimes also referred to as the
    tracial state.

    The maximally mixed state is returned as a sparse matrix if :code:`is_sparse = True` and is full if :code:`is_sparse
    = False`.

    Examples
    ==========

    Using :code:`|toqito⟩`, we can generate the :math:`2`-dimensional maximally mixed state

    .. math::
        \omega_2 = \frac{1}{2}
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}

    as follows.

    .. jupyter-execute::

        from toqito.states import max_mixed
        max_mixed(2, is_sparse=False)



    One may also generate a maximally mixed state returned as a sparse matrix

    .. jupyter-execute::

        from toqito.states import max_mixed
        max_mixed(2, is_sparse=True)


    References
    ==========
    .. footbibliography::



    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is sparse and `False` otherwise.
    :return: The maximally mixed state of dimension `dim`.

    """
    if is_sparse:
        return 1 / dim * eye_array(dim)
    return 1 / dim * np.eye(dim)
