"""Compute a list of Kraus operators from the Choi matrix."""


import numpy as np

from toqito.helper import channel_dim
from toqito.matrix_ops import unvec
from toqito.matrix_props import is_hermitian, is_positive_semidefinite


def choi_to_kraus(
    choi_mat: np.ndarray, tol: float = 1e-9, dim: int | list[int] | np.ndarray = None
) -> list[np.ndarray] | list[list[np.ndarray]]:
    r"""Compute a list of Kraus operators from the Choi matrix from :cite:`Rigetti_2022_Forest`.

    Note that unlike the Choi or natural representation of operators, the Kraus representation is
    *not* unique.

    If the input channel maps :math:`M_{r,c}` to :math:`M_{x,y}` then :code:`dim` should be the
    list :code:`[[r,x], [c,y]]`. If it maps :math:`M_m` to :math:`M_n`, then :code:`dim` can simply
    be the vector :code:`[m,n]`.

    For completely positive maps the output is a single flat list of numpy arrays since the left and
    right Kraus maps are the same.

    This function has been adapted from :cite:`Rigetti_2022_Forest` and QETLAB :cite:`QETLAB_link`.

    Examples
    ========

    Consider taking the Kraus operators of the Choi matrix that characterizes the "swap operator"
    defined as

    .. math::
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    The corresponding Kraus operators of the swap operator are given as follows,

    .. math::
        \begin{equation}
        \big[
            \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},
            \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
        \big],
        \big[
            \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
            \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
        \big],
        \big[
            \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix},
            \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
        \big],
        \big[
            \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix},
            \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
        \big]
        \end{equation}

    This can be verified in :code:`toqito` as follows.

    >>> import numpy as np
    >>> from toqito.channel_ops import choi_to_kraus
    >>> choi_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    >>> kraus_ops = choi_to_kraus(choi_mat)
    >>> kraus_ops
    [
        [
            array([[0.,  0.70710678], [-0.70710678, 0.]]),
            array([[0., -0.70710678], [ 0.70710678, 0.]])
        ],
        [
            array([[0., 0.70710678], [0.70710678, 0.]]),
            array([[0., 0.70710678], [0.70710678, 0.]])
        ],
        [array([[1., 0.], [0., 0.]]), array([[1., 0.], [0., 0.]])],
        [array([[0., 0.], [0., 1.]]), array([[0., 0.], [0., 1.]])]
    ]

    See Also
    ========
    kraus_to_choi

    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param choi_mat: A Choi matrix
    :param tol: optional threshold parameter for eigenvalues/kraus ops to be discarded
    :param dim: A scalar, vector or matrix containing the input and output dimensions of Choi matrix.
    :return: List of Kraus operators

    """
    d_in, d_out, _ = channel_dim(choi_mat, dim=dim, compute_env_dim=False)
    if is_hermitian(choi_mat):
        eigvals, v_mat = np.linalg.eigh(choi_mat)
        kraus_0 = [
            np.sqrt(abs(eigval)) * unvec(evec, shape=(d_out[0], d_in[0]))
            for eigval, evec in zip(eigvals, v_mat.T)
            if abs(eigval) > tol
        ]

        if is_positive_semidefinite(choi_mat):
            return kraus_0

        kraus_1 = [
            np.sign(eigval) * k_mat
            for eigval, k_mat in zip(filter(lambda eigval: abs(eigval) > tol, eigvals), kraus_0)
        ]
    else:
        u_mat, singular_values, vh_mat = np.linalg.svd(choi_mat, full_matrices=False)
        kraus_0 = [
            np.sqrt(s_val) * unvec(evec, shape=(d_out[0], d_in[0]))
            for s_val, evec in zip(singular_values, u_mat.T)
            if abs(s_val) > tol
        ]

        kraus_1 = [
            np.sqrt(s_val) * unvec(evec.conj(), shape=(d_out[1], d_in[1]))
            for s_val, evec in zip(singular_values, vh_mat)
            if abs(s_val) > tol
        ]

    return [[ka, kb] for ka, kb in zip(kraus_0, kraus_1)]
