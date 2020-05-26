"""Matrices."""
from typing import List, Union
from cmath import exp, pi
from scipy.sparse import csr_matrix
from scipy import sparse

import numpy as np

from toqito.matrix_ops import tensor


__all__ = [
    "clock",
    "cnot",
    "fourier",
    "gell_mann",
    "gen_gell_mann",
    "gen_pauli",
    "hadamard",
    "iden",
    "pauli",
    "shift",
]


def clock(dim: int) -> np.ndarray:
    r"""
    Produce clock matrix [WikClock]_.

    Returns the clock matrix of dimension `dim` described in [WikClock]_. The
    clock matrix generates the following `dim`-by-`dim` matrix

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & \omega & 0 & \ldots & 0 \\
                        0 & 0 & \omega^2 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & \omega^{d-1}
                   \end{pmatrix}

    where :math:`\omega` is the n-th primitive root of unity.

    The clock matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The clock matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & \omega & 0 \\
            0 & 0 & \omega^2
        \end{pmatrix}

    >>> from toqito.matrices import clock
    >>> clock(3)
    [[ 1. +0.j       ,  0. +0.j       ,  0. +0.j       ],
     [ 0. +0.j       , -0.5+0.8660254j,  0. +0.j       ],
     [ 0. +0.j       ,  0. +0.j       , -0.5-0.8660254j]]

    References
    ==========
    .. [WikClock] Wikipedia: Generalizations of Pauli matrices,
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    :param dim: Dimension of the matrix.
    :return: `dim`-by-`dim` clock matrix.
    """
    c_var = 2j * pi / dim
    omega = (exp(k * c_var) for k in range(dim))
    return np.diag(list(omega))


def cnot() -> np.ndarray:
    r"""
    Produce the CNOT matrix [WikCNOT]_.

    The CNOT matrix is defined as

    .. math::
        \text{CNOT} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{pmatrix}.

    Examples
    ==========

    >>> from toqito.matrices import cnot
    >>> cnot()
    [[1 0 0 0]
     [0 1 0 0]
     [0 0 0 1]
     [0 0 1 0]]

    References
    ==========
    .. [WikCNOT] Wikipedia: Controlled NOT gate
        https://en.wikipedia.org/wiki/Controlled_NOT_gate

    :return: The CNOT matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def fourier(dim: int) -> np.ndarray:
    r"""
    Generate the Fourier transform matrix [WikDFT]_.

    Generates the `dim`-by-`dim` unitary matrix that implements the quantum
    Fourier transform.

    The Fourier matrix is defined as:

    .. math::
        W_N = \frac{1}{N}
        \begin{pmatrix}
            1 & 1 & 1 & 1 & \ldots & 1 \\
            1 & \omega & \omega^2 & \omega^3 & \ldots & \omega^{N-1} \\
            1 & \omega^2 & \omega^4 & \omega^6 & \ldots & \omega^{2(N-1)} \\
            1 & \omega^3 & \omega^6 & \omega^9 & \ldots & \omega^{3(N-1)} \\
            \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
            1 & \omega^{N-1} & \omega^{2(N-1)} & \omega^{3(N-1)} &
            \ldots & \omega^{3(N-1)}
        \end{pmatrix}

    Examples
    ==========

    The Fourier matrix generated from :math:`d = 3` yields the following
    matrix:

    .. math::
        W_3 = \frac{1}{3}
        \begin{pmatrix}
            1 & 1 & 1 \\
            0 & \omega & \omega^2 \\
            1 & \omega^2 & \omega^4
        \end{pmatrix}

    >>> from toqito.matrices import fourier
    >>> fourier(3)
    [[ 0.57735027+0.j ,  0.57735027+0.j ,  0.57735027+0.j ],
     [ 0.57735027+0.j , -0.28867513+0.5j, -0.28867513-0.5j],
     [ 0.57735027+0.j , -0.28867513-0.5j, -0.28867513+0.5j]]

    References
    ==========
    .. [WikDFT] Wikipedia: DFT matrix,
        https://en.wikipedia.org/wiki/DFT_matrix

    :param dim: The size of the Fourier matrix.
    :return: The Fourier matrix of dimension `dim`.
    """
    # Primitive root of unity.
    root_unity = np.exp(2 * 1j * np.pi / dim)
    entry_1 = np.arange(0, dim)[:, None]
    entry_2 = np.arange(0, dim)
    return np.power(root_unity, entry_1 * entry_2) / np.sqrt(dim)


def gell_mann(ind: int, is_sparse: bool = False) -> np.ndarray:
    r"""
    Produce a Gell-Mann operator [WikGM]_.

    Generates the 3-by-3 Gell-Mann matrix indicated by the value of `ind`.
    `ind = 0` gives the identity matrix, while values 1 through 8 each indicate
    one of the other 8 Gell-Mann matrices.

    The 9 Gell-Mann matrices are defined as follows:

    .. math::
        \begin{equation}
            \begin{aligned}
                \lambda_0 = \begin{pmatrix}
                                1 & 0 & 0 \\
                                0 & 1 & 0 \\
                                0 & 0 & 1
                            \end{pmatrix}, \quad
                \lambda_1 = \begin{pmatrix}
                                0 & 1 & 0 \\
                                1 & 0 & 0 \\
                                0 & 0 & 0
                            \end{pmatrix}, \quad &
                \lambda_2 = \begin{pmatrix}
                                0 & -i & 0 \\
                                i & 0 & 0 \\
                                0 & 0 & 0
                            \end{pmatrix},  \\
                \lambda_3 = \begin{pmatrix}
                                1 & 0 & 0 \\
                                0 & -1 & 0 \\
                                0 & 0 & 0
                            \end{pmatrix}, \quad
                \lambda_4 = \begin{pmatrix}
                                0 & 0 & 1 \\
                                0 & 0 & 0 \\
                                1 & 0 & 0
                            \end{pmatrix}, \quad &
                \lambda_5 = \begin{pmatrix}
                                0 & 0 & -i \\
                                0 & 0 & 0 \\
                                i & 0 & 0
                            \end{pmatrix},  \\
                \lambda_6 = \begin{pmatrix}
                                0 & 0 & 0 \\
                                0 & 0 & 1 \\
                                0 & 1 & 0
                            \end{pmatrix}, \quad
                \lambda_7 = \begin{pmatrix}
                                0 & 0 & 0 \\
                                0 & 0 & -i \\
                                0 & i & 0
                            \end{pmatrix}, \quad &
                \lambda_8 = \frac{1}{\sqrt{3}} \begin{pmatrix}
                                                    1 & 0 & 0 \\
                                                    0 & 1 & 0 \\
                                                    0 & 0 & -2
                                                \end{pmatrix}.
                \end{aligned}
            \end{equation}

    Examples
    ==========

    The Gell-Mann matrix generated from :math:`idx = 2` yields the following
    matrix:

    .. math::

        \lambda_2 = \begin{pmatrix}
                            0 & -i & 0 \\
                            i & 0 & 0 \\
                            0 & 0 & 0
                    \end{pmatrix}

    >>> from toqito.matrices import gell_mann
    >>> gell_mann(2)
    [[ 0.+0.j, -0.-1.j,  0.+0.j],
     [ 0.+1.j,  0.+0.j,  0.+0.j],
     [ 0.+0.j,  0.+0.j,  0.+0.j]]

    References
    ==========
    .. [WikGM] Wikipedia: Gell-Mann matrices,
        https://en.wikipedia.org/wiki/Gell-Mann_matrices

    :param ind: An integer between 0 and 8 (inclusive).
    :param is_sparse: Boolean to determine whether matrix is sparse.
    """
    if ind == 0:
        gm_op = np.identity(3)
    elif ind == 1:
        gm_op = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    elif ind == 2:
        gm_op = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    elif ind == 3:
        gm_op = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    elif ind == 4:
        gm_op = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    elif ind == 5:
        gm_op = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    elif ind == 6:
        gm_op = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    elif ind == 7:
        gm_op = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    elif ind == 8:
        gm_op = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3)
    else:
        raise ValueError(
            "Gell-Mann index values can only be values from 0 to " "8 (inclusive)."
        )

    if is_sparse:
        gm_op = csr_matrix(gm_op)

    return gm_op


def gen_gell_mann(
    ind_1: int, ind_2: int, dim: int, is_sparse: bool = False
) -> Union[np.ndarray, sparse.lil_matrix]:
    r"""
    Produce a generalized Gell-Mann operator [WikGM2]_.

    Construct a `dim`-by-`dim` Hermitian operator. These matrices span the
    entire space of `dim`-by-`dim` matrices as `ind_1` and `ind_2` range from 0
    to `dim-1`, inclusive, and they generalize the Pauli operators when `dim =
    2` and the Gell-Mann operators when `dim = 3`.

    Examples
    ==========

    The generalized Gell-Mann matrix for `ind_1 = 0`, `ind_2 = 1` and `dim = 2`
    is given as

    .. math::
        G_{0, 1, 2} = \begin{pmatrix}
                         0 & 1 \\
                         1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.

    >>> from toqito.matrices import gen_gell_mann
    >>> gen_gell_mann(0, 1, 2)
    [[0., 1.],
     [1., 0.]])

    The generalized Gell-Mann matrix `ind_1 = 2`, `ind_2 = 3`, and `dim = 4` is
    given as

    .. math::
        G_{2, 3, 4} = \begin{pmatrix}
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1 \\
                        0 & 0 & 1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.

    >>> from toqito.matrices import gen_gell_mann
    >>> gen_gell_mann(2, 3, 4)
    [[0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 1.],
     [0., 0., 1., 0.]])

    References
    ==========
    .. [WikGM2] Wikipedia: Gell-Mann matrices,
        https://en.wikipedia.org/wiki/Gell-Mann_matrices

    :param ind_1: A non-negative integer from 0 to `dim-1` (inclusive).
    :param ind_2: A non-negative integer from 0 to `dim-1` (inclusive).
    :param dim: The dimension of the Gell-Mann operator.
    :param is_sparse: If set to `True`, the returned Gell-Mann operator is a
                      sparse lil_matrix and if set to `False`, the returned
                      Gell-Mann operator is a dense numpy array.
    :return: The generalized Gell-Mann operator.
    """
    if ind_1 == ind_2:
        if ind_1 == 0:
            gm_op = sparse.eye(dim)
        else:
            scalar = np.sqrt(2 / (ind_1 * (ind_1 + 1)))
            diag = np.ones((ind_1, 1))
            diag = np.append(diag, -ind_1)
            diag = scalar * np.append(diag, np.zeros((dim - ind_1 - 1, 1)))

            gm_op = sparse.lil_matrix((dim, dim))
            gm_op.setdiag(diag)

    else:
        e_mat = sparse.lil_matrix((dim, dim))
        e_mat[ind_1, ind_2] = 1
        if ind_1 < ind_2:
            gm_op = e_mat + e_mat.conj().T
        else:
            gm_op = 1j * e_mat - 1j * e_mat.conj().T

    if not is_sparse:
        return gm_op.todense()
    return gm_op


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""
    Produce generalized Pauli operator [WikGenPaul]_.

    Generates a `dim`-by-`dim` unitary operator. More specifically, it is the
    operator `X^k_1*Z^k_2`, where X and Z are the "shift" and "clock" operators
    that naturally generalize the Pauli X and Z operators. These matrices span
    the entire space of `dim`-by-`dim` matrices as `k_1` and `k_2` range from 0
    to `dim-1`, inclusive.

    Examples
    ==========

    The generalized Pauli operator for `k_1 = 1`, `k_2 = 0` and `dim = 2` is
    given as the standard Pauli-X matrix

    .. math::
        G_{1, 0, 2} = \begin{pmatrix}
                         0 & 1 \\
                         1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.

    >>> from toqito.matrices import gen_pauli
    >>> dim = 2
    >>> k_1 = 1
    >>> k_2 = 0
    >>> gen_pauli(k_1, k_2, dim)
    [[0.+0.j, 1.+0.j],
     [1.+0.j, 0.+0.j]])

    The generalized Pauli matrix `k_1 = 1`, `k_2 = 1`, and `dim = 2` is given as
    the standard Pauli-Y matrix

    .. math::
        G_{1, 1, 2} = \begin{pmatrix}
                        0 & -1 \\
                        1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.`

    >>> from toqito.matrices import gen_pauli
    >>> dim = 2
    >>> k_1 = 1
    >>> k_2 = 1
    >>> gen_pauli(k_1, k_2, dim)
    [[ 0.+0.0000000e+00j, -1.+1.2246468e-16j],
     [ 1.+0.0000000e+00j,  0.+0.0000000e+00j]])

    References
    ==========
    .. [WikGenPaul] Wikipedia: Generalizations of Pauli matrices
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    :param k_1: (a non-negative integer from 0 to `dim-1` inclusive).
    :param k_2: (a non-negative integer from 0 to `dim-1` inclusive).
    :param dim: (a positive integer indicating the dimension).
    :return: A generalized Pauli operator.
    """
    gen_pauli_x = shift(dim)
    gen_pauli_z = clock(dim)

    gen_pauli_w = np.matmul(
        np.linalg.matrix_power(gen_pauli_x, k_1),
        np.linalg.matrix_power(gen_pauli_z, k_2),
    )

    return gen_pauli_w


def hadamard(n_param: int = 1) -> np.ndarray:
    r"""
    Produce a 2^{n_param} dimensional Hadamard matrix [WikHad]_.

    The standard Hadamard matrix that is often used in quantum information as a
    two-qubit quantum gate is defined as

    .. math::
        H_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
                                    1 & 1 \\
                                    1 & -1
                                 \end{pmatrix}

    In general, the Hadamard matrix of dimension 2^{n_param} may be defined as

    .. math::
        \left( H_n \right)_{i, j} = \frac{1}{2^{\frac{n}{2}}
        \left(-1\right)^{i \dot j}

    Examples
    ==========

    The standard 2-qubit Hadamard matrix can be generated in `toqito` as

    >>> from toqito.matrices import hadamard
    >>> hadamard(1)
    [[ 0.70710678  0.70710678]
     [ 0.70710678 -0.70710678]]

    References
    ==========
    .. [WikHad] Wikipedia: Hadamard transform
        https://en.wikipedia.org/wiki/Hadamard_transform

    :param n_param: A non-negative integer (default = 1).
    :return: The Hadamard matrix of dimension `2^{n_param}`.
    """
    if n_param == 0:
        return np.array([1])
    if n_param == 1:
        return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    if n_param == 2:
        return (
            1
            / 2
            * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
        )
    if n_param > 1:
        mat_1 = hadamard(1)
        mat_2 = hadamard(1)
        mat = np.kron(mat_1, mat_2)
        for _ in range(2, n_param):
            mat_1 = mat_2
            mat_2 = mat
            mat = np.kron(mat_1, mat_2)
        return mat
    raise ValueError(f"Improper dimension {n_param} provided.")


def iden(dim: int, is_sparse: bool = False) -> np.ndarray:
    r"""
    Calculate the `dim`-by-`dim` identity matrix [WIKID]_.

    Returns the `dim`-by-`dim` identity matrix. If `is_sparse = False` then
    the matrix will be full. If `is_sparse = True` then the matrix will be
    sparse.

    .. math::
        \mathbb{I} = \begin{pmatrix}
                        1 & 0 & 0 & \ldots & 0 \\
                        0 & 1 & 0 & \ldots & 0 \\
                        0 & 0 & 1 & \ldots & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots \\
                        0 & 0 & 0 & \ldots & 1
                   \end{pmatrix}

    Only use this function within other functions to easily get the correct
    identity matrix. If you always want either the full or the sparse
    identity matrix, just use numpy's built-in np.identity function.

    Examples
    ==========

    The identity matrix generated from :math:`d = 3` yields the following
    matrix:

    .. math::
        \mathbb{I}_3 = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1
        \end{pmatrix}

    >>> from toqito.matrices import iden
    >>> iden(3)
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]])

    It is also possible to create sparse identity matrices. The sparse identity
    matrix generated from :math:`d = 10` yields the following matrix:

    >>> from toqito.matrices import iden
    >>> iden(10, True)
    <10x10 sparse matrix of type '<class 'numpy.float64'>' with 10 stored
    elements (1 diagonals) in DIAgonal format>

    References
    ==========
    .. [WIKID] Wikipedia: Identity matrix
        https://en.wikipedia.org/wiki/Identity_matrix

    :param dim: Integer representing dimension of identity matrix.
    :param is_sparse: Whether or not the matrix is sparse.
    :return: Sparse identity matrix of dimension `dim`.
    """
    if is_sparse:
        id_mat = sparse.eye(dim)
    else:
        id_mat = np.identity(dim)
    return id_mat


def pauli(
    ind: Union[int, str, List[int], List[str]], is_sparse: bool = False
) -> Union[np.ndarray, sparse.csr_matrix]:
    r"""
    Produce a Pauli operator [WikPauli]_.

    Provides the 2-by-2 Pauli matrix indicated by the value of `ind`. The
    variable `ind = 1` gives the Pauli-X operator, `ind = 2` gives the Pauli-Y
    operator, `ind =3` gives the Pauli-Z operator, and `ind = 0`gives the
    identity operator. Alternatively, `ind` can be set to "I", "X", "Y", or "Z"
    (case insensitive) to indicate the Pauli identity, X, Y, or Z operator.

    The 2-by-2 Pauli matrices are defined as the following matrices:

    .. math::

        \begin{equation}
            \begin{aligned}
                X = \begin{pmatrix}
                        0 & 1 \\
                        1 & 0
                    \end{pmatrix}, \quad
                Y = \begin{pmatrix}
                        0 & -i \\
                        i & 0
                    \end{pmatrix}, \quad
                Z = \begin{pmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{pmatrix}, \quad
                I = \begin{pmatrix}
                        1 & 0 \\
                        0 & 1
                    \end{pmatrix}.
                \end{aligned}
            \end{equation}

    Examples
    ==========

    Example for identity Pauli matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("I")
    [[1., 0.],
     [0., 1.]])

    Example for Pauli-X matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("X")
    [[0, 1],
     [1, 0]])

    Example for Pauli-Y matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Y")
    [[ 0.+0.j, -0.-1.j],
     [ 0.+1.j,  0.+0.j]])

    Example for Pauli-Z matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Z")
    [[ 1,  0],
     [ 0, -1]])

    References
    ==========
    .. [WikPauli] Wikipedia: Pauli matrices
        https://en.wikipedia.org/wiki/Pauli_matrices

    :param ind: The index to indicate which Pauli operator to generate.
    :param is_sparse: Returns a sparse matrix if set to True and a non-sparse
                      matrix if set to False.
    """
    if isinstance(ind, (int, str)):
        if ind in ("x", "X", 1):
            pauli_mat = np.array([[0, 1], [1, 0]])
        elif ind in ("y", "Y", 2):
            pauli_mat = np.array([[0, -1j], [1j, 0]])
        elif ind in ("z", "Z", 3):
            pauli_mat = np.array([[1, 0], [0, -1]])
        else:
            pauli_mat = np.identity(2)

        if is_sparse:
            pauli_mat = sparse.csr_matrix(pauli_mat)

        return pauli_mat

    num_qubits = len(ind)
    pauli_mats = []
    for i in range(num_qubits - 1, -1, -1):
        pauli_mats.append(pauli(ind[i], is_sparse))
    return tensor(pauli_mats)


def shift(dim: int) -> np.ndarray:
    r"""
    Produce a `dim`-by-`dim` shift matrix [WikShift]_.

    Returns the shift matrix of dimension `dim` described in [WikShift]_. The
    shift matrix generates the following `dim`-by-`dim` matrix:

    .. math::
        \Sigma_{1, d} = \begin{pmatrix}
                        0 & 0 & 0 & \ldots & 0 & 1 \\
                        1 & 0 & 0 & \ldots & 0 & 0 \\
                        0 & 1 & 0 & \ldots & 0 & 0 \\
                        0 & 0 & 1 & \ldots & 0 & 0 \\
                        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
                        0 & 0 & 0 & \ldots & 1 & 0
                    \end{pmatrix}

    The shift matrix is primarily used in the construction of the generalized
    Pauli operators.

    Examples
    ==========

    The shift matrix generated from :math:`d = 3` yields the following matrix:

    .. math::
        \Sigma_{1, 3} =
        \begin{pmatrix}
            0 & 0 & 1 \\
            1 & 0 & 0 \\
            0 & 1 & 0
        \end{pmatrix}

    >>> from toqito.matrices import shift
    >>> shift(3)
    [[0., 0., 1.],
     [1., 0., 0.],
     [0., 1., 0.]]

    References
    ==========
    .. [WikShift] Wikipedia: Generalizations of Pauli matrices
        (https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices.

    :param dim: Dimension of the matrix.
    :return: `dim`-by-`dim` shift matrix.
    """
    shift_mat = np.identity(dim)
    shift_mat = np.roll(shift_mat, -1)
    shift_mat[:, -1] = np.array([0] * dim)
    shift_mat[0, -1] = 1

    return shift_mat
