"""Quantum states."""
from typing import List, Union

import itertools
import numpy as np

from scipy import sparse

from toqito.matrices import gen_pauli
from toqito.matrix_ops import vec
from toqito.matrices import iden
from toqito.perms import permutation_operator
from toqito.perms import swap_operator


__all__ = [
    "basis",
    "bell",
    "chessboard",
    "domino",
    "gen_bell",
    "ghz",
    "gisin",
    "horodecki",
    "isotropic",
    "max_entangled",
    "max_mixed",
    "tile",
    "w_state",
    "werner",
]


def basis(dim: int, pos: int) -> np.ndarray:
    r"""

    Obtain the ket of dimension `dim` [WikKet]_.

    Examples
    ==========

    The standard basis bra vectors given as :math:`|0\rangle` and
    :math:`|1\rangle` where

    .. math::
        |0 \rangle = \left[1, 0 \right]^{\text{T}} \quad \text{and} \quad
        |1\rangle = \left[0, 1 \right]^{\text{T}},

    can be obtained in `toqito` as follows.

    Example:  Ket vector: :math:`| 0 \rangle`.

    >>> from toqito.states import basis
    >>> basis(2, 0)
    [[1]
    [0]]

    Example:  Ket vector: :math:`| 1 \rangle`.

    >>> from toqito.states import basis
    >>> basis(2, 1)
    [[0]
    [1]]

    References
    ==========
    .. [WikKet] Wikipedia page for bra–ket notation:
           https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation

    :param dim: The dimension of the column vector.
    :param pos: The position in which to place a 1.
    :return: The column vector of dimension `dim` with all entries set to `0`
             except the entry at position `1`.
    """
    if pos >= dim:
        raise ValueError(
            "Invalid: The `pos` variable needs to be less than "
            "`dim` for ket function."
        )

    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    ret = ret.conj().T.reshape(-1, 1)
    return ret


def bell(idx: int) -> np.ndarray:
    r"""
    Produce a Bell state [WikBell]_.

    Returns one of the following four Bell states depending on the value
    of `idx`:

    .. math::
        \begin{equation}
            \begin{aligned}
                \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) &
                \qquad &
                \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right) \\
                \frac{1}{\sqrt{2}} \left( |01 \rangle + |10 \rangle \right) &
                \qquad &
                \frac{1}{\sqrt{2}} \left( |01 \rangle - |10 \rangle \right)
            \end{aligned}
        \end{equation}


    Examples
    ==========

    When `idx = 0`, this produces the following Bell state

    .. math::
        \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    Using `toqito`, we can see that this yields the proper state.

    >>> from toqito.states import bell
    >>> import numpy as np
    >>> bell(0)
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.70710678]]

    References
    ==========
    .. [WikBell] Wikipedia: Bell state
        https://en.wikipedia.org/wiki/Bell_state

    :param idx: A parameter in [0, 1, 2, 3]
    :return: Bell state with index `idx`.
    """
    e_0, e_1 = basis(2, 0), basis(2, 1)
    if idx == 0:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
    if idx == 1:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_0) - np.kron(e_1, e_1))
    if idx == 2:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_1) + np.kron(e_1, e_0))
    if idx == 3:
        return 1 / np.sqrt(2) * (np.kron(e_0, e_1) - np.kron(e_1, e_0))
    raise ValueError("Invalid integer value for Bell state.")


def chessboard(
    mat_params: List[float], s_param: float = None, t_param: float = None
) -> np.ndarray:
    r"""
    Produce a chessboard state [BP00]_.

    Generates the chessboard state defined in [BP00]_. Note that, for certain
    choices of S and T, this state will not have positive partial transpose,
    and thus may not be bound entangled.

    Examples
    ==========

    The standard chessboard state can be invoked using `toqito` as

    >>> from toqito.states import chessboard
    >>> chessboard([1, 2, 3, 4, 5, 6], 7, 8)
    [[ 0.22592593,  0.        ,  0.12962963,  0.        ,  0.        ,
       0.        ,  0.17777778,  0.        ,  0.        ],
     [ 0.        ,  0.01851852,  0.        ,  0.        ,  0.        ,
       0.01111111,  0.        ,  0.02962963,  0.        ],
     [ 0.12962963,  0.        ,  0.18148148,  0.        ,  0.15555556,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.01851852,  0.        ,
       0.02222222,  0.        , -0.01481481,  0.        ],
     [ 0.        ,  0.        ,  0.15555556,  0.        ,  0.22592593,
       0.        , -0.14814815,  0.        ,  0.        ],
     [ 0.        ,  0.01111111,  0.        ,  0.02222222,  0.        ,
       0.03333333,  0.        ,  0.        ,  0.        ],
     [ 0.17777778,  0.        ,  0.        ,  0.        , -0.14814815,
       0.        ,  0.23703704,  0.        ,  0.        ],
     [ 0.        ,  0.02962963,  0.        , -0.01481481,  0.        ,
       0.        ,  0.        ,  0.05925926,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ]]

    References
    ==========
    .. [BP00] Three qubits can be entangled in two inequivalent ways.
        D. Bruss and A. Peres
        Phys. Rev. A, 61:30301(R), 2000
        arXiv: 991.1056

    :param mat_params:
    :param s_param:
    :param t_param:
    :return:
    """
    if s_param is None:
        s_param = np.conj(mat_params[2]) / np.conj(mat_params[5])
    if t_param is None:
        t_param = mat_params[0] * mat_params[3] / mat_params[4]

    v_1 = np.array([[mat_params[4], 0, s_param, 0, mat_params[5], 0, 0, 0, 0]])

    v_2 = np.array([[0, mat_params[0], 0, mat_params[1], 0, mat_params[2], 0, 0, 0]])

    v_3 = np.array(
        [[np.conj(mat_params[5]), 0, 0, 0, -np.conj(mat_params[4]), 0, t_param, 0, 0]]
    )

    v_4 = np.array(
        [
            [
                0,
                np.conj(mat_params[1]),
                0,
                -np.conj(mat_params[0]),
                0,
                0,
                0,
                mat_params[3],
                0,
            ]
        ]
    )

    rho = (
        v_1.conj().T * v_1
        + v_2.conj().T * v_2
        + v_3.conj().T * v_3
        + v_4.conj().T * v_4
    )
    return rho / np.trace(rho)


def domino(idx: int) -> np.ndarray:
    r"""
    Produce a domino state [CBDOM99]_, [UPB99]_.

    The orthonormal product basis of domino states is given as

    .. math::

        \begin{equation}
            \begin{aligned}
            |\phi_0\rangle = |1\rangle
                            |1 \rangle \qquad
            |\phi_1\rangle = |0 \rangle
                            \left(\frac{|0 \rangle + |1 \rangle}{\sqrt{2}}
                            \right) & \qquad
            |\phi_2\rangle = |0\rangle
                            \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            \\
            |\phi_3\rangle = |2\rangle
                            \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            \qquad
            |\phi_4\rangle = |2\rangle
                            \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            & \qquad
            |\phi_5\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            |0\rangle \\
            |\phi_6\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            |0\rangle \qquad
            |\phi_7\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)
                            |2\rangle & \qquad
            |\phi_8\rangle = \left(\frac{|0\rangle - |1\rangle}{\sqrt{2}}\right)
                            |2\rangle
            \end{aligned}
        \end{equation}

    Returns one of the following nine domino states depending on the value
    of `idx`.

    Examples
    ==========

    When `idx = 0`, this produces the following Domino state

    .. math::
        |\phi_0 \rangle = |11 \rangle |11 \rangle.

    Using `toqito`, we can see that this yields the proper state.

    >>> from toqito.states import domino
    >>> domino(0)
    [[0],
     [0],
     [0],
     [0],
     [1],
     [0],
     [0],
     [0],
     [0]]

    When `idx = 3`, this produces the following Domino state

    .. math::
        |\phi_3\rangle = |2\rangle \left(\frac{|0\rangle + |1\rangle}
        {\sqrt{2}}\right)

    Using `toqito`, we can see that this yields the proper state.

    >>> from toqito.states import domino
    >>> domino(3)
    [[0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.70710678],
     [0.70710678]]

    References
    ==========
    .. [CBDOM99] Bennett, Charles H., et al.
        Quantum nonlocality without entanglement.
        Phys. Rev. A, 59:1070–1091, Feb 1999.
        https://arxiv.org/abs/quant-ph/9804053

    .. [UPB99] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.
        https://arxiv.org/abs/quant-ph/9808030

    :param idx: A parameter in [0, 1, 2, 3, 4, 5, 6, 7, 8]
    :return: Domino state of index `idx`.
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    if idx == 0:
        return np.kron(e_1, e_1)
    if idx == 1:
        return np.kron(e_0, 1 / np.sqrt(2) * (e_0 + e_1))
    if idx == 2:
        return np.kron(e_0, 1 / np.sqrt(2) * (e_0 - e_1))
    if idx == 3:
        return np.kron(e_2, 1 / np.sqrt(2) * (e_1 + e_2))
    if idx == 4:
        return np.kron(e_2, 1 / np.sqrt(2) * (e_1 - e_2))
    if idx == 5:
        return np.kron(1 / np.sqrt(2) * (e_1 + e_2), e_0)
    if idx == 6:
        return np.kron(1 / np.sqrt(2) * (e_1 - e_2), e_0)
    if idx == 7:
        return np.kron(1 / np.sqrt(2) * (e_0 + e_1), e_2)
    if idx == 8:
        return np.kron(1 / np.sqrt(2) * (e_0 - e_1), e_2)
    raise ValueError("Invalid integer value for Domino state.")


def gen_bell(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""
    Produce a generalized Bell state [DL09]_.

    Produces a generalized Bell state. Note that the standard Bell states
    can be recovered as:

    bell(0) -> gen_bell(0, 0, 2)
    bell(1) -> gen_bell(0, 1, 2)
    bell(2) -> gen_bell(1, 0, 2)
    bell(3) -> gen_bell(1, 1, 2)

    Examples
    ==========

    For :math:`d = 2` and :math:`k_1 = k_2 = 0`, this generates the following
    matrix

    .. math::
        \frac{1}{2} \begin{pmatrix}
                        1 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 \\
                        1 & 0 & 0 & 1
                    \end{pmatrix}

    which is equivalent to :math:`|\phi_0 \rangle \langle \phi_0 |` where

    .. math::
        |\phi_0\rangle = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle
        \right)

    is one of the four standard Bell states. This can be computed via `toqito`
    as follows.

    >>> from toqito.states import gen_bell
    >>> dim = 2
    >>> k_1 = 0
    >>> k_2 = 0
    >>> gen_bell(k_1, k_2, dim)
    [[0.5+0.j, 0. +0.j, 0. +0.j, 0.5+0.j],
     [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
     [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
     [0.5+0.j, 0. +0.j, 0. +0.j, 0.5+0.j]]

    It is possible for us to consider higher dimensional Bell states. For
    instance, we can consider the :math:`3`-dimensional Bell state for
    :math:`k_1 = k_2 = 0` as follows.

    >>> from toqito.states import gen_bell
    >>> dim = 3
    >>> k_1 = 0
    >>> k_2 = 0
    >>> gen_bell(k_1, k_2, dim)
    [[0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.        +0.j],
     [0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j, 0.        +0.j, 0.        +0.j, 0.        +0.j,
      0.33333333+0.j]]

    References
    ==========
    .. [DL09] Sych, Denis, and Gerd Leuchs.
        "A complete basis of generalized Bell states."
        New Journal of Physics 11.1 (2009): 013006.

    :param k_1: An integer 0 <= k_1 <= n.
    :param k_2: An integer 0 <= k_2 <= n.
    :param dim: The dimension of the generalized Bell state.
    """
    gen_pauli_w = gen_pauli(k_1, k_2, dim)
    return 1 / dim * vec(gen_pauli_w) * vec(gen_pauli_w).conj().T


def ghz(dim: int, num_qubits: int, coeff: List[int] = None) -> sparse:
    r"""
    Generate a (generalized) GHZ state [GHZ07]_.

    Returns a `num_qubits`-partite GHZ state acting on `dim` local dimensions,
    described in [GHZ07]_. For example, `ghz(2, 3)` returns the standard
    3-qubit GHZ state on qubits. The output of this function is sparse.

    For a system of `num_qubits` qubits (i.e., `dim = 2`), the GHZ state can be
    written as

    .. math::
        |GHZ \rangle = \frac{1}{\sqrt{n}} \left(|0\rangle^{\otimes n} +
        |1 \rangle^{\otimes n} \right))

    Examples
    ==========

    When `dim = 2`, and `num_qubits = 3` this produces the standard GHZ state

    .. math::
        \frac{1}{\sqrt{2}} \left( |000 \rangle + |111 \rangle \right).

    Using `toqito`, we can see that this yields the proper state.

    >>> from toqito.states import ghz
    >>> ghz(2, 3).toarray()
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.70710678]]

    As this function covers the generalized GHZ state, we can consider higher
    dimensions. For instance here is the GHZ state in
    :math:`\mathbb{C^4}^{\otimes 7}` as

    .. math::
        \frac{1}{\sqrt{30}} \left( |0000000 \rangle 2|1111111 \rangle +
        3|2222222 \rangle + 4|3333333\rangle \right)

    Using `toqito`, we can see this generates the appropriate generalized GHZ
    state.

    >>> from toqito.states import ghz
    >>> ghz(4, 7, np.array([1, 2, 3, 4]) / np.sqrt(30)).toarray()
    [[0.18257419],
     [0.        ],
     [0.        ],
     ...,
     [0.        ],
     [0.        ],
     [0.73029674]])

    References
    ==========
    .. [GHZ07] Going beyond Bell's theorem.
        D. Greenberger and M. Horne and A. Zeilinger.
        E-print: [quant-ph] arXiv:0712.0921. 2007.

    :param dim: The local dimension.
    :param num_qubits: The number of parties (qubits/qudits)
    :param coeff: (default `[1, 1, ..., 1])/sqrt(dim)`:
                  a 1-by-`dim` vector of coefficients.
    :returns: Numpy vector array as GHZ state.
    """
    if coeff is None:
        coeff = np.ones(dim) / np.sqrt(dim)

    # Error checking:
    if dim < 2:
        raise ValueError("InvalidDim: `dim` must be at least 2.")
    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if len(coeff) != dim:
        raise ValueError(
            "InvalidCoeff: The variable `coeff` must be a vector"
            " of length equal to `dim`."
        )

    # Construct the state (and do it in a way that is less memory-intensive
    # than naively tensoring things together.
    dim_sum = 1
    for i in range(1, num_qubits):
        dim_sum += dim ** i

    ret_ghz_state = sparse.lil_matrix((dim ** num_qubits, 1))
    for i in range(1, dim + 1):
        ret_ghz_state[(i - 1) * dim_sum] = coeff[i - 1]
    return ret_ghz_state


def gisin(lambda_var: float, theta: float) -> np.ndarray:
    r"""
    Produce a Gisin state [GIS96]_.

    Returns the Gisin state described in [GIS96]_.

    Specifically, the Gisin state can be defined as:

    .. math::

        \begin{equation}
            \rho_{\lambda, \theta} = \lambda
                                    \begin{pmatrix}
                                        0 & 0 & 0 & 0 \\
                                        0 & \sin^2(\theta) &
                                        -\sin(\theta)\cos(\theta) & 0 \\
                                        0 & -\sin(\theta)\cos(\theta) &
                                        \cos^2(\theta) & 0 \\
                                        0 & 0 & 0 & 0
                                    \end{pmatrix} +
                                    \frac{1 - \lambda}{2}
                                    \begin{pmatrix}
                                        1 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 1
                                    \end{pmatrix}
        \end{equation}

    Examples
    ==========

    The following code generates the Gisin state :math:`\rho_{0.5, 1}`.

    >>> from toqito.states import gisin
    >>> gisin(0.5, 1)
    [[ 0.25      ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.35403671, -0.22732436,  0.        ],
     [ 0.        , -0.22732436,  0.14596329,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.25      ]]

    References
    ==========
    .. [GIS96] N. Gisin.
        Hidden quantum nonlocality revealed by local filters.
        (http://dx.doi.org/10.1016/S0375-9601(96)80001-6). 1996.

    :param lambda_var: A real parameter in [0, 1].
    :param theta: A real parameter.
    :return: Gisin state.
    """
    if lambda_var < 0 or lambda_var > 1:
        raise ValueError("InvalidLambda: Variable lambda must be between 0 and 1.")

    rho_theta = np.array(
        [
            [0, 0, 0, 0],
            [0, np.sin(theta) ** 2, -np.sin(2 * theta) / 2, 0],
            [0, -np.sin(2 * theta) / 2, np.cos(theta) ** 2, 0],
            [0, 0, 0, 0],
        ]
    )

    rho_uu_dd = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    return lambda_var * rho_theta + (1 - lambda_var) * rho_uu_dd / 2


def horodecki(a_param: float, dim: List[int] = None) -> np.ndarray:
    r"""
    Produce a Horodecki state [HOR]_, [CHR]_.

    Returns the Horodecki state in either :math:`(3 \otimes 3)`-dimensional
    space or :math:`(2 \otimes 4)`-dimensional space, depending on the
    dimensions in the 1-by-2 vector `dim`.

    The Horodecki state was introduced in [1] which serves as an example in
    :math:`\mathbb{C}^3 \otimes \mathbb{C}` or :math:`\mathbb{C}^2 \otimes
    \mathbb{C}^4` of an entangled state that is positive under partial
    transpose (PPT). The state is PPT for all :math:`a \in [0, 1]` and
    separable only for `a_param = 0` or `a_param = 1`.

    These states have the following definitions:

    .. math::

        \begin{equation}
            \rho_a^{3 \otimes 3} = \frac{1}{8a + 1}
            \begin{pmatrix}
                a & 0 & 0 & 0 & a & 0 & 0 & 0 & a \\
                0 & a & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & a & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & a & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & a & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & a & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & \frac{1}{2}
                \left( 1 + a \right) & 0 & \frac{1}{2} \sqrt{1 - a^2} \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & a & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & \frac{1}{2} \sqrt{1 - a^2} & 0
                & \frac{1}{2} \left(1 + a \right) \\
            \end{pmatrix}
        \end{equation}

    .. math::

        \begin{equation}
            \rho_a^{2 \otimes 4} = \frac{1}{7a + 1}
            \begin{pmatrix}
                a & 0 & 0 & 0 & 0 & a & 0 & 0  \\
                0 & a & 0 & 0 & 0 & 0 & a & 0 \\
                0 & 0 & a & 0 & 0 & 0 & 0 & a  \\
                0 & 0 & 0 & a & 0 & 0 & 0 & 0  \\
                0 & 0 & 0 & 0 & \frac{1}{2} \left(1 + a\right) & 0 & 0
                & \frac{1}{2}\sqrt{1 -a^2} \\
                a & 0 & 0 & 0 & 0 & a & 0 & 0 \\
                0 & a & 0 & 0 & 0 & 0 & a & 0 \\
                0 & 0 & a & 0 & \frac{1}{2}\sqrt{1 - a^2} & 0 & 0
                & \frac{1}{2}\left(1 +a \right)
            \end{pmatrix}
        \end{equation}

    Note: Refer to [CHR]_ (specifically equations (1) and (2)) for more
    information on this state and its properties. The 3x3 Horodecki state is
    defined explicitly in Section 4.1 of [HOR]_ and the 2x4 Horodecki state is
    defined explicitly in Section 4.2 of [HOR]_.

    Examples
    ==========

    The following code generates a Horodecki state in
    :math:`\mathbb{C}^3 \otimes \mathbb{C}^3`

    >>> from toqito.states import horodecki
    >>> horodecki(0.5, [3, 3])
    [[0.1       , 0.        , 0.        , 0.        , 0.1       ,
      0.        , 0.        , 0.        , 0.1       ],
     [0.        , 0.1       , 0.        , 0.        , 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.1       , 0.        , 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.1       , 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.1       , 0.        , 0.        , 0.        , 0.1       ,
      0.        , 0.        , 0.        , 0.1       ],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.1       , 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.        , 0.15      , 0.        , 0.08660254],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.        , 0.        , 0.1       , 0.        ],
     [0.1       , 0.        , 0.        , 0.        , 0.1       ,
      0.        , 0.08660254, 0.        , 0.15      ]]

    The following code generates a Horodecki state in
    :math:`\mathbb{C}^2 \otimes \mathbb{C}^4`

    >>> from toqito.states import horodecki
    >>> horodecki(0.5, [2, 4])
    [[0.11111111, 0.        , 0.        , 0.        , 0.        ,
      0.11111111, 0.        , 0.        ],
     [0.        , 0.11111111, 0.        , 0.        , 0.        ,
      0.        , 0.11111111, 0.        ],
     [0.        , 0.        , 0.11111111, 0.        , 0.        ,
      0.        , 0.        , 0.11111111],
     [0.        , 0.        , 0.        , 0.11111111, 0.        ,
      0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.        , 0.16666667,
      0.        , 0.        , 0.09622504],
     [0.11111111, 0.        , 0.        , 0.        , 0.        ,
      0.11111111, 0.        , 0.        ],
     [0.        , 0.11111111, 0.        , 0.        , 0.        ,
      0.        , 0.11111111, 0.        ],
     [0.        , 0.        , 0.11111111, 0.        , 0.09622504,
      0.        , 0.        , 0.16666667]]

    References
    ==========
    .. [HOR] P. Horodecki.
        Separability criterion and inseparable mixed states with positive
        partial transpose.
        arXiv: 970.3004.

    .. [CHR] K. Chruscinski.
        On the symmetry of the seminal Horodecki state.
        arXiv: 1009.4385.
    """
    if a_param < 0 or a_param > 1:
        raise ValueError("Invalid: Argument A_PARAM must be in the interval " "[0, 1].")

    if dim is None:
        dim = np.array([3, 3])

    if np.array_equal(dim, np.array([3, 3])):
        n_a_param = 1 / (8 * a_param + 1)
        b_param = (1 + a_param) / 2
        c_param = np.sqrt(1 - a_param ** 2) / 2

        horo_state = n_a_param * np.array(
            [
                [a_param, 0, 0, 0, a_param, 0, 0, 0, a_param],
                [0, a_param, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, a_param, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, a_param, 0, 0, 0, 0, 0],
                [a_param, 0, 0, 0, a_param, 0, 0, 0, a_param],
                [0, 0, 0, 0, 0, a_param, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, b_param, 0, c_param],
                [0, 0, 0, 0, 0, 0, 0, a_param, 0],
                [a_param, 0, 0, 0, a_param, 0, c_param, 0, b_param],
            ]
        )
        return horo_state

    if np.array_equal(dim, np.array([2, 4])):
        n_a_param = 1 / (7 * a_param + 1)
        b_param = (1 + a_param) / 2
        c_param = np.sqrt(1 - a_param ** 2) / 2

        horo_state = n_a_param * np.array(
            [
                [a_param, 0, 0, 0, 0, a_param, 0, 0],
                [0, a_param, 0, 0, 0, 0, a_param, 0],
                [0, 0, a_param, 0, 0, 0, 0, a_param],
                [0, 0, 0, a_param, 0, 0, 0, 0],
                [0, 0, 0, 0, b_param, 0, 0, c_param],
                [a_param, 0, 0, 0, 0, a_param, 0, 0],
                [0, a_param, 0, 0, 0, 0, a_param, 0],
                [0, 0, a_param, 0, c_param, 0, 0, b_param],
            ]
        )
        return horo_state
    raise ValueError("InvalidDim: DIM must be one of [3, 3], or [2, 4].")


def isotropic(dim: int, alpha: float) -> np.ndarray:
    r"""
    Produce a isotropic state [HH99]_.

    Returns the isotropic state with parameter `alpha` acting on
    (`dim`-by-`dim`)-dimensional space. More specifically, the state is the
    density operator defined by `(1-alpha)*I(dim)/dim**2 + alpha*E`, where I is
    the identity operator and E is the projection onto the standard
    maximally-entangled pure state on two copies of `dim`-dimensional space.

    The isotropic state has the following form

    .. math::

        \begin{equation}
            \rho_{\alpha} = \frac{1 - \alpha}{d^2} \mathbb{I} \otimes
            \mathbb{I} + \alpha |\psi_+ \rangle \langle \psi_+ | \in
            \mathbb{C}^d \otimes \mathbb{C}^2
        \end{equation}

    where :math:`|\psi_+ \rangle = \frac{1}{\sqrt{d}} \sum_j |j \rangle \otimes
    |j \rangle` is the maximally entangled state.

    Examples
    ==========

    To generate the isotropic state with parameter :math:`\alpha=1/2`, we can
    make the following call to `toqito` as

    >>> from toqito.states import isotropic
    >>> isotropic(3, 1 / 2)
    [[0.22222222, 0.        , 0.        , 0.        , 0.16666667,
      0.        , 0.        , 0.        , 0.16666667],
     [0.        , 0.05555556, 0.        , 0.        , 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.05555556, 0.        , 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.05555556, 0.        ,
      0.        , 0.        , 0.        , 0.        ],
     [0.16666667, 0.        , 0.        , 0.        , 0.22222222,
      0.        , 0.        , 0.        , 0.16666667],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.05555556, 0.        , 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.        , 0.05555556, 0.        , 0.        ],
     [0.        , 0.        , 0.        , 0.        , 0.        ,
      0.        , 0.        , 0.05555556, 0.        ],
     [0.16666667, 0.        , 0.        , 0.        , 0.16666667,
      0.        , 0.        , 0.        , 0.22222222]]

    References
    ==========
    .. [HH99] Horodecki, Michał, and Paweł Horodecki.
        "Reduction criterion of separability and limits for a class of
        distillation protocols." Physical Review A 59.6 (1999): 4206.

    :param dim: The local dimension.
    :param alpha: The parameter of the isotropic state.
    :return: Isotropic state.
    """
    # Compute the isotropic state.
    psi = max_entangled(dim, True, False)
    return (1 - alpha) * sparse.identity(
        dim ** 2
    ) / dim ** 2 + alpha * psi * psi.conj().T / dim


def max_entangled(
    dim: int, is_sparse: bool = False, is_normalized: bool = True
) -> [np.ndarray, sparse.dia.dia_matrix]:
    r"""
    Produce a maximally entangled bipartite pure state [WikEnt]_.

    Produces a maximally entangled pure state as above that is sparse
    if `is_sparse = True` and is full if `is_sparse = False`. The pure state
    is normalized to have Euclidean norm 1 if `is_normalized = True`, and it
    is unnormalized (i.e. each entry in the vector is 0 or 1 and the
    Euclidean norm of the vector is `sqrt(dim)` if `is_normalized = False`.

    Examples
    ==========

    We can generate the canonical :math:`2`-dimensional maximally entangled
    state

    .. math::
        \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right)

    using `toqito` as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(2)
    [[0.70710678],
     [0.        ],
     [0.        ],
     [0.70710678]]

    By default, the state returned in normalized, however we can generate the
    unnormalized state

    .. math::
        |00\rangle + |11 \rangle

    using `toqito` as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(2, False, False)
    [[1.],
     [0.],
     [0.],
     [1.]]

    References
    ==========
    .. [WikEnt] Wikipedia: Quantum entanglement
        https://en.wikipedia.org/wiki/Quantum_entanglement

    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is spare and `False` otherwise.
    :param is_normalized: `True` if vector is normalized and `False` otherwise.
    :return: The maximally entangled state of dimension `dim`.
    """
    psi = np.reshape(iden(dim, is_sparse), (dim ** 2, 1))
    if is_normalized:
        psi = psi / np.sqrt(dim)
    return psi


def max_mixed(dim: int, is_sparse: bool = False) -> [np.ndarray, sparse.dia.dia_matrix]:
    r"""
    Produce the maximally mixed state [AAR6]_.

    Produces the maximally mixed state on of `dim` dimensions. The maximally
    mixed state is defined as

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

    for some complex Euclidean space :math:`\mathcal{X}`. The maximally mixed
    state is sometimes also referred to as the tracial state.

    The maximally mixed state is returned as a sparse matrix if
    `is_sparse = True` and is full if `is_sparse = False`.

    Examples
    ==========

    Using `toqito`, we can generate the :math:`2`-dimensional maximally mixed
    state

    .. math::
        \frac{1}{2}
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}

    as follows.

    >>> from toqito.states import max_mixed
    >>> max_mixed(2, is_sparse=False)
    [[0.5, 0. ],
     [0. , 0.5]]

    One may also generate a maximally mixed state returned as a sparse matrix

    >>> from toqito.states import max_mixed
    >>> max_mixed(2, is_sparse=True)
        <2x2 sparse matrix of type '<class 'numpy.float64'>'
        with 2 stored elements (1 diagonals) in DIAgonal format>

    References
    ==========
    .. [AAR6] Scott Aaronson: Lecture 6, Thurs Feb 2: Mixed States
        https://www.scottaaronson.com/qclec/6.pdf

    :param dim: Dimension of the entangled state.
    :param is_sparse: `True` if vector is spare and `False` otherwise.
    :return: The maximally mixed state of dimension `dim`.
    """
    if is_sparse:
        return 1 / dim * sparse.eye(dim)
    return 1 / dim * np.eye(dim)


def tile(idx: int) -> np.ndarray:
    r"""
    Produce a Tile state [UPBTile99]_.

    The Tile states constitute five states on 3-by-3 dimensional space that form
    a UPB (unextendible product basis).

    Returns one of the following five Tile states depending on the value
    of `idx`:

    .. math::
        \begin{equation}
            \begin{aligned}
                |\psi_0 \rangle = \frac{1}{\sqrt{2}} |0 \rangle
                \left(|0\rangle - |1\rangle \right) &
                \qquad &
                |\psi_1\rangle = \frac{1}{\sqrt{2}}
                \left(|0\rangle - |1\rangle \right) |2\rangle \\
                |\psi_2\rangle = \frac{1}{\sqrt{2}} |2\rangle
                \left(|1\rangle - |2\rangle \right) &
                \qquad &
                |\psi_3\rangle = \frac{1}{\sqrt{2}}
                \left(|1\rangle - |2\rangle \right) |0\rangle \\
                |\psi_4\rangle = \frac{1}{3}
                \left(|0\rangle + |1\rangle + |2\rangle)\right)
                \left(|0\rangle + |1\rangle + |2\rangle.
            \end{aligned}
        \end{equation}


    Examples
    ==========

    When `idx = 0`, this produces the following Tile state

    .. math::
        \frac{1}{\sqrt{2}} |0\rangle \left( |0\rangle - |1\rangle \right).

    Using `toqito`, we can see that this yields the proper state.

    >>> from toqito.states import tile
    >>> import numpy as np
    >>> tile(0)
    [[ 0.70710678]
     [-0.        ]
     [ 0.        ]]

    References
    ==========
    .. [UPBTile99] Bennett, Charles H., et al.
        "Unextendible product bases and bound entanglement."
        Physical Review Letters 82.26 (1999): 5385.
        https://arxiv.org/abs/quant-ph/9808030

    :param idx: A parameter in [0, 1, 2, 3, 4]
    :return: Tile state.
    """
    e_0, e_1, e_2 = basis(3, 0), basis(3, 1), basis(3, 2)
    if idx == 0:
        return 1 / np.sqrt(2) * e_0 * (e_0 - e_1)
    if idx == 1:
        return 1 / np.sqrt(2) * (e_0 - e_1) * e_2
    if idx == 2:
        return 1 / np.sqrt(2) * e_2 * (e_1 - e_2)
    if idx == 3:
        return 1 / np.sqrt(2) * (e_1 - e_2) * e_0
    if idx == 4:
        return 1 / 3 * (e_0 + e_1 + e_2) * (e_0 + e_1 + e_2)
    raise ValueError("Invalid integer value for Tile state.")


def w_state(num_qubits: int, coeff: List[int] = None) -> np.ndarray:
    r"""
    Produce a W-state [DVC00]_.

    Returns the W-state described in [DVC00]. The W-state on `num_qubits` qubits
    is defined by:

    .. math::
        |W \rangle = \frac{1}{\sqrt{num\_qubits}}
        \left(|100 \ldots 0 \rangle + |010 \ldots 0 \rangle + \ldots +
        |000 \ldots 1 \rangle \right).

    Examples
    ==========

    Using `toqito`, we can generate the :math:`3`-qubit W-state

    .. math::
        |W_3 \rangle = \frac{1}{\sqrt{3}} \left( |100\rangle + |010 \rangle +
        |001 \rangle \right)

    as follows.

    >>> from toqito.states import w_state
    >>> w_state(3)
    [[0.    ],
     [0.5774],
     [0.5774],
     [0.    ],
     [0.5774],
     [0.    ],
     [0.    ],
     [0.    ]]

    We may also generate a generalized :math:`W`-state. For instance, here is a
    :math:`4`-dimensional :math:`W`-state

    .. math::
        \frac{1}{\sqrt{30}} \left( |1000 \rangle + 2|0100 \rangle + 3|0010
        \rangle + 4 |0001 \rangle \right)

    We can generate this state in `toqito` as

    >>> from toqito.states import w_state
    >>> import numpy as np
    >>> coeffs = np.array([1, 2, 3, 4]) / np.sqrt(30)
    >>> w_state(4, coeffs)
    [[0.    ],
     [0.7303],
     [0.5477],
     [0.    ],
     [0.3651],
     [0.    ],
     [0.    ],
     [0.    ],
     [0.1826],
     [0.    ],
     [0.    ],
     [0.    ],
     [0.    ],
     [0.    ],
     [0.    ],
     [0.    ]]

    References
    ==========
    .. [DVC00] Three qubits can be entangled in two inequivalent ways.
        W. Dur, G. Vidal, and J. I. Cirac.
        E-print: arXiv:quant-ph/0005115, 2000.

    :param num_qubits: An integer representing the number of qubits.
    :param coeff: default is `[1, 1, ..., 1]/sqrt(num_qubits)`: a
                  1-by-`num_qubts` vector of coefficients.
    """
    if coeff is None:
        coeff = np.ones(num_qubits) / np.sqrt(num_qubits)

    if num_qubits < 2:
        raise ValueError("InvalidNumQubits: `num_qubits` must be at least 2.")
    if len(coeff) != num_qubits:
        raise ValueError(
            "InvalidCoeff: The variable `coeff` must be a vector "
            "of length equal to `num_qubits`."
        )

    ret_w_state = sparse.csr_matrix((2 ** num_qubits, 1)).toarray()

    for i in range(num_qubits):
        ret_w_state[2 ** i] = coeff[num_qubits - i - 1]

    return np.around(ret_w_state, 4)


def werner(dim: int, alpha: Union[float, List[float]]) -> np.ndarray:
    r"""
    Produce a Werner state [Wer89]_.

    A Werner state is a state of the following form

    .. math::

        \begin{equation}
            \rho_{\alpha} = \frac{1}{d^2 - d\alpha} \left(\mathbb{I} \otimes
            \mathbb{I} - \alpha S \right) \in \mathbb{C}^d \otimes \mathbb{C}^d
        \end{equation}

    Yields a Werner state with parameter `alpha` acting on `(dim * dim)`-
    dimensional space. More specifically, `rho` is the density operator
    defined by (I - `alpha`*S) (normalized to have trace 1), where I is the
    density operator and S is the operator that swaps two copies of
    `dim`-dimensional space (see swap and swap_operator for example).

    If `alpha` is a vector with p!-1 entries, for some integer p > 1, then a
    multipartite Werner state is returned. This multipartite Werner state is
    the normalization of I - `alpha(1)*P(2)` - ... - `alpha(p!-1)*P(p!)`, where
    P(i) is the operator that permutes p subsystems according to the i-th
    permutation when they are written in lexicographical order (for example,
    the lexicographical ordering when p = 3 is:
    `[1, 2, 3], [1, 3, 2], [2, 1,3], [2, 3, 1], [3, 1, 2], [3, 2, 1],`

    so P(4) in this case equals permutation_operator(dim, [2, 3, 1]).

    Examples
    ==========

    Computing the qutrit Werner state with :math:`\alpha = 1/2` can be done in
    `toqito` as

    >>> from toqito.states import werner
    >>> werner(3, 1 / 2)
    [[ 0.06666667,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.13333333,  0.        , -0.06666667,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.13333333,  0.        ,  0.        ,
       0.        , -0.06666667,  0.        ,  0.        ],
     [ 0.        , -0.06666667,  0.        ,  0.13333333,  0.        ,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.06666667,
       0.        ,  0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.13333333,  0.        , -0.06666667,  0.        ],
     [ 0.        ,  0.        , -0.06666667,  0.        ,  0.        ,
       0.        ,  0.13333333,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.06666667,  0.        ,  0.13333333,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ,  0.06666667]]

    We may also compute multipartite Werner states in `toqito` as well.

    >>> from toqito.states import werner
    >>> werner(2, [0.01, 0.02, 0.03, 0.04, 0.05])
    [[ 0.12179487,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.12820513,  0.        ,  0.        , -0.00641026,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.12179487,  0.        ,  0.        ,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.12820513,  0.        ,
       0.        , -0.00641026,  0.        ],
     [ 0.        , -0.00641026,  0.        ,  0.        ,  0.12820513,
       0.        ,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.12179487,  0.        ,  0.        ],
     [ 0.        ,  0.        ,  0.        , -0.00641026,  0.        ,
       0.        ,  0.12820513,  0.        ],
     [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       0.        ,  0.        ,  0.12179487]]

    References
    ==========
    .. [Wer89] R. F. Werner.
        Quantum states with Einstein-Podolsky-Rosen correlations admitting a
        hidden-variable model. Phys. Rev. A, 40(8):4277–4281. 1989

    :param dim: The dimension of the Werner state.
    :param alpha: Parameter to specify Werner state.
    :return: A Werner state of dimension `dim`.
    """
    # The total number of permutation operators.
    if isinstance(alpha, float):
        n_fac = 2
    else:
        n_fac = len(alpha) + 1

    # Multipartite Werner state.
    if n_fac > 2:
        # Compute the number of parties from `len(alpha)`.
        n_var = n_fac
        # We won't actually go all the way to `n_fac`.
        for i in range(2, n_fac):
            n_var = n_var // i
            if n_var == i + 1:
                break
            if n_var < i:
                raise ValueError(
                    "InvalidAlpha: The `alpha` vector must contain"
                    " p!-1 entries for some integer p > 1."
                )

        # Done error checking and computing the number of parties -- now
        # compute the Werner state.
        perms = list(itertools.permutations(np.arange(n_var)))
        sorted_perms = np.argsort(perms, axis=1) + 1

        for i in range(2, n_fac):
            rho = np.identity(dim ** n_var) - alpha[i - 1] * permutation_operator(
                dim, sorted_perms[i, :], False, True
            )
        rho = rho / np.trace(rho)
        return rho
    # Bipartite Werner state.
    return (np.identity(dim ** 2) - alpha * swap_operator(dim, True)) / (
        dim * (dim - alpha)
    )
