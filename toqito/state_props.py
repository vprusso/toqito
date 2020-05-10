"""Properties of quantum states."""
from typing import Any, List, Union

import numpy as np

from toqito.matrix_props import is_psd
from toqito.matrices import pauli
from toqito.state_ops import pure_to_mixed
from toqito.state_ops import schmidt_decomposition
from toqito.channels import partial_trace
from toqito.channels import partial_transpose


__all__ = [
    "is_ensemble",
    "is_mixed",
    "is_mub",
    "is_ppt",
    "is_product_vector",
    "is_pure",
    "concurrence",
    "negativity",
    "schmidt_rank",
]


def is_ensemble(states: List[np.ndarray]) -> bool:
    r"""
    Determine if a set of states constitute an ensemble [WatEns18]_.

    An ensemble of quantum states is defined by a function

    .. math::
        \eta : \Gamma \rightarrow \text{Pos}(\mathcal{X})

    that satisfies

    .. math::
        \text{Tr}\left( \sum_{a \in \Gamma} \eta(a) \right) = 1.

    Examples
    ==========

    Consider the following set of matrices

    .. math::
        \eta = \left\{ \rho_0, \rho_1 \right\}

    where

    .. math::
        \rho_0 = \frac{1}{2} \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad
        \rho_1 = \frac{1}{2} \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}.

    The set :math:`\eta` constitutes a valid ensemble

    >>> from toqito.state_props import is_ensemble
    >>> import numpy as np
    >>> rho_0 = np.array([[0.5, 0], [0, 0]])
    >>> rho_1 = np.array([[0, 0], [0, 0.5]])
    >>> states = [rho_0, rho_1]
    >>> is_ensemble(states)
    True

    References
    ==========
    .. [WatEns18] Watrous, John.
        "The theory of quantum information."
        Section: "Ensemble of quantum states".
        Cambridge University Press, 2018.

    :param states: The list of states to check.
    :return: True if states form an ensemble and False otherwise.
    """
    trace_sum = 0
    for state in states:
        trace_sum += np.trace(state)
        # Constraint: All states in ensemble must be positive semidefinite.
        if not is_psd(state):
            return False
    # Constraint: The sum of the traces of all states within the ensemble must
    # be equal to 1.
    return np.allclose(trace_sum, 1)


def is_mixed(state: np.ndarray) -> bool:
    r"""
    Determine if a given quantum state is mixed [WikMix]_.

    A mixed state by definition is a state that is not pure.

    Examples
    ==========

    Consider the following density matrix

    .. math::
        \rho =  \begin{pmatrix}
                    \frac{3}{4} & 0 \\
                    0 & \frac{1}{4}
                \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the rank of $\rho$ yields that the $\rho$ is a mixed state. This
    can be confirmed in `toqito` as follows:

    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mixed
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> is_mixed(rho)
    True

    References
    ==========
    .. [WikMix] Wikipedia: Quantum state - Mixed states
        https://en.wikipedia.org/wiki/Quantum_state#Mixed_states

    :param state: The density matrix representing the quantum state.
    :return: True if state is mixed and False otherwise.
    """
    return not is_pure(state)


def is_mub(vec_list: List[Union[np.ndarray, List[Union[float, Any]]]]) -> bool:
    r"""
    Check if list of vectors constitute a mutually unbiased basis [WikMUB]_.

    We say that two orthonormal bases

    .. math::
        \begin{equation}
            \mathcal{B}_0 = \left\{u_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
            \quad \text{and} \quad
            \mathcal{B}_1 = \left\{v_a: a \in \Sigma \right\} \subset \mathbb{C}^{\Sigma}
        \end{equation}

    are mutually unbiased if and only if
    :math:`\left|\langle u_a, v_b \rangle\right| = 1/\sqrt{\Sigma}`
    for all :math:`a, b \in \Sigma`.

    For :math:`n \in \mathbb{N}`, a set of orthonormal bases :math:`\left\{
    \mathcal{B}_0, \ldots, \mathcal{B}_{n-1} \right\}` are mutually unbiased
    bases if and only if every basis is mutually unbiased with every other
    basis in the set, i.e. :math:`\mathcal{B}_x` is mutually unbiased with
    :math:`\mathcal{B}_x^{\prime}` for all :math:`x \not= x^{\prime}` with
    :math:`x, x^{\prime} \in \Sigma`.

    Examples
    ==========

    MUB of dimension 2.

    For :math:`d=2`, the following constitutes a mutually unbiased basis:

    .. math::
        \begin{equation}
            M_0 = \left\{ |0 \rangle, |1 \rangle \right\}, \\
            M_1 = \left\{ \frac{|0 \rangle + |1 \rangle}{\sqrt{2}},
            \frac{|0 \rangle - |1 \rangle}{\sqrt{2}} \right\}, \\
            M_2 = \left\{ \frac{|0 \rangle i|1 \rangle}{\sqrt{2}},
            \frac{|0 \rangle - i|1 \rangle}{\sqrt{2}} \right\}, \\
        \end{equation}

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mub
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> mub_1 = [e_0, e_1]
    >>> mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), 1 / np.sqrt(2) * (e_0 - e_1)]
    >>> mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), 1 / np.sqrt(2) * (e_0 - 1j * e_1)]
    >>> mubs = [mub_1, mub_2, mub_3]
    >>> is_mub(mubs)
    True

    Non non-MUB of dimension 2.

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import is_mub
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> mub_1 = [e_0, e_1]
    >>> mub_2 = [1 / np.sqrt(2) * (e_0 + e_1), e_1]
    >>> mub_3 = [1 / np.sqrt(2) * (e_0 + 1j * e_1), e_0]
    >>> mubs = [mub_1, mub_2, mub_3]
    >>> is_mub(mubs)
    False

    References
    ==========
    .. [WikMUB] Wikipedia: Mutually unbiased bases
        https://en.wikipedia.org/wiki/Mutually_unbiased_bases

    :param vec_list: The list of vectors to check.
    :return: True if `vec_list` constitutes a mutually unbiased basis, and
             False otherwise.
    """
    if len(vec_list) <= 1:
        raise ValueError("There must be at least two bases provided as input.")

    dim = vec_list[0][0].shape[0]
    for i, _ in enumerate(vec_list):
        for j, _ in enumerate(vec_list):
            for k in range(dim):
                if i != j:
                    if not np.isclose(
                        np.abs(
                            np.inner(
                                vec_list[i][k].conj().T[0], vec_list[j][k].conj().T[0]
                            )
                        )
                        ** 2,
                        1 / dim,
                    ):
                        return False
    return True


def is_ppt(
    mat: np.ndarray, sys: int = 2, dim: Union[int, List[int]] = None, tol: float = None
) -> bool:
    r"""
    Determine whether or not a matrix has positive partial transpose [WikPPT]_.

    Yields either `True` or `False`, indicating that `mat` does or does not
    have positive partial transpose (within numerical error). The variable
    `mat` is assumed to act on bipartite space.

    For shared systems of :math:`2 \otimes 2` or :math:`2 \otimes 3`, the PPT
    criterion serves as a method to determine whether a given state is entangled
    or separable. Therefore, for systems of this size, the return value "True"
    would indicate that the state is separable and a value of "False" would
    indicate the state is entangled.

    Examples
    ==========

    Consider the following matrix

    .. math::
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
        \end{pmatrix}.

    This matrix trivially satisfies the PPT criterion as can be seen using the
    `toqito` package.

    >>> from toqito.state_props import is_ppt
    >>> import numpy as np
    >>> mat = np.identity(9)
    >>> is_ppt(mat)
    True

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}}\left( |01 \rangle + |10 \rangle \right)

    For the density matrix :math:`\rho = u u^*`, as this is an entangled state
    of dimension :math:`2`, it will violate the PPT criterion, which can be seen
    using the `toqito` package.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_ppt
    >>> rho = bell(2) * bell(2).conj().T
    >>> is_ppt(rho)
    False

    References
    ==========
    .. [WikPPT] Quantiki: Positive partial transpose
        https://www.quantiki.org/wiki/positive-partial-transpose

    :param mat: A square matrix.
    :param sys: Scalar or vector indicating which subsystems the transpose
                should be applied on.
    :param dim: The dimension is a vector containing the dimensions of the
                subsystems on which `mat` acts.
    :param tol: Tolerance with which to check whether `mat` is PPT.
    :return: True if `mat` is PPT and False if not.
    """
    eps = np.finfo(float).eps

    sqrt_rho_dims = np.round(np.sqrt(list(mat.shape)))

    if dim is None:
        dim = np.array(
            [[sqrt_rho_dims[0], sqrt_rho_dims[0]], [sqrt_rho_dims[1], sqrt_rho_dims[1]]]
        )
    if tol is None:
        tol = np.sqrt(eps)
    return is_psd(partial_transpose(mat, sys, dim), tol)


def _is_product_vector(
    vec: np.ndarray, dim: Union[int, List[int]] = None
) -> [int, bool]:
    """
    Determine if a given vector is a product vector recursive helper.

    :param vec: The vector to check.
    :param dim: The dimension of the vector
    :return: True if `vec` is a product vector and False otherwise.
    """
    if dim is None:
        dim = np.round(np.sqrt(len(vec)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, float):
        num_sys = 1
    else:
        num_sys = len(dim)

    if num_sys == 1:
        dim = np.array([dim, len(vec) // dim])
        dim[1] = np.round(dim[1])
        num_sys = 2

    dec = 0
    # If there are only two subsystems, just use the Schmidt decomposition.
    if num_sys == 2:
        singular_vals, u_mat, vt_mat = schmidt_decomposition(vec, dim, 2)
        ipv = singular_vals[1] <= np.prod(dim) * np.spacing(singular_vals[0])

        # Provide this even if not requested, since it is needed if this
        # function was called as part of its recursive algorithm (see below)
        if ipv:
            u_mat = u_mat * np.sqrt(singular_vals[0])
            vt_mat = vt_mat * np.sqrt(singular_vals[0])
            dec = [u_mat[:, 0], vt_mat[:, 0]]
    else:
        new_dim = [dim[0] * dim[1]]
        new_dim.extend(dim[2:])
        ipv, dec = _is_product_vector(vec, new_dim)
        if ipv:
            ipv, tdec = _is_product_vector(dec[0], [dim[0], dim[1]])
            if ipv:
                dec = [tdec, dec[1:]]

    return ipv, dec


def is_product_vector(vec: np.ndarray, dim: Union[int, List[int]] = None) -> bool:
    r"""
    Determine if a given vector is a product vector [4]_.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the rank of :math:`\rho` yields that the :math:`\rho` is a pure
    state. This can be confirmed in `toqito` as follows:

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u = bell(0)
    >>> rho = u * u.conj().T
    >>> is_pure(rho)
    True

    It is also possible to determine whether a set of density matrices are pure.
    For instance, we can see that the density matrices corresponding to the four
    Bell states yield a result of `True` indicating that all states provided to
    the function are pure.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u0, u1, u2, u3 = bell(0), bell(1), bell(2), bell(3)
    >>> rho0 = u0 * u0.conj().T
    >>> rho1 = u1 * u1.conj().T
    >>> rho2 = u2 * u2.conj().T
    >>> rho3 = u3 * u3.conj().T
    >>>
    >>> is_pure([rho0, rho1, rho2, rho3])
    True

    References
    ==========
    .. [4] Wikipedia: Quantum state - Pure states
        https://en.wikipedia.org/wiki/Quantum_state#Pure_states

    :param vec: The vector to check.
    :param dim: The dimension of the vector
    :return: True if `vec` is a product vector and False otherwise.
    """
    return _is_product_vector(vec, dim)[0][0]


def is_pure(state: Union[List[np.ndarray], np.ndarray]) -> bool:
    r"""
    Determine if a given state is pure or list of states are pure [WikIsPure]_.

    A state is said to be pure if it is a density matrix with rank equal to
    1. Equivalently, the state :math: `\rho` is pure if there exists a unit
    vector :math: `u` such that:

    ..math::
        \rho = u u^*

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the rank of :math:`\rho` yields that the :math:`\rho` is a pure
    state. This can be confirmed in `toqito` as follows:

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u = bell(0)
    >>> rho = u * u.conj().T
    >>> is_pure(rho)
    True

    It is also possible to determine whether a set of density matrices are pure.
    For instance, we can see that the density matrices corresponding to the four
    Bell states yield a result of `True` indicating that all states provided to
    the function are pure.

    >>> from toqito.states import bell
    >>> from toqito.state_props import is_pure
    >>> u0, u1, u2, u3 = bell(0), bell(1), bell(2), bell(3)
    >>> rho0 = u0 * u0.conj().T
    >>> rho1 = u1 * u1.conj().T
    >>> rho2 = u2 * u2.conj().T
    >>> rho3 = u3 * u3.conj().T
    >>>
    >>> is_pure([rho0, rho1, rho2, rho3])
    True

    References
    ==========
    .. [WikIsPure] Wikipedia: Quantum state - Pure states
        https://en.wikipedia.org/wiki/Quantum_state#Pure_states

    :param state: The density matrix representing the quantum state or a list
                  of density matrices representing quantum states.
    :return: True if state is pure and False otherwise.
    """
    # Allow the user to enter a list of states to check.
    if isinstance(state, list):
        for rho in state:
            eigs, _ = np.linalg.eig(rho)
            if not np.allclose(np.max(np.diag(eigs)), 1):
                return False
        return True

    # Otherwise, if the user just put in a single state, check that.
    eigs, _ = np.linalg.eig(state)
    return np.allclose(np.max(np.diag(eigs)), 1)


def concurrence(rho: np.ndarray) -> float:
    r"""
    Calculate the concurrence of a bipartite state [WikCon]_.

    The concurrence of a bipartite state :math:`\rho` is defined as

    .. math::
        \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4),

    where :math:`\lambda_1, \ldots, \lambda_4` are the eigenvalues in
    decreasing order of the matrix.

    Concurrence can serve as a measure of entanglement.

    Examples
    ==========

    Consider the following Bell state:

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    The concurrence of the density matrix :math:`\rho = u u^*` defined by the
    vector :math:`u` is given as

    .. math::
        \mathcal{C}(\rho) \approx 1.

    The following example calculates this quantity using the `toqito` package.

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import concurrence
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00, e_11 = np.kron(e_0, e_0), np.kron(e_1, e_1)
    >>> u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    >>> rho = u_vec * u_vec.conj().T
    >>> concurrence(rho)
    0.9999999999999998

    Consider the concurrence of the following product state

    .. math::
        v = |0\rangle \otimes |1 \rangle.

    As this state has no entanglement, the concurrence is zero

    >>> import numpy as np
    >>> from toqito.states import basis
    >>> from toqito.state_props import concurrence
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> v_vec = np.kron(e_0, e_1)
    >>> sigma = v_vec * v_vec.conj().T
    >>> concurrence(sigma)
    0

    References
    ==========
    .. [WikCon] Wikipedia page for concurrence (quantum computing)
       https://en.wikipedia.org/wiki/Concurrence_(quantum_computing)

    :param rho: The bipartite system specified as a matrix.
    :return: The concurrence of the bipartite state :math:`\rho`.
    """
    if rho.shape != (4, 4):
        raise ValueError(
            "InvalidDim: Concurrence is only defined for bipartite systems."
        )

    sigma_y = pauli("Y", False)
    sigma_y_y = np.kron(sigma_y, sigma_y)

    rho_hat = np.matmul(np.matmul(sigma_y_y, rho.conj().T), sigma_y_y)

    eig_vals = np.linalg.eigvalsh(np.matmul(rho, rho_hat))
    eig_vals = np.sort(np.sqrt(eig_vals))[::-1]
    return max(0, eig_vals[0] - eig_vals[1] - eig_vals[2] - eig_vals[3])


def negativity(rho: np.ndarray, dim: Union[List[int], int] = None) -> float:
    r"""
    Compute the negativity of a bipartite quantum state [WikNeg]_.

    The negativity of a subsystem can be defined in terms of a density matrix
    :math:`\rho`:

    .. math::
        \mathcal{N}(\rho) \equiv \frac{||\rho^{\Gamma_A}||_1-1}{2}

    Calculate the negativity of the quantum state `rho`, assuming that the two
    subsystems on which `rho` acts are of equal dimension (if the local
    dimensions are unequal, specify them in the optional `dim` argument). The
    negativity of `rho` is the sum of the absolute value of the negative
    eigenvalues of the partial transpose of `rho`.

    Examples
    ==========

    Example of the negativity of density matrix of Bell state.

    >>> from toqito.states import bell
    >>> from toqito.state_props import negativity
    >>> rho = bell(0) * bell(0).conj().T
    >>> negativity(rho)
    0.4999999999999998

    References
    ==========
    .. [WikNeg] Wikipedia page for negativity (quantum mechanics):
        https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics)

    :param rho: A density matrix of a pure state vector.
    :param dim: The default has both subsystems of equal dimension.
    :return: A value between 0 and 1 that corresponds to the negativity of
            :math:`\rho`.
    """
    # Allow the user to input either a pure state vector or a density matrix.
    rho = pure_to_mixed(rho)
    rho_dims = rho.shape
    round_dim = np.round(np.sqrt(rho_dims))

    if dim is None:
        dim = np.array([round_dim])
        dim = dim.T
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, int):
        dim = np.array([dim, rho_dims[0] / dim])
        if abs(dim[1] - np.round(dim[1])) >= 2 * rho_dims[0] * np.finfo(float).eps:
            raise ValueError(
                "InvalidDim: If `dim` is a scalar, `rho` must be "
                "square and `dim` must evenly divide `len(rho)`. "
                "Please provide the `dim` array containing the "
                "dimensions of the subsystems."
            )
        dim[1] = np.round(dim[1])

    if np.prod(dim) != rho_dims[0]:
        raise ValueError(
            "InvalidDim: Please provide local dimensions in the "
            "argument `dim` that match the size of `rho`."
        )

    # Compute the negativity.
    return (np.linalg.norm(partial_transpose(rho, 2, dim), ord="nuc") - 1) / 2


def schmidt_rank(
    vec: np.ndarray, dim: Union[int, List[int], np.ndarray] = None
) -> float:
    r"""
    Compute the Schmidt rank [WikSR]_.

    For complex Euclidean spaces :math:`\mathcal{X}` and :math:`\mathcal{Y}`, a
    pure state :math:`u \in \mathcal{X} \otimes \mathcal{Y}` possesses an
    expansion of the form:

    .. math::
        u = \sum_{i} \lambda_i v_i w_i

    where :math:`v_i \in \mathcal{X}` and :math:`w_i \in \mathcal{Y}` are
    orthonormal states.

    The Schmidt coefficients are calculated from

    .. math::
        A = \text{Tr}_{\mathcal{B}}(u^* u).

    The Schmidt rank is the number of non-zero eigenvalues of A. The Schmidt
    rank allows us to determine if a given state is entangled or separable.
    For instance:

        - If the Schmidt rank is 1: The state is separable
        - If the Schmidt rank > 1: The state is entangled.

    Compute the Schmidt rank of the vector `vec`, assumed to live in bipartite
    space, where both subsystems have dimension equal to `sqrt(len(vec))`.

    The dimension may be specified by the 1-by-2 vector `dim` and the rank in
    that case is determined as the number of Schmidt coefficients larger than
    `tol`.

    Examples
    ==========

    Computing the Schmidt rank of the entangled Bell state should yield a value
    greater than one.

    >>> from toqito.states import bell
    >>> from toqito.state_props import schmidt_rank
    >>> rho = bell(0).conj().T * bell(0)
    >>> schmidt_rank(rho)
    2

    Computing the Schmidt rank of the entangled singlet state should yield a
    value greater than :math:`1`.

    >>> from toqito.states import bell
    >>> from toqito.state_props import schmidt_rank
    >>> u = bell(2).conj().T * bell(2)
    >>> schmidt_rank(u)
    2

    Computing the Schmidt rank of a separable state should yield a value equal
    to :math:`1`.

    >>> from toqito.states import basis
    >>> from toqito.state_props import schmidt_rank
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00 = np.kron(e_0, e_0)
    >>> e_01 = np.kron(e_0, e_1)
    >>> e_10 = np.kron(e_1, e_0)
    >>> e_11 = np.kron(e_1, e_1)
    >>>
    >>> rho = 1 / 2 * (e_00 - e_01 - e_10 + e_11)
    >>> rho = rho.conj().T * rho
    >>> schmidt_rank(rho)
    1

    References
    ==========
    .. [WikSR] Wikipedia: Schmidt rank
        https://en.wikipedia.org/wiki/Schmidt_decomposition#Schmidt_rank_and_entanglement

    :param vec: A bipartite vector to have its Schmidt rank computed.
    :param dim: A 1-by-2 vector.
    :return: The Schmidt rank of vector `vec`.
    """
    eps = np.finfo(float).eps
    slv = int(np.round(np.sqrt(len(vec))))

    if dim is None:
        dim = slv

    if isinstance(dim, int):
        dim = np.array([dim, len(vec) / dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            raise ValueError(
                "Invalid: The value of `dim` must evenly divide "
                "`len(vec)`; please provide a `dim` array "
                "containing the dimensions of the subsystems"
            )
        dim[1] = np.round(dim[1])

    rho = vec.conj().T * vec
    rho_a = partial_trace(rho, 2)

    # Return the number of non-zero eigenvalues of the
    # matrix that traced out the second party's portion.
    return len(np.nonzero(np.linalg.eigvalsh(rho_a))[0])
