"""Distance metrics for quantum states."""
import cvxpy
import scipy
import numpy as np

from toqito.matrix_props import is_density

__all__ = [
    "fidelity",
    "helstrom_holevo",
    "hilbert_schmidt",
    "purity",
    "sub_fidelity",
    "trace_distance",
    "trace_norm",
    "von_neumann_entropy",
]


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the fidelity of two density matrices [WikFid]_.

    Calculate the fidelity between the two density matrices :code:`rho` and
    :code:`sigma`, defined by:

    .. math::
        ||\sqrt(\rho) * \sqrt(\sigma)||_1s

    where :math:`|| \cdot ||_1` denotes the trace norm. The return is a value
    between 0 and 1, with 0 corresponding to matrices :code:`rho` and
    :code:`sigma` with orthogonal support, and 1 corresponding to the case
    :code:`rho = sigma`.

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
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    In the event where we calculate the fidelity between states that are
    identical, we should obtain the value of :math:`1`. This can be observed in
    :code:`toqito` as follows.

    >>> from toqito.state_metrics import fidelity
    >>> import numpy as np
    >>> rho = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> sigma = rho
    >>> fidelity(rho, sigma)
    1.0000000000000002

    References
    ==========
    .. [WikFid] Wikipedia: Fidelity of quantum states
        https://en.wikipedia.org/wiki/Fidelity_of_quantum_states

    :param rho: Density operator.
    :param sigma: Density operator.
    :return: The fidelity between `rho` and `sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError(
            "InvalidDim: `rho` and `sigma` must be matrices of the" " same size."
        )

    # If `rho` or `sigma` is a cvxpy variable then compute fidelity via
    # semidefinite programming, so that this function can be used in the
    # objective function or constraints of other cvxpy optimization problems.
    if isinstance(rho, cvxpy.atoms.affine.vstack.Vstack) or isinstance(
        sigma, cvxpy.atoms.affine.vstack.Vstack
    ):
        z_var = cvxpy.Variable(rho.shape, complex=True)
        objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(z_var + z_var.H)))
        constraints = [cvxpy.bmat([[rho, z_var], [z_var.H, sigma]]) >> 0]
        problem = cvxpy.Problem(objective, constraints)

        return 1 / 2 * problem.solve()

    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Fidelity is only defined for density operators.")

    # If `rho` or `sigma` are *not* cvxpy variables, compute fidelity normally,
    # since this is much faster.
    sq_rho = scipy.linalg.sqrtm(rho)
    sq_fid = scipy.linalg.sqrtm(np.matmul(np.matmul(sq_rho, sigma), sq_rho))
    return np.real(np.trace(sq_fid))


def helstrom_holevo(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Helstrom-Holevo distance between density matrices.

    In general, the best success probability to discriminate
    two mixed states represented by :math:`\rho` and :math:`\sigma` is given
    by [WikHeHo]_.

    In general, the best success probability to discriminate two mixed states
    represented by :math:`\rho` and :math:`\sigma` is given by

    .. math::
         \frac{1}{2}+\frac{1}{2} \left(\frac{1}{2} \left|\rho - \sigma
         \right|_1\right).

    Examples
    ==========
    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the Helstrom-Holevo distance of states that are identical yield
    a value of :math:`1/2`. This can be verified in `toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.state_metrics import helstrom_holevo
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00 = np.kron(e_0, e_0)
    >>> e_11 = np.kron(e_1, e_1)
    >>>
    >>> u_vec = 1 / np.sqrt(2) * (e_00 + e_11)
    >>> rho = u_vec * u_vec.conj().T
    >>> sigma = rho
    >>>
    >>> helstrom_holevo(rho, sigma)
    0.5

    References
    ==========
    .. [WikHeHo] Wikipedia: Holevo's theorem.
        https://en.wikipedia.org/wiki/Holevo%27s_theorem

    :param rho: Density operator.
    :param sigma: Density operator.
    :return: The Helstrom-Holevo distance between `rho` and `sigma`.
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Helstrom-Holevo is only defined for density " "operators.")
    return 1 / 2 + 1 / 2 * (trace_norm(rho - sigma)) / 2


def hilbert_schmidt(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the Hilbert-Schmidt distance between two states [WikHS]_.

    The Hilbert-Schmidt distance between density operators :math:`\rho` and
    :math:`\sigma` is defined as

    .. math::
        D_{\text{HS}}(\rho, \sigma) = \text{Tr}((\rho - \sigma)^2) =
        \left\lVert \rho - \sigma \right\rVert_2^2.

    Examples
    ==========

    One may consider taking the Hilbert-Schmidt distance between two Bell
    states. In `toqito`, one may accomplish this as

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import hilbert_schmidt
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = bell(3) * bell(3).conj().T
    >>> hilbert_schmidt(rho, sigma)
    1

    References
    ==========
    .. [WikHS] Wikipedia: Hilbert-Schmidt operator.
        https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_operator

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The Hilbert-Schmidt distance between `rho` and `sigma`.
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Hilbert-Schmidt is only defined for density " "operators.")
    return np.linalg.norm(rho - sigma, ord=2) ** 2


def purity(rho: np.ndarray) -> float:
    r"""
    Compute the purity of a quantum state [WikPurity]_.

    Examples
    ==========

    Consider the following scaled state defined as the scaled identity matrix

    .. math::
        \rho = \frac{1}{4} \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 1 & 0 & 0 \\
                         0 & 0 & 1 & 0 \\
                         0 & 0 & 0 & 1
                       \end{pmatrix} \in \text{D}(\mathcal{X}).

    Calculating the purity of :math:`\rho` yields :math:`\frac{1}{4}`. This can
    be observed using `toqito` as follows.

    >>> from toqito.state_metrics import purity
    >>> import numpy as np
    >>> purity(np.identity(4) / 4)
    0.25

    References
    ==========
    .. [WikPurity] Wikipedia: Purity (quantum mechanics)
        https://en.wikipedia.org/wiki/Purity_(quantum_mechanics)

    :param rho: Density operator.
    :return: The purity of the quantum state `rho` (i.e., `gamma` is the)
             quantity `np.trace(rho**2)`.
    """
    if not is_density(rho):
        raise ValueError("Purity is only defined for density operators.")
    # "np.real" get rid of the close-to-0 imaginary part.
    return np.real(np.trace(rho ** 2))


def sub_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the sub fidelity of two density matrices [MPHUZSub08]_.

    The sub-fidelity is a measure of similarity between density operators.
    It is defined as

    .. math::
        E(\rho, \sigma) = \text{Tr}(\rho \sigma) + \sqrt{2
        \left[ \text{Tr}(\rho \sigma)^2 - \text{Tr}(\rho \sigma \rho \sigma)
        \right]},

    where :math:`\sigma` and :math:`\rho` are density matrices. The
    sub-fidelity serves as an lower bound for the fidelity.

    Examples
    ==========

    Consider the following pair of states

    .. math::
        \rho = \frac{3}{4}|0\rangle \langle 0| +
               \frac{1}{4}|1 \rangle \langle 1|
        \sigma = \frac{1}{8}|0 \rangle \langle 0| +
                 \frac{7}{8}|1 \rangle \langle 1|.

    Calculating the fidelity between the states :math:`\rho` and :math:`\sigma`
    as :math:`F(\rho, \sigma) \approx 0.774`. This can be observed in `toqito`
    as

    >>> from toqito.states import basis
    >>> from toqito.state_metrics import fidelity
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> sigma = 1/8 * e_0 * e_0.conj().T + 7/8 * e_1 * e_1.conj().T
    >>> fidelity(rho, sigma)
    0.77389339119464

    As the sub-fidelity is a lower bound on the fidelity, that is
    :math:`E(\rho, \sigma) \leq F(\rho, \sigma)`, we can use `toqito` to observe
    that :math:`E(\rho, \sigma) \approx 0.599\leq F(\rho, \sigma \approx 0.774`.

    >>> from toqito.states import basis
    >>> from toqito.state_metrics import sub_fidelity
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> rho = 3 / 4 * e_0 * e_0.conj().T + 1 / 4 * e_1 * e_1.conj().T
    >>> sigma = 1/8 * e_0 * e_0.conj().T + 7/8 * e_1 * e_1.conj().T
    >>> sub_fidelity(rho, sigma)
    0.5989109809347399

    References
    ==========
    .. [MPHUZSub08] J. A. Miszczak, Z. Puchała, P. Horodecki, A. Uhlmann, K. Życzkowski
        "Sub--and super--fidelity as bounds for quantum fidelity."
        arXiv preprint arXiv:0805.2037 (2008).
        https://arxiv.org/abs/0805.2037

    :param rho: Density operator.
    :param sigma: Density operator.
    :return: The sub-fidelity between `rho` and `sigma`.
    """
    # Perform some error checking.
    if not np.all(rho.shape == sigma.shape):
        raise ValueError(
            "InvalidDim: `rho` and `sigma` must be matrices of the same size."
        )
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Sub-fidelity is only defined for density operators.")

    return np.real(
        np.trace(rho * sigma)
        + np.sqrt(
            2 * (np.trace(rho * sigma) ** 2 - np.trace(rho * sigma * rho * sigma))
        )
    )


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Compute the trace distance between density operators `rho` and `sigma`.

    The trace distance between :math:`\rho` and :math:`\sigma` is defined as

    .. math::
        \delta(\rho, \sigma) = \frac{1}{2} \left( \text{Tr}(\left| \rho - \sigma
         \right| \right).

    More information on the trace distance can be found in [WIKTD]_.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    The trace distance between :math:`\rho` and another state :math:`\sigma` is
    equal to :math:`0` if any only if :math:`\rho = \sigma`. We can check this
    using the `toqito` package.

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import trace_norm
    >>> rho = bell(0) * bell(0).conj().T
    >>> sigma = rho
    >>> trace_distance(rho, sigma)
    0.0

    References
    ==========
    .. [WIKTD] Quantiki: Trace distance
            https://www.quantiki.org/wiki/trace-distance

    :param rho: An input matrix.
    :param sigma: An input matrix.
    :return: The trace distance between `rho` and `sigma`.
    """
    if not is_density(rho) or not is_density(sigma):
        raise ValueError("Trace distance is only defined for density " "operators.")
    return trace_norm(np.abs(rho - sigma)) / 2


def trace_norm(rho: np.ndarray) -> float:
    r"""
    Compute the trace norm of the matrix `rho` [WikTn]_.

    The trace norm :math:`||\rho||_1` of a density matrix :math:`\rho` is the
    sum of the singular values of :math:`\rho`. The singular values are the
    roots of the eigenvalues of :math:`\rho \rho^*`.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of :math:`u` may be calculated by:

    .. math::
        \rho = u u^* = \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    It can be observed using `toqito` that :math:`||\rho||_1 = 1` as follows.

    >>> from toqito.states import bell
    >>> from toqito.state_metrics import trace_norm
    >>> rho = bell(0) * bell(0).conj().T
    >>> trace_norm(rho)
    0.9999999999999999

    References
    ==========
    .. [WikTn] Quantiki: Trace norm
        https://www.quantiki.org/wiki/trace-norm

    :param rho: Density operator.
    :return: The trace norm of `rho`.
    """
    return np.linalg.norm(rho, ord="nuc")


def von_neumann_entropy(rho: np.ndarray) -> float:
    r"""
    Compute the von Neumann entropy of a density matrix [WikVent]_. [WatVec]_.

    Let :math:`P \in \text{Pos}(\mathcal{X})` be a positive semidefinite
    operator, for a complex Euclidean space :math:`\mathcal{\X}`. Then one
    defines the *von Neumann entropy* as

    .. math::
        H(P) = H(\lambda(P)),

    where :math:`\lambda(P)` is the vector of eigenvalues of :math:`P` and where
    the function `H(\dot)` is the Shannon entropy function defined as

    .. math::
        H(u) = - \sum_{\substack{a \in \Sigma \\ u(a) > 0}} u(a) \text{log}(u(a)).

    where the :math:`\text{log}` function is assumed to be the base-2 logarithm,
    and where :math:`\Sigma` is an alphabet where :math:`u \in [0, \infty`)^{\Sigma}
    is a vector of nonnegative real numbers indexed by :math:`\Sigma`.

    Examples
    ==========

    Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left(e_0 \otimes e_0 + e_1 \otimes e_1 \right)
        \in \mathcal{X}.

    The corresponding density matrix of $u$ may be calculated by:

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                         1 & 0 & 0 & 1 \\
                         0 & 0 & 0 & 0 \\
                         0 & 0 & 0 & 0 \\
                         1 & 0 & 0 & 1
                       \end{pmatrix} \text{D}(\mathcal{X}).

    Calculating the von Neumann entropy of :math:`\rho` in `toqito` can be done
    as follows.

    >>> from toqito.state_metrics import von_neumann_entropy
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0],
    >>>      [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> von_neumann_entropy(test_input_mat)
    5.88418203051333e-15

    Consider the density operator corresponding to the maximally mixed state of
    dimension two

    .. math::
        \rho = \frac{1}{2}
        \begin{pmatrix}
            1 & 0 \\
            0 & 1
        \end{pmatrix}.

    As this state is maximally mixed, the von Neumann entropy of :math:`\rho` is
    equal to one. We can see this in `toqito` as follows.

    >>> from toqito.state_metrics import von_neumann_entropy
    >>> import numpy as np
    >>> rho = 1/2 * np.identity(2)
    >>> von_neumann_entropy(rho)
    1.0

    References
    ==========
    .. [WikVent] Wikipedia: Von Neumann entropy
        https://en.wikipedia.org/wiki/Von_Neumann_entropy

    .. [WatVec] Watrous, John.
        "The theory of quantum information."
        Section: "Definitions of quantum entropic functions".
        Cambridge University Press, 2018.

    :param rho: Density operator.
    :return: The von Neumann entropy of `rho`.
    """
    if not is_density(rho):
        raise ValueError(
            "Von Neumann entropy is only defined for density " "operators."
        )
    eigs, _ = np.linalg.eig(rho)
    eigs = [eig for eig in eigs if eig > 0]
    return -np.sum(np.real(eigs * np.log2(eigs)))
